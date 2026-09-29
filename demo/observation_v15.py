"""Canonical per-step heterogeneous signed graph snapshots for v15.

V15 keeps the v14 node features, simulator, reward and candidate action set,
but replaces the five v14 relations with six binary relations.  Every edge is
limited to the physical 3 km communication radius.  Cooperation edges connect
providers to active EVs; competition edges connect nearby providers and nearby
active EVs.

The six binary matrices are bit-packed before they enter the PPO rollout.  In
particular, the 300 x 300 EV competition matrix is not stored as 90,000
float32 values in every Low transition.  The neural encoder losslessly decodes
the bits to sparse edge lists before attention message passing.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict

import numpy as np

from config import COMM_RANGE
from core import euclidean_distance
from observation import ObservationBuilder
from observation_v14 import (
    BASE_CANDIDATE_DIM,
    CANDIDATE_DISTANCE_RATIO,
    CANDIDATE_ENERGY_MARGIN,
    CANDIDATE_EV_INDEX,
    CANDIDATE_IS_STAY,
    EV_NODE_COUNT,
    EV_NODE_DIM,
    FCS_NODE_COUNT,
    FCS_NODE_DIM,
    MCS_NODE_COUNT,
    MCS_NODE_DIM,
    V14_CANDIDATE_DIM,
    GlobalGraphSnapshotBuilder,
    ObservationBuilderV14,
)


V15_CANDIDATE_DIM = V14_CANDIDATE_DIM


@dataclass(frozen=True)
class SignedGraphSnapshotLayout:
    """Fixed-width node features followed by bit-packed binary edges."""

    mcs_features: int = MCS_NODE_COUNT * MCS_NODE_DIM
    fcs_features: int = FCS_NODE_COUNT * FCS_NODE_DIM
    ev_features: int = EV_NODE_COUNT * EV_NODE_DIM
    mcs_ev_positive_bits: int = MCS_NODE_COUNT * EV_NODE_COUNT
    fcs_ev_positive_bits: int = FCS_NODE_COUNT * EV_NODE_COUNT
    mcs_mcs_negative_bits: int = MCS_NODE_COUNT * (MCS_NODE_COUNT - 1) // 2
    mcs_fcs_negative_bits: int = MCS_NODE_COUNT * FCS_NODE_COUNT
    fcs_fcs_negative_bits: int = FCS_NODE_COUNT * (FCS_NODE_COUNT - 1) // 2
    ev_ev_negative_bits: int = EV_NODE_COUNT * (EV_NODE_COUNT - 1) // 2

    @property
    def node_feature_dim(self) -> int:
        return self.mcs_features + self.fcs_features + self.ev_features

    @property
    def edge_bit_count(self) -> int:
        return (
            self.mcs_ev_positive_bits
            + self.fcs_ev_positive_bits
            + self.mcs_mcs_negative_bits
            + self.mcs_fcs_negative_bits
            + self.fcs_fcs_negative_bits
            + self.ev_ev_negative_bits
        )

    @property
    def edge_byte_count(self) -> int:
        return (self.edge_bit_count + 7) // 8

    @property
    def graph_dim(self) -> int:
        return self.node_feature_dim + self.edge_byte_count

    @property
    def policy_self_dim(self) -> int:
        return 1 + self.graph_dim


V15_LAYOUT = SignedGraphSnapshotLayout()
V15_SELF_DIM = V15_LAYOUT.policy_self_dim


EDGE_RELATIONS = (
    ('mcs_ev_positive', '+', V15_LAYOUT.mcs_ev_positive_bits),
    ('fcs_ev_positive', '+', V15_LAYOUT.fcs_ev_positive_bits),
    ('mcs_mcs_negative', '-', V15_LAYOUT.mcs_mcs_negative_bits),
    ('mcs_fcs_negative', '-', V15_LAYOUT.mcs_fcs_negative_bits),
    ('fcs_fcs_negative', '-', V15_LAYOUT.fcs_fcs_negative_bits),
    ('ev_ev_negative', '-', V15_LAYOUT.ev_ev_negative_bits),
)


FEATURE_SCHEMA = {
    'version': 1,
    'scope': 'one_canonical_global_signed_graph_per_environment_step',
    'edge_encoding': 'lossless_little_endian_bit_packing',
    'edge_values': [-1, 0, 1],
    'communication_range_km': float(COMM_RANGE),
    'global_pooling': False,
    'regional_tokens': False,
    'actor_global_readout': False,
    'reservation': False,
    'soft_occupancy': False,
    'node_counts': {
        'mcs': MCS_NODE_COUNT,
        'fcs': FCS_NODE_COUNT,
        'ev': EV_NODE_COUNT,
    },
    'node_features_identical_to_v14': True,
    'active_masks': {
        'mcs': 'not_broken_and_not_energy_stranded',
        'fcs': 'all_physical_stations',
        'ev': 'quasi_or_iev',
    },
    'positive_relations': ['mcs_ev', 'fcs_ev'],
    'negative_relations': [
        'mcs_mcs', 'mcs_fcs', 'fcs_fcs', 'ev_ev',
    ],
    'edge_rule': 'binary_edge_if_and_only_if_distance_at_most_3km',
    'candidate_policy_fields': [
        'ev_node_index', 'is_stay', 'distance_ratio',
        'mcs_energy_margin_ratio',
    ],
    'legacy_candidate_fields_policy_visible': False,
}


def _pairwise_within_range(left_positions, right_positions) -> np.ndarray:
    """Vectorized haversine mask with the simulator's exact 3 km rule."""
    left = np.asarray(left_positions, dtype=np.float64)
    right = np.asarray(right_positions, dtype=np.float64)
    if not left.size or not right.size:
        return np.zeros((len(left), len(right)), dtype=bool)
    left_lon = np.radians(left[:, 0])[:, None]
    left_lat = np.radians(left[:, 1])[:, None]
    right_lon = np.radians(right[:, 0])[None, :]
    right_lat = np.radians(right[:, 1])[None, :]
    delta_lon = right_lon - left_lon
    delta_lat = right_lat - left_lat
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.sin(delta_lon / 2.0) ** 2
        * np.cos(left_lat) * np.cos(right_lat)
    )
    distance_km = 2.0 * 6371.0 * np.arcsin(
        np.sqrt(np.clip(haversine, 0.0, 1.0))
    )
    return distance_km <= float(COMM_RANGE)


class GlobalSignedGraphSnapshotBuilder(GlobalGraphSnapshotBuilder):
    """Build the losslessly packed v15 signed graph once per step."""

    def _relations(self):
        mcss = list(self.world.MCSs[:MCS_NODE_COUNT])
        fcss = list(self.world.FCSs[:FCS_NODE_COUNT])
        evs = list(self.world.EVs[:EV_NODE_COUNT])
        if (
            len(mcss) != MCS_NODE_COUNT
            or len(fcss) != FCS_NODE_COUNT
            or len(evs) != EV_NODE_COUNT
        ):
            raise RuntimeError('v15 requires the configured fixed node counts')
        active_mcs = np.asarray([
            not bool(mcs.is_broken)
            and not bool(getattr(mcs, 'is_energy_stranded', False))
            for mcs in mcss
        ], dtype=bool)
        active_ev = np.asarray([
            bool(ev.is_quasi or ev.is_iev) for ev in evs
        ], dtype=bool)
        mcs_positions = [mcs.pos for mcs in mcss]
        fcs_positions = [fcs.pos for fcs in fcss]
        ev_positions = [ev.pos for ev in evs]

        mcs_ev = _pairwise_within_range(mcs_positions, ev_positions)
        mcs_ev &= active_mcs[:, None] & active_ev[None, :]
        fcs_ev = _pairwise_within_range(fcs_positions, ev_positions)
        fcs_ev &= active_ev[None, :]
        mm = _pairwise_within_range(mcs_positions, mcs_positions)
        mm &= active_mcs[:, None] & active_mcs[None, :]
        np.fill_diagonal(mm, False)
        mf = _pairwise_within_range(mcs_positions, fcs_positions)
        mf &= active_mcs[:, None]
        ff = _pairwise_within_range(fcs_positions, fcs_positions)
        np.fill_diagonal(ff, False)
        ee = _pairwise_within_range(ev_positions, ev_positions)
        ee &= active_ev[:, None] & active_ev[None, :]
        np.fill_diagonal(ee, False)

        return mcs_ev, fcs_ev, mm, mf, ff, ee, active_mcs, active_ev

    @staticmethod
    def _edge_bits(mcs_ev, fcs_ev, mm, mf, ff, ee) -> np.ndarray:
        mcs_tri = np.triu_indices(MCS_NODE_COUNT, 1)
        fcs_tri = np.triu_indices(FCS_NODE_COUNT, 1)
        ev_tri = np.triu_indices(EV_NODE_COUNT, 1)
        bits = np.concatenate((
            mcs_ev.reshape(-1),
            fcs_ev.reshape(-1),
            mm[mcs_tri],
            mf.reshape(-1),
            ff[fcs_tri],
            ee[ev_tri],
        )).astype(np.uint8)
        if bits.size != V15_LAYOUT.edge_bit_count:
            raise RuntimeError(
                f'v15 edge bit count {bits.size} != '
                f'{V15_LAYOUT.edge_bit_count}'
            )
        return bits

    @staticmethod
    def _topology_metrics(
        mcs_ev, fcs_ev, mm, mf, ff, ee, active_mcs, active_ev
    ) -> dict[str, float]:
        node_count = MCS_NODE_COUNT + FCS_NODE_COUNT + EV_NODE_COUNT
        adjacency = np.zeros((node_count, node_count), dtype=bool)
        mcs_end = MCS_NODE_COUNT
        fcs_end = mcs_end + FCS_NODE_COUNT
        adjacency[:mcs_end, fcs_end:] = mcs_ev
        adjacency[fcs_end:, :mcs_end] = mcs_ev.T
        adjacency[mcs_end:fcs_end, fcs_end:] = fcs_ev
        adjacency[fcs_end:, mcs_end:fcs_end] = fcs_ev.T
        adjacency[:mcs_end, :mcs_end] = mm
        adjacency[:mcs_end, mcs_end:fcs_end] = mf
        adjacency[mcs_end:fcs_end, :mcs_end] = mf.T
        adjacency[mcs_end:fcs_end, mcs_end:fcs_end] = ff
        adjacency[fcs_end:, fcs_end:] = ee

        valid = np.concatenate((
            active_mcs,
            np.ones(FCS_NODE_COUNT, dtype=bool),
            active_ev,
        ))
        valid_indices = np.flatnonzero(valid)
        degree = adjacency.sum(axis=1).astype(np.float64)
        valid_degree = degree[valid_indices]

        parent = np.arange(node_count, dtype=np.int32)

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = int(parent[value])
            return value

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left, right in np.argwhere(np.triu(adjacency, 1)):
            if valid[int(left)] and valid[int(right)]:
                union(int(left), int(right))
        component_sizes: dict[int, int] = {}
        for node in valid_indices:
            root = find(int(node))
            component_sizes[root] = component_sizes.get(root, 0) + 1

        positive_edges = int(mcs_ev.sum() + fcs_ev.sum())
        negative_edges = int(
            np.triu(mm, 1).sum()
            + mf.sum()
            + np.triu(ff, 1).sum()
            + np.triu(ee, 1).sum()
        )
        edge_count = positive_edges + negative_edges
        possible = len(valid_indices) * max(len(valid_indices) - 1, 0) / 2

        def selected(values, offset, length):
            mask = valid[offset:offset + length]
            return values[offset:offset + length][mask]

        mcs_degree = selected(degree, 0, MCS_NODE_COUNT)
        fcs_degree = selected(degree, mcs_end, FCS_NODE_COUNT)
        ev_degree = selected(degree, fcs_end, EV_NODE_COUNT)

        def mean_or_zero(values):
            return float(values.mean()) if values.size else 0.0

        def max_or_zero(values):
            return float(values.max()) if values.size else 0.0

        return {
            'v15_active_mcs_count': float(active_mcs.sum()),
            'v15_active_ev_count': float(active_ev.sum()),
            'v15_active_node_count': float(len(valid_indices)),
            'v15_positive_edge_count': float(positive_edges),
            'v15_negative_edge_count': float(negative_edges),
            'v15_total_edge_count': float(edge_count),
            'v15_graph_density': float(edge_count / max(possible, 1.0)),
            'v15_average_degree': mean_or_zero(valid_degree),
            'v15_max_degree': max_or_zero(valid_degree),
            'v15_p95_degree': float(
                np.percentile(valid_degree, 95) if valid_degree.size else 0.0
            ),
            'v15_mcs_average_degree': mean_or_zero(mcs_degree),
            'v15_mcs_max_degree': max_or_zero(mcs_degree),
            'v15_fcs_average_degree': mean_or_zero(fcs_degree),
            'v15_fcs_max_degree': max_or_zero(fcs_degree),
            'v15_ev_average_degree': mean_or_zero(ev_degree),
            'v15_ev_max_degree': max_or_zero(ev_degree),
            'v15_isolated_node_count': float((valid_degree == 0).sum()),
            'v15_connected_component_count': float(len(component_sizes)),
            'v15_largest_component_node_count': float(
                max(component_sizes.values(), default=0)
            ),
        }

    def build(self) -> tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
        step = int(self.world.current_step)
        if self._cached_step == step and self._cached_snapshot is not None:
            return (
                self._cached_snapshot,
                self._cached_mcs_index,
                self._cached_ev_index,
            )

        started = time.perf_counter()
        mcs = self._mcs_features()
        fcs = self._fcs_features()
        ev = self._ev_features()
        relations = self._relations()
        mcs_ev, fcs_ev, mm, mf, ff, ee, active_mcs, active_ev = relations
        bits = self._edge_bits(mcs_ev, fcs_ev, mm, mf, ff, ee)
        packed_edges = np.packbits(bits, bitorder='little')
        if packed_edges.size != V15_LAYOUT.edge_byte_count:
            raise RuntimeError(
                f'v15 packed edge bytes {packed_edges.size} != '
                f'{V15_LAYOUT.edge_byte_count}'
            )
        snapshot = np.concatenate((
            mcs.reshape(-1),
            fcs.reshape(-1),
            ev.reshape(-1),
            packed_edges.astype(np.float32),
        )).astype(np.float32)
        if snapshot.shape != (V15_LAYOUT.graph_dim,):
            raise RuntimeError(
                f'v15 graph snapshot shape {snapshot.shape} != '
                f'{(V15_LAYOUT.graph_dim,)}'
            )

        self._cached_step = step
        self._cached_snapshot = snapshot
        self._cached_mcs_index = {
            int(item.id): index
            for index, item in enumerate(self.world.MCSs[:MCS_NODE_COUNT])
        }
        self._cached_ev_index = {
            int(item.id): index
            for index, item in enumerate(self.world.EVs[:EV_NODE_COUNT])
        }
        row = self._topology_metrics(*relations)
        row.update({
            'v15_graph_build_time_ms': (
                time.perf_counter() - started
            ) * 1000.0,
            'v15_quasi_count': float(sum(
                item.is_quasi for item in self.world.EVs
            )),
            'v15_iev_count': float(sum(
                item.is_iev for item in self.world.EVs
            )),
            'v15_mcs_ev_positive_edge_count': float(mcs_ev.sum()),
            'v15_fcs_ev_positive_edge_count': float(fcs_ev.sum()),
            'v15_mcs_mcs_negative_edge_count': float(
                np.triu(mm, 1).sum()
            ),
            'v15_mcs_fcs_negative_edge_count': float(mf.sum()),
            'v15_fcs_fcs_negative_edge_count': float(
                np.triu(ff, 1).sum()
            ),
            'v15_ev_ev_negative_edge_count': float(
                np.triu(ee, 1).sum()
            ),
        })
        self.rows.append(row)
        return snapshot, self._cached_mcs_index, self._cached_ev_index


class ObservationBuilderV15(ObservationBuilderV14):
    """V14-compatible candidate observations with a v15 signed graph."""

    def __init__(self, world):
        ObservationBuilder.__init__(self)
        self.world = world
        self.graph_builder = GlobalSignedGraphSnapshotBuilder(world)

    def obs_mcs(self, mcs, all_fcss=None):
        observation = super().obs_mcs(mcs, all_fcss)
        observation['v15_mcs_node_index'] = observation.pop(
            'v14_mcs_node_index'
        )
        observation['v15_graph_step'] = observation.pop('v14_graph_step')
        return observation


__all__ = [
    'BASE_CANDIDATE_DIM',
    'CANDIDATE_DISTANCE_RATIO',
    'CANDIDATE_ENERGY_MARGIN',
    'CANDIDATE_EV_INDEX',
    'CANDIDATE_IS_STAY',
    'EDGE_RELATIONS',
    'EV_NODE_COUNT',
    'EV_NODE_DIM',
    'FCS_NODE_COUNT',
    'FCS_NODE_DIM',
    'FEATURE_SCHEMA',
    'GlobalSignedGraphSnapshotBuilder',
    'MCS_NODE_COUNT',
    'MCS_NODE_DIM',
    'ObservationBuilderV15',
    'V15_CANDIDATE_DIM',
    'V15_LAYOUT',
    'V15_SELF_DIM',
]
