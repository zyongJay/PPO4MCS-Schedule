"""Canonical per-step global graph snapshots for v14.

This module deliberately does not modify :mod:`observation`.  It subclasses
the existing observation builder, keeps the v10-onlylow action set and audit
fields unchanged, and adds one canonical graph snapshot to every active MCS
observation.  The snapshot contains exactly one slot for every physical MCS,
FCS and EV.  All Low Actors at the same environment step therefore consume
the same node definitions and the same relation matrices.

Only primitive entity state and physical/topological relations enter the v14
policy.  The six legacy candidate columns are retained at the front of the
candidate tensor solely because the unchanged rollout/reward logging code
audits them; the v14 neural modules explicitly ignore those columns.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict

import numpy as np

from config import (
    AREA_LAT_MAX,
    AREA_LAT_MIN,
    AREA_LON_MAX,
    AREA_LON_MIN,
    COMM_RANGE,
    EV_BATTERY_CAPACITY,
    MAX_CHARGE_PER_SESSION_KWH,
    MAX_STEPS_PER_EPISODE,
    MCS_BATTERY_CAPACITY,
    NUM_EV,
    NUM_FCS,
    NUM_MCS,
    POWER_UNIT,
    STEP_DURATION_MIN,
)
from core import euclidean_distance
from observation import MCS_STAY_CANDIDATE_ID, ObservationBuilder


BASE_CANDIDATE_DIM = 6
MCS_NODE_DIM = 14
FCS_NODE_DIM = 8
EV_NODE_DIM = 12

CANDIDATE_EV_INDEX = 6
CANDIDATE_IS_STAY = 7
CANDIDATE_DISTANCE_RATIO = 8
CANDIDATE_ENERGY_MARGIN = 9
V14_CANDIDATE_DIM = 10

MCS_NODE_COUNT = int(NUM_MCS)
FCS_NODE_COUNT = int(NUM_FCS)
EV_NODE_COUNT = int(NUM_EV)


@dataclass(frozen=True)
class GraphSnapshotLayout:
    """Fixed-width layout stored after the leading current-MCS index."""

    mcs_features: int = MCS_NODE_COUNT * MCS_NODE_DIM
    fcs_features: int = FCS_NODE_COUNT * FCS_NODE_DIM
    ev_features: int = EV_NODE_COUNT * EV_NODE_DIM
    mcs_ev_attraction: int = MCS_NODE_COUNT * EV_NODE_COUNT
    fcs_ev_attraction: int = FCS_NODE_COUNT * EV_NODE_COUNT
    mcs_mcs_competition: int = MCS_NODE_COUNT * MCS_NODE_COUNT
    mcs_fcs_competition: int = MCS_NODE_COUNT * FCS_NODE_COUNT
    fcs_fcs_competition: int = FCS_NODE_COUNT * FCS_NODE_COUNT

    @property
    def graph_dim(self) -> int:
        return (
            self.mcs_features
            + self.fcs_features
            + self.ev_features
            + self.mcs_ev_attraction
            + self.fcs_ev_attraction
            + self.mcs_mcs_competition
            + self.mcs_fcs_competition
            + self.fcs_fcs_competition
        )

    @property
    def policy_self_dim(self) -> int:
        # The first scalar is the current MCS node index.  It is an addressing
        # field rather than a learned ID feature.
        return 1 + self.graph_dim


V14_LAYOUT = GraphSnapshotLayout()
V14_SELF_DIM = V14_LAYOUT.policy_self_dim


FEATURE_SCHEMA = {
    'version': 1,
    'scope': 'one_canonical_global_graph_per_environment_step',
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
    'mcs_node': [
        'x', 'y', 'target_dx', 'target_dy', 'remain_ratio', 'idle',
        'task', 'recharging', 'broken', 'energy_stranded', 'has_target',
        'target_is_quasi', 'target_is_iev', 'target_is_fcs',
    ],
    'fcs_node': [
        'x', 'y', 'available_slot_ratio', 'occupied_slot_ratio',
        'min_release_ratio', 'mean_release_ratio', 'busy', 'idle',
    ],
    'ev_node': [
        'x', 'y', 'next_dx', 'next_dy', 'remain_ratio', 'need_ratio',
        'quasi', 'iev', 'charging', 'success', 'failed', 'wait_ratio',
    ],
    'relations': [
        'mcs_ev_attraction', 'fcs_ev_attraction',
        'mcs_mcs_competition', 'mcs_fcs_competition',
        'fcs_fcs_competition',
    ],
    'candidate_policy_fields': [
        'ev_node_index', 'is_stay', 'distance_ratio',
        'mcs_energy_margin_ratio',
    ],
    'legacy_candidate_fields_policy_visible': False,
}


_MAP_DIAGONAL_M = max(
    euclidean_distance(
        AREA_LON_MIN, AREA_LAT_MIN, AREA_LON_MAX, AREA_LAT_MAX
    ),
    1.0,
)
_MAX_TIME_MIN = max(float(MAX_STEPS_PER_EPISODE * STEP_DURATION_MIN), 1.0)


def _position(pos) -> tuple[float, float]:
    return (
        float(np.clip(
            (float(pos[0]) - AREA_LON_MIN)
            / max(AREA_LON_MAX - AREA_LON_MIN, 1e-8),
            0.0,
            1.0,
        )),
        float(np.clip(
            (float(pos[1]) - AREA_LAT_MIN)
            / max(AREA_LAT_MAX - AREA_LAT_MIN, 1e-8),
            0.0,
            1.0,
        )),
    )


def _relative_position(origin, target) -> tuple[float, float]:
    if target is None:
        return 0.0, 0.0
    return (
        float(np.clip(
            (float(target[0]) - float(origin[0]))
            / max(AREA_LON_MAX - AREA_LON_MIN, 1e-8),
            -1.0,
            1.0,
        )),
        float(np.clip(
            (float(target[1]) - float(origin[1]))
            / max(AREA_LAT_MAX - AREA_LAT_MIN, 1e-8),
            -1.0,
            1.0,
        )),
    )


def _smooth_radius_weight(distance_km: float) -> float:
    """Continuous compact-support edge weight, exactly zero at COMM_RANGE."""
    radius = max(float(COMM_RANGE), 1e-8)
    if distance_km >= radius:
        return 0.0
    ratio = max(float(distance_km), 0.0) / radius
    return float(0.5 * (1.0 + math.cos(math.pi * ratio)))


class GlobalGraphSnapshotBuilder:
    """Build one canonical, fixed-width heterogeneous graph snapshot."""

    def __init__(self, world):
        self.world = world
        self._cached_step: int | None = None
        self._cached_snapshot: np.ndarray | None = None
        self._cached_mcs_index: Dict[int, int] = {}
        self._cached_ev_index: Dict[int, int] = {}
        self.rows: list[dict[str, float]] = []

    def _mcs_features(self) -> np.ndarray:
        features = np.zeros((MCS_NODE_COUNT, MCS_NODE_DIM), np.float32)
        for index, mcs in enumerate(self.world.MCSs[:MCS_NODE_COUNT]):
            x, y = _position(mcs.pos)
            dx, dy = _relative_position(mcs.pos, mcs.current_target_pos)
            target_type = str(getattr(mcs, 'current_target_type', '')).upper()
            features[index] = np.asarray([
                x,
                y,
                dx,
                dy,
                np.clip(float(mcs.remain) / MCS_BATTERY_CAPACITY, 0.0, 1.0),
                float(mcs.is_idle),
                float(mcs.is_task),
                float(mcs.is_recharging),
                float(mcs.is_broken),
                float(getattr(mcs, 'is_energy_stranded', False)),
                float(mcs.current_target_pos is not None),
                float(target_type == 'QUASI'),
                float(target_type == 'IEV'),
                float(target_type == 'FCS'),
            ], np.float32)
        return features

    def _fcs_features(self) -> np.ndarray:
        features = np.zeros((FCS_NODE_COUNT, FCS_NODE_DIM), np.float32)
        for index, fcs in enumerate(self.world.FCSs[:FCS_NODE_COUNT]):
            x, y = _position(fcs.pos)
            capacity = max(float(fcs.capacity), 1.0)
            release = [
                max(float(value), 0.0)
                for value in fcs.slot_charge_remain_min
                if float(value) > 0.0
            ]
            minimum = min(release, default=0.0) / _MAX_TIME_MIN
            mean = float(np.mean(release)) / _MAX_TIME_MIN if release else 0.0
            features[index] = np.asarray([
                x,
                y,
                float(fcs.available_slots) / capacity,
                float(fcs.occupied_slots) / capacity,
                np.clip(minimum, 0.0, 1.0),
                np.clip(mean, 0.0, 1.0),
                float(fcs.is_busy),
                float(fcs.is_idle),
            ], np.float32)
        return features

    @staticmethod
    def _next_ev_waypoint(ev):
        track = list(getattr(ev, 'track', []))
        index = int(getattr(ev, 'track_index', 0))
        if track:
            index = min(max(index, 0), len(track) - 1)
            return track[index]
        return getattr(ev, 'destination', None)

    def _ev_features(self) -> np.ndarray:
        features = np.zeros((EV_NODE_COUNT, EV_NODE_DIM), np.float32)
        for index, ev in enumerate(self.world.EVs[:EV_NODE_COUNT]):
            x, y = _position(ev.pos)
            dx, dy = _relative_position(
                ev.pos, self._next_ev_waypoint(ev)
            )
            charging = bool(ev.is_charged and ev.charge_pos is not None)
            success = bool(ev.is_charged and ev.charge_pos is None)
            features[index] = np.asarray([
                x,
                y,
                dx,
                dy,
                np.clip(float(ev.remain) / EV_BATTERY_CAPACITY, 0.0, 1.5),
                np.clip(
                    float(ev.need_power) / MAX_CHARGE_PER_SESSION_KWH,
                    0.0,
                    2.0,
                ),
                float(ev.is_quasi),
                float(ev.is_iev),
                float(charging),
                float(success),
                float(ev.fail_charge),
                np.clip(
                    float(ev.wait_time_steps)
                    * STEP_DURATION_MIN / _MAX_TIME_MIN,
                    0.0,
                    1.0,
                ),
            ], np.float32)
        return features

    def _relations(self):
        mcs_ev = np.zeros((MCS_NODE_COUNT, EV_NODE_COUNT), np.float32)
        fcs_ev = np.zeros((FCS_NODE_COUNT, EV_NODE_COUNT), np.float32)
        active_ev = np.zeros(EV_NODE_COUNT, bool)

        for ev_index, ev in enumerate(self.world.EVs[:EV_NODE_COUNT]):
            active_ev[ev_index] = bool(ev.is_quasi or ev.is_iev)
            if not active_ev[ev_index]:
                continue
            for mcs_index, mcs in enumerate(
                self.world.MCSs[:MCS_NODE_COUNT]
            ):
                available = bool(
                    mcs.is_idle
                    and not mcs.is_recharging
                    and not mcs.is_broken
                    and not getattr(mcs, 'is_energy_stranded', False)
                )
                if not available:
                    continue
                distance_km = euclidean_distance(
                    *mcs.pos, *ev.pos
                ) / 1000.0
                mcs_ev[mcs_index, ev_index] = _smooth_radius_weight(
                    distance_km
                )
            for fcs_index, fcs in enumerate(
                self.world.FCSs[:FCS_NODE_COUNT]
            ):
                distance_km = euclidean_distance(
                    *fcs.pos, *ev.pos
                ) / 1000.0
                fcs_ev[fcs_index, ev_index] = _smooth_radius_weight(
                    distance_km
                )

        mcs_incidence = mcs_ev > 0.0
        fcs_incidence = fcs_ev > 0.0
        mm = (mcs_incidence.astype(np.int16) @ mcs_incidence.T) > 0
        mf = (mcs_incidence.astype(np.int16) @ fcs_incidence.T) > 0
        ff = (fcs_incidence.astype(np.int16) @ fcs_incidence.T) > 0
        np.fill_diagonal(mm, False)
        np.fill_diagonal(ff, False)
        return (
            mcs_ev,
            fcs_ev,
            mm.astype(np.float32),
            mf.astype(np.float32),
            ff.astype(np.float32),
            active_ev,
        )

    @staticmethod
    def _component_statistics(
        mcs_ev: np.ndarray,
        fcs_ev: np.ndarray,
        mm: np.ndarray,
        mf: np.ndarray,
        ff: np.ndarray,
        active_ev: np.ndarray,
    ) -> tuple[int, int]:
        """Count physical components without adding any artificial bridge."""
        provider_count = MCS_NODE_COUNT + FCS_NODE_COUNT
        active_indices = np.flatnonzero(active_ev)
        node_count = provider_count + len(active_indices)
        if node_count == 0:
            return 0, 0
        parent = list(range(node_count))

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        active_local = {
            int(ev_index): provider_count + local_index
            for local_index, ev_index in enumerate(active_indices)
        }
        for mcs_index, ev_index in np.argwhere(mcs_ev > 0.0):
            union(int(mcs_index), active_local[int(ev_index)])
        for fcs_index, ev_index in np.argwhere(fcs_ev > 0.0):
            union(
                MCS_NODE_COUNT + int(fcs_index),
                active_local[int(ev_index)],
            )
        for left, right in np.argwhere(mm > 0.0):
            union(int(left), int(right))
        for mcs_index, fcs_index in np.argwhere(mf > 0.0):
            union(int(mcs_index), MCS_NODE_COUNT + int(fcs_index))
        for left, right in np.argwhere(ff > 0.0):
            union(
                MCS_NODE_COUNT + int(left),
                MCS_NODE_COUNT + int(right),
            )
        sizes: dict[int, int] = {}
        for node in range(node_count):
            root = find(node)
            sizes[root] = sizes.get(root, 0) + 1
        return len(sizes), max(sizes.values(), default=0)

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
        mcs_ev, fcs_ev, mm, mf, ff, active_ev = self._relations()
        component_count, largest_component = self._component_statistics(
            mcs_ev, fcs_ev, mm, mf, ff, active_ev
        )
        snapshot = np.concatenate([
            mcs.reshape(-1),
            fcs.reshape(-1),
            ev.reshape(-1),
            mcs_ev.reshape(-1),
            fcs_ev.reshape(-1),
            mm.reshape(-1),
            mf.reshape(-1),
            ff.reshape(-1),
        ]).astype(np.float32)
        if snapshot.shape != (V14_LAYOUT.graph_dim,):
            raise RuntimeError(
                f'v14 graph snapshot shape {snapshot.shape} != '
                f'{(V14_LAYOUT.graph_dim,)}'
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
        self.rows.append({
            'v14_graph_build_time_ms': (
                time.perf_counter() - started
            ) * 1000.0,
            'v14_active_ev_count': float(active_ev.sum()),
            'v14_quasi_count': float(sum(
                ev_item.is_quasi for ev_item in self.world.EVs
            )),
            'v14_iev_count': float(sum(
                ev_item.is_iev for ev_item in self.world.EVs
            )),
            'v14_mcs_ev_attraction_edge_count': float(
                np.count_nonzero(mcs_ev)
            ),
            'v14_fcs_ev_attraction_edge_count': float(
                np.count_nonzero(fcs_ev)
            ),
            'v14_mcs_mcs_competition_edge_count': float(
                np.count_nonzero(np.triu(mm, 1))
            ),
            'v14_mcs_fcs_competition_edge_count': float(
                np.count_nonzero(mf)
            ),
            'v14_fcs_fcs_competition_edge_count': float(
                np.count_nonzero(np.triu(ff, 1))
            ),
            'v14_connected_component_count': float(component_count),
            'v14_largest_component_node_count': float(largest_component),
        })
        return snapshot, self._cached_mcs_index, self._cached_ev_index

    def metrics(self) -> dict[str, float]:
        if not self.rows:
            return {}
        result: dict[str, float] = {}
        for name in self.rows[0]:
            values = np.asarray([row[name] for row in self.rows], np.float64)
            result[f'{name}_episode_mean'] = float(values.mean())
            result[f'{name}_episode_max'] = float(values.max())
        return result


class ObservationBuilderV14(ObservationBuilder):
    """v10 observation compatibility plus a canonical v14 graph snapshot."""

    def __init__(self, world):
        super().__init__()
        self.world = world
        self.graph_builder = GlobalGraphSnapshotBuilder(world)

    def obs_mcs(self, mcs, all_fcss=None):
        observation = super().obs_mcs(mcs, all_fcss)
        snapshot, mcs_index, ev_index = self.graph_builder.build()
        current_index = mcs_index[int(mcs.id)]
        observation['low_self_state'] = np.concatenate((
            np.asarray([current_index], np.float32), snapshot,
        )).astype(np.float32)
        candidates = np.zeros(
            (observation['low_candidates'].shape[0], V14_CANDIDATE_DIM),
            np.float32,
        )
        # The legacy values remain available only for unchanged rollout and
        # reward auditing.  V14 actor/critic code starts reading at column 6.
        candidates[:, :BASE_CANDIDATE_DIM] = np.asarray(
            observation['low_candidates'], np.float32
        )[:, :BASE_CANDIDATE_DIM]
        candidate_ids = np.asarray(observation['candidate_ids'], np.int64)
        candidate_mask = np.asarray(
            observation['low_candidate_mask'], bool
        )
        nearest_fcs_distance = {}
        for index, candidate_id in enumerate(candidate_ids):
            is_stay = int(candidate_id) == int(MCS_STAY_CANDIDATE_ID)
            candidates[index, CANDIDATE_EV_INDEX] = float(
                ev_index.get(int(candidate_id), -1)
            )
            candidates[index, CANDIDATE_IS_STAY] = float(is_stay)
            if not candidate_mask[index] or is_stay or candidate_id < 0:
                candidates[index, CANDIDATE_ENERGY_MARGIN] = float(
                    np.clip(mcs.remain / MCS_BATTERY_CAPACITY, 0.0, 1.0)
                )
                continue
            ev = self.world.EVs[ev_index[int(candidate_id)]]
            distance_m = euclidean_distance(*mcs.pos, *ev.pos)
            candidates[index, CANDIDATE_DISTANCE_RATIO] = float(
                np.clip(distance_m / _MAP_DIAGONAL_M, 0.0, 1.0)
            )
            if int(candidate_id) not in nearest_fcs_distance:
                nearest_fcs_distance[int(candidate_id)] = min(
                    euclidean_distance(*ev.pos, *fcs.pos) / 1000.0
                    for fcs in self.world.FCSs
                )
            required = (
                distance_m / 1000.0
                + nearest_fcs_distance[int(candidate_id)]
            ) * POWER_UNIT
            candidates[index, CANDIDATE_ENERGY_MARGIN] = float(np.clip(
                (float(mcs.remain) - required) / MCS_BATTERY_CAPACITY,
                -1.0,
                1.0,
            ))
        observation['low_candidates'] = candidates
        observation['obs_self'] = observation['low_self_state']
        observation['obs_tgt'] = candidates
        observation['v14_mcs_node_index'] = int(current_index)
        observation['v14_graph_step'] = int(self.world.current_step)
        return observation
