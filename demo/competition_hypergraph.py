"""Competition-hypergraph construction and neural encoders for v12.

The builder is deliberately side-effect free with respect to the simulator.  It
only augments copies of current observations with a fixed-width, replayable
snapshot.  Actor snapshots are constructed from each MCS's legal local view;
the critic snapshot is constructed independently from the global state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn

from config import (
    AREA_LAT_MAX, AREA_LAT_MIN, AREA_LON_MAX, AREA_LON_MIN, COMM_RANGE,
    EV_BATTERY_CAPACITY, FCS_SLOTS_PER_STATION, MAX_CHARGE_PER_SESSION_KWH,
    MAX_STEPS_PER_EPISODE, MCS_BATTERY_CAPACITY, MOVE_SPEED, NUM_FCS, NUM_MCS,
    POWER_UNIT, STEP_DURATION_MIN, TOP_K_MCS_CANDIDATES,
)
from core import EV, FCS, MCS, euclidean_distance
from observation import MCS_STAY_CANDIDATE_ID


BASE_CANDIDATE_DIM = 6
CONTROL_DIM = 2
SOFT_OCCUPANCY_OFFSET = -2
RESERVED_OFFSET = -1
QUASI_DIM = 7
MCS_DIM = 8
FCS_DIM = 6
QMCS_REL_DIM = 3
QFCS_REL_DIM = 3
MAX_MCS_MEMBERS = int(NUM_MCS)
MAX_FCS_MEMBERS = int(NUM_FCS)
MAX_GLOBAL_EDGES = int(NUM_MCS) * max(int(TOP_K_MCS_CANDIDATES) - 1, 0)
MCS_SLOT_DIM = 1 + MCS_DIM + QMCS_REL_DIM
FCS_SLOT_DIM = 1 + FCS_DIM + QFCS_REL_DIM
HG_CONTEXT_DIM = (
    QUASI_DIM + 1 + QUASI_DIM
    + MAX_MCS_MEMBERS * MCS_SLOT_DIM
    + MAX_FCS_MEMBERS * FCS_SLOT_DIM + 2
)
V12_CANDIDATE_DIM = BASE_CANDIDATE_DIM + HG_CONTEXT_DIM + CONTROL_DIM
V12_GLOBAL_GRAPH_DIM = MAX_GLOBAL_EDGES * HG_CONTEXT_DIM

FEATURE_SCHEMA = {
    'version': 2,
    'base_candidate': [
        'urgency', 'distance', 'attraction', 'immediate_iev_attraction',
        'removed_artificial_mcs_competition',
        'removed_artificial_fcs_competition',
    ],
    'quasi': [
        'urgency', 'attraction', 'immediate_iev_attraction', 'x', 'y',
        'remain_ratio', 'need_ratio',
    ],
    'mcs': [
        'x', 'y', 'remain_ratio', 'idle', 'task', 'recharging',
        'unavailable', 'target_valid',
    ],
    'fcs': [
        'x', 'y', 'available_slot_ratio', 'occupied_slot_ratio',
        'mean_release_time_ratio', 'busy',
    ],
    'q_mcs_relation': ['distance_ratio', 'eta_ratio', 'travel_energy_ratio'],
    'q_fcs_relation': ['distance_ratio', 'eta_ratio', 'available_slot_ratio'],
    'candidate_control': ['local_soft_occupancy', 'locally_reserved'],
    'actor_visibility': 'independently_built_from_current_local_neighbor_lists',
    'critic_visibility': 'global',
    'artificial_competition_columns_zeroed': [4, 5],
}


@dataclass(frozen=True)
class CompetitionHypergraphConfig:
    r_merge_km: float = 1.0
    jaccard_eta: float = 0.5
    hidden_dim: int = 64

    def __post_init__(self):
        if self.r_merge_km < 0:
            raise ValueError('r_merge_km must be nonnegative')
        if not 0.0 <= self.jaccard_eta <= 1.0:
            raise ValueError('jaccard_eta must be in [0, 1]')
        if self.hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')


def _position(pos) -> np.ndarray:
    return np.asarray([
        (float(pos[0]) - AREA_LON_MIN) / (AREA_LON_MAX - AREA_LON_MIN),
        (float(pos[1]) - AREA_LAT_MIN) / (AREA_LAT_MAX - AREA_LAT_MIN),
    ], np.float32)


_MAP_DIAGONAL_M = max(euclidean_distance(
    AREA_LON_MIN, AREA_LAT_MIN, AREA_LON_MAX, AREA_LAT_MAX
), 1.0)
_MAX_ETA_MIN = max(float(MAX_STEPS_PER_EPISODE * STEP_DURATION_MIN), 1.0)


def _relation(origin, target, available_ratio: float | None = None) -> np.ndarray:
    distance_m = euclidean_distance(*origin, *target)
    distance_ratio = np.clip(distance_m / _MAP_DIAGONAL_M, 0.0, 2.0)
    eta_ratio = np.clip(distance_m / (MOVE_SPEED * 60.0) / _MAX_ETA_MIN, 0.0, 2.0)
    third = (
        np.clip(distance_m / 1000.0 * POWER_UNIT / MCS_BATTERY_CAPACITY, 0.0, 2.0)
        if available_ratio is None else np.clip(available_ratio, 0.0, 1.0)
    )
    return np.asarray([distance_ratio, eta_ratio, third], np.float32)


def _quasi_feature(ev: EV | None, base: np.ndarray, fallback_pos) -> np.ndarray:
    position = _position(ev.pos if ev is not None else fallback_pos)
    remain = 0.0 if ev is None else np.clip(float(ev.remain) / EV_BATTERY_CAPACITY, 0.0, 1.5)
    need = 0.0 if ev is None else np.clip(float(ev.need_power) / MAX_CHARGE_PER_SESSION_KWH, 0.0, 2.0)
    return np.asarray([
        base[0], base[2], base[3], position[0], position[1], remain, need,
    ], np.float32)


def _mcs_feature(mcs: MCS) -> np.ndarray:
    position = _position(mcs.pos)
    unavailable = bool(mcs.is_broken or mcs.is_energy_stranded)
    return np.asarray([
        position[0], position[1],
        np.clip(float(mcs.remain) / MCS_BATTERY_CAPACITY, 0.0, 1.0),
        float(mcs.is_idle and not unavailable), float(mcs.is_task),
        float(mcs.is_recharging), float(unavailable),
        float(mcs.current_target_pos is not None),
    ], np.float32)


def _fcs_feature(fcs: FCS) -> np.ndarray:
    position = _position(fcs.pos)
    available = float(fcs.available_slots) / max(float(fcs.capacity), 1.0)
    occupied = float(fcs.occupied_slots) / max(float(fcs.capacity), 1.0)
    release = [max(float(x), 0.0) for x in fcs.slot_charge_remain_min]
    mean_release = float(np.mean(release)) / _MAX_ETA_MIN if release else 0.0
    return np.asarray([
        position[0], position[1], available, occupied,
        np.clip(mean_release, 0.0, 1.0), float(fcs.is_busy),
    ], np.float32)


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / max(len(union), 1)


def _deduplicate(
    qids: Iterable[int], q_to_mcs: Dict[int, set[int]], ev_by_id: Dict[int, EV],
    attraction: Dict[int, tuple[float, float]], config: CompetitionHypergraphConfig,
) -> tuple[list[int], dict[int, int], list[list[int]]]:
    ids = sorted(set(int(x) for x in qids))
    parent = {qid: qid for qid in ids}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            distance_km = euclidean_distance(*ev_by_id[left].pos, *ev_by_id[right].pos) / 1000.0
            if (distance_km <= config.r_merge_km
                    and _jaccard(q_to_mcs[left], q_to_mcs[right]) >= config.jaccard_eta):
                union(left, right)
    grouped: Dict[int, list[int]] = {}
    for qid in ids:
        grouped.setdefault(find(qid), []).append(qid)
    clusters = [sorted(values) for _, values in sorted(grouped.items())]
    representative = {}
    centers = []
    for cluster in clusters:
        # Attraction and urgency only; persistent qid is the stable tie break.
        rep = min(cluster, key=lambda qid: (
            -attraction.get(qid, (0.0, 0.0))[0],
            -attraction.get(qid, (0.0, 0.0))[1], qid,
        ))
        centers.append(rep)
        representative.update({qid: rep for qid in cluster})
    return sorted(centers), representative, clusters


def _visibility(mcs: MCS) -> tuple[set[int], set[int]]:
    visible_mcs = {int(mcs.id)} | {
        int(other.id) for other in list(mcs.near_idle_mcs) + list(mcs.near_task_mcs)
    }
    visible_fcs = {
        int(fcs.id) for fcs in list(mcs.near_available_fcs) + list(mcs.near_busy_fcs)
    }
    return visible_mcs, visible_fcs


def _serviceable_fcs(q: EV) -> list[FCS]:
    by_id = {
        int(fcs.id): fcs
        for fcs in list(q.near_available_fcs) + list(q.near_busy_fcs)
    }
    return [by_id[key] for key in sorted(by_id)]


def _pack_context(
    candidate_q: EV | None, candidate_base: np.ndarray, fallback_pos,
    center_q: EV | None = None, mcs_members: Iterable[MCS] = (),
    fcs_members: Iterable[FCS] = (),
) -> np.ndarray:
    result = np.zeros(HG_CONTEXT_DIM, np.float32)
    cursor = 0
    candidate_feature = _quasi_feature(candidate_q, candidate_base, fallback_pos)
    result[cursor:cursor + QUASI_DIM] = candidate_feature
    cursor += QUASI_DIM
    has_edge = center_q is not None
    result[cursor] = float(has_edge)
    cursor += 1
    if has_edge:
        center_base = candidate_base if candidate_q is center_q else np.asarray([
            candidate_base[0], 0.0, candidate_base[2], candidate_base[3], 0.0, 0.0,
        ], np.float32)
        result[cursor:cursor + QUASI_DIM] = _quasi_feature(center_q, center_base, center_q.pos)
    cursor += QUASI_DIM
    mcss = sorted(mcs_members, key=lambda item: int(item.id))[:MAX_MCS_MEMBERS]
    for index, mcs in enumerate(mcss):
        offset = cursor + index * MCS_SLOT_DIM
        result[offset] = 1.0
        result[offset + 1:offset + 1 + MCS_DIM] = _mcs_feature(mcs)
        result[offset + 1 + MCS_DIM:offset + MCS_SLOT_DIM] = _relation(center_q.pos, mcs.pos)
    cursor += MAX_MCS_MEMBERS * MCS_SLOT_DIM
    fcss = sorted(fcs_members, key=lambda item: int(item.id))[:MAX_FCS_MEMBERS]
    for index, fcs in enumerate(fcss):
        offset = cursor + index * FCS_SLOT_DIM
        result[offset] = 1.0
        result[offset + 1:offset + 1 + FCS_DIM] = _fcs_feature(fcs)
        available = float(fcs.available_slots) / max(float(fcs.capacity), 1.0)
        result[offset + 1 + FCS_DIM:offset + FCS_SLOT_DIM] = _relation(
            center_q.pos, fcs.pos, available
        )
    cursor += MAX_FCS_MEMBERS * FCS_SLOT_DIM
    if has_edge:
        result[cursor] = math.log1p(len(mcss))
        result[cursor + 1] = math.log1p(len(fcss))
    return result


class CompetitionHypergraphBuilder:
    def __init__(self, config=CompetitionHypergraphConfig()):
        self.config = config

    def build(self, world, observation_by_agent: Dict[object, dict]):
        entries = [
            (agent, observation_by_agent[agent])
            for agent in sorted(observation_by_agent, key=lambda item: getattr(item, 'id', -1))
            if isinstance(agent, MCS)
        ]
        ev_by_id = {int(ev.id): ev for ev in world.EVs}
        mcs_by_id = {int(mcs.id): mcs for mcs in world.MCSs}
        q_to_mcs: Dict[int, set[int]] = {}
        attraction: Dict[int, tuple[float, float]] = {}
        raw_by_mcs: Dict[int, np.ndarray] = {}
        for mcs, observation in entries:
            raw = np.asarray(observation['low_candidates'], np.float32)
            if raw.shape[-1] != BASE_CANDIDATE_DIM:
                # A repeated call in the same physical state is harmless.
                raw = raw[:, :BASE_CANDIDATE_DIM]
            raw_by_mcs[int(mcs.id)] = raw.copy()
            ids = np.asarray(observation['candidate_ids'], np.int64)
            mask = np.asarray(observation['low_candidate_mask'], bool)
            for index in np.flatnonzero(mask & (ids >= 0)):
                qid = int(ids[index])
                q_to_mcs.setdefault(qid, set()).add(int(mcs.id))
                score = (float(raw[index, 2] + raw[index, 3]), float(raw[index, 0]))
                attraction[qid] = max(attraction.get(qid, score), score)
        qcomp = {
            qid for qid, members in q_to_mcs.items()
            if len(members) >= 2 and qid in ev_by_id
        }
        centers, representative, clusters = _deduplicate(
            qcomp, q_to_mcs, ev_by_id, attraction, self.config
        )
        global_contexts = []
        global_m_sizes, global_f_sizes = [], []
        for center_id in centers:
            center = ev_by_id[center_id]
            member_mcss = [mcs_by_id[mid] for mid in sorted(q_to_mcs[center_id])]
            member_fcss = _serviceable_fcs(center)
            source_base = next(
                raw_by_mcs[mid][np.flatnonzero(
                    (np.asarray(observation_by_agent[mcs_by_id[mid]]['candidate_ids']) == center_id)
                    & np.asarray(observation_by_agent[mcs_by_id[mid]]['low_candidate_mask'], bool)
                )[0]]
                for mid in sorted(q_to_mcs[center_id])
            )
            global_contexts.append(_pack_context(
                center, source_base, center.pos, center, member_mcss, member_fcss
            ))
            global_m_sizes.append(len(member_mcss))
            global_f_sizes.append(len(member_fcss))
        global_graph = np.zeros((MAX_GLOBAL_EDGES, HG_CONTEXT_DIM), np.float32)
        if len(global_contexts) > MAX_GLOBAL_EDGES:
            raise RuntimeError('Competition edge capacity is smaller than Q_cand upper bound')
        if global_contexts:
            global_graph[:len(global_contexts)] = np.stack(global_contexts)

        visible_edge_counts = []
        for mcs, observation in entries:
            raw = raw_by_mcs[int(mcs.id)]
            ids = np.asarray(observation['candidate_ids'], np.int64)
            mask = np.asarray(observation['low_candidate_mask'], bool)
            visible_mcs, visible_fcs = _visibility(mcs)
            local_q_to_mcs = {
                qid: members & visible_mcs for qid, members in q_to_mcs.items()
            }
            local_qcomp = {
                qid for qid, members in local_q_to_mcs.items()
                if len(members) >= 2 and qid in ev_by_id
            }
            local_centers, local_rep, _ = _deduplicate(
                local_qcomp, local_q_to_mcs, ev_by_id, attraction, self.config
            )
            visible_edge_counts.append(len(local_centers))
            contexts = np.zeros((raw.shape[0], HG_CONTEXT_DIM), np.float32)
            rep_ids = np.full(raw.shape[0], -1, np.int64)
            for index in np.flatnonzero(mask):
                qid = int(ids[index])
                q = ev_by_id.get(qid)
                if qid in local_rep:
                    rep_id = local_rep[qid]
                    center = ev_by_id[rep_id]
                    member_mcss = [
                        mcs_by_id[mid] for mid in sorted(local_q_to_mcs[rep_id])
                    ]
                    member_fcss = [
                        fcs for fcs in _serviceable_fcs(center)
                        if int(fcs.id) in visible_fcs
                    ]
                    contexts[index] = _pack_context(
                        q, raw[index], mcs.pos, center, member_mcss, member_fcss
                    )
                    rep_ids[index] = rep_id
                else:
                    contexts[index] = _pack_context(q, raw[index], mcs.pos)
            base_without_competition = raw.copy()
            base_without_competition[:, 4:6] = 0.0
            observation['low_candidates'] = np.concatenate(
                (
                    base_without_competition,
                    contexts,
                    np.zeros((raw.shape[0], CONTROL_DIM), np.float32),
                ), axis=-1
            ).astype(np.float32)
            observation['obs_tgt'] = observation['low_candidates']
            observation['competition_representative_ids'] = rep_ids
            local_member_ids = np.full(
                (raw.shape[0], MAX_MCS_MEMBERS), -1, np.int64
            )
            for index in np.flatnonzero(mask & (ids >= 0)):
                members = sorted(
                    local_q_to_mcs.get(int(ids[index]), set()) - {int(mcs.id)}
                )[:MAX_MCS_MEMBERS]
                local_member_ids[index, :len(members)] = members
            observation['competition_local_member_ids'] = local_member_ids
            observation['v12_mcs_id'] = int(mcs.id)

        qcount = len(q_to_mcs)
        qcomp_count = len(qcomp)
        def percentile(values, q):
            return float(np.percentile(values, q)) if values else 0.0
        return global_graph.reshape(-1), {
            'hg_q_count': sum(ev.is_quasi for ev in world.EVs),
            'hg_q_cand_count': qcount,
            'hg_q_comp_count': qcomp_count,
            'hg_q_center_count': len(centers),
            'hg_dedup_ratio': 1.0 - len(centers) / max(qcomp_count, 1),
            'hg_empty_graph': float(not centers),
            'hg_mcs_members_mean': float(np.mean(global_m_sizes)) if global_m_sizes else 0.0,
            'hg_mcs_members_p50': percentile(global_m_sizes, 50),
            'hg_mcs_members_p95': percentile(global_m_sizes, 95),
            'hg_mcs_members_max': float(max(global_m_sizes, default=0)),
            'hg_fcs_members_mean': float(np.mean(global_f_sizes)) if global_f_sizes else 0.0,
            'hg_fcs_members_p50': percentile(global_f_sizes, 50),
            'hg_fcs_members_p95': percentile(global_f_sizes, 95),
            'hg_fcs_members_max': float(max(global_f_sizes, default=0)),
            'hg_actor_visible_edges_mean': float(np.mean(visible_edge_counts)) if visible_edge_counts else 0.0,
            'hg_actor_visible_edges_p95': percentile(visible_edge_counts, 95),
            'hg_actor_visible_edges_max': float(max(visible_edge_counts, default=0)),
            'hg_cluster_count': len(clusters),
        }

    @staticmethod
    def augment_unseen_observation(observation: dict) -> dict:
        raw = np.asarray(observation['low_candidates'], np.float32)[:, :BASE_CANDIDATE_DIM]
        raw[:, 4:6] = 0.0
        contexts = np.zeros((raw.shape[0], HG_CONTEXT_DIM), np.float32)
        controls = np.zeros((raw.shape[0], CONTROL_DIM), np.float32)
        observation['low_candidates'] = np.concatenate(
            (raw, contexts, controls), axis=-1
        )
        observation['obs_tgt'] = observation['low_candidates']
        observation['competition_local_member_ids'] = np.full(
            (raw.shape[0], MAX_MCS_MEMBERS), -1, np.int64
        )
        observation.setdefault('v12_mcs_id', -1)
        return observation


class _MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.SiLU(),
            nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim), nn.SiLU(),
        )

    def forward(self, value):
        return self.net(value)


class CompetitionHyperedgeEncoder(nn.Module):
    """Type-specific, relation-aware gated-sum competition encoder."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        h = int(hidden_dim)
        self.quasi_encoder = _MLP(QUASI_DIM, h)
        self.mcs_encoder = _MLP(MCS_DIM, h)
        self.fcs_encoder = _MLP(FCS_DIM, h)
        self.qm_rel_encoder = _MLP(QMCS_REL_DIM, h)
        self.qf_rel_encoder = _MLP(QFCS_REL_DIM, h)
        self.mcs_gate = nn.Sequential(nn.Linear(3 * h, h), nn.Sigmoid())
        self.fcs_gate = nn.Sequential(nn.Linear(3 * h, h), nn.Sigmoid())
        self.mcs_message = _MLP(2 * h, h)
        self.fcs_message = _MLP(2 * h, h)
        self.edge_mlp = _MLP(3 * h + 2, h)
        self.hidden_dim = h
        self._diagnostic_rows: list[dict[str, float]] = []

    def _record_diagnostics(
        self, edge, edge_present, mcs_gate, mcs_mask, fcs_gate, fcs_mask,
        elapsed_ms,
    ):
        with torch.no_grad():
            valid_edges = edge[edge_present.squeeze(-1) > 0.5]
            if valid_edges.numel():
                edge_mean = float(valid_edges.mean().item())
                edge_std = float(valid_edges.std(unbiased=False).item())
                edge_norm = float(valid_edges.norm(dim=-1).mean().item())
            else:
                edge_mean = edge_std = edge_norm = 0.0
            mcs_values = mcs_gate[mcs_mask.expand_as(mcs_gate) > 0.5]
            fcs_values = fcs_gate[fcs_mask.expand_as(fcs_gate) > 0.5]
            gate_values = torch.cat((mcs_values, fcs_values))
            gate_saturation = float(
                ((gate_values < 0.05) | (gate_values > 0.95))
                .float().mean().item()
            ) if gate_values.numel() else 0.0
            self._diagnostic_rows.append({
                'output_mean': edge_mean,
                'output_std': edge_std,
                'output_norm': edge_norm,
                'gate_saturation_ratio': gate_saturation,
                'encode_time_ms': float(elapsed_ms),
                'nan_count': float(torch.isnan(edge).sum().item()),
                'inf_count': float(torch.isinf(edge).sum().item()),
            })

    def pop_diagnostics(self, prefix: str) -> dict[str, float]:
        rows, self._diagnostic_rows = self._diagnostic_rows, []
        names = (
            'output_mean', 'output_std', 'output_norm',
            'gate_saturation_ratio', 'encode_time_ms', 'nan_count',
            'inf_count',
        )
        return {
            f'{prefix}_{name}': (
                float(np.mean([row[name] for row in rows])) if rows else 0.0
            )
            for name in names
        }

    def forward(self, packed):
        started = time.perf_counter()
        shape = packed.shape[:-1]
        flat = packed.reshape(-1, HG_CONTEXT_DIM)
        cursor = 0
        candidate_q = flat[:, cursor:cursor + QUASI_DIM]
        cursor += QUASI_DIM
        edge_present = flat[:, cursor:cursor + 1]
        cursor += 1
        center_q = flat[:, cursor:cursor + QUASI_DIM]
        cursor += QUASI_DIM
        candidate_h = self.quasi_encoder(candidate_q)
        center_h = self.quasi_encoder(center_q)
        mcs_raw = flat[:, cursor:cursor + MAX_MCS_MEMBERS * MCS_SLOT_DIM].reshape(
            -1, MAX_MCS_MEMBERS, MCS_SLOT_DIM
        )
        cursor += MAX_MCS_MEMBERS * MCS_SLOT_DIM
        fcs_raw = flat[:, cursor:cursor + MAX_FCS_MEMBERS * FCS_SLOT_DIM].reshape(
            -1, MAX_FCS_MEMBERS, FCS_SLOT_DIM
        )
        cursor += MAX_FCS_MEMBERS * FCS_SLOT_DIM
        counts = flat[:, cursor:cursor + 2]

        mcs_mask = mcs_raw[..., :1]
        mcs_h = self.mcs_encoder(mcs_raw[..., 1:1 + MCS_DIM])
        mcs_rel = self.qm_rel_encoder(mcs_raw[..., 1 + MCS_DIM:])
        center_m = center_h.unsqueeze(1).expand_as(mcs_h)
        mcs_gate = self.mcs_gate(torch.cat((center_m, mcs_h, mcs_rel), -1))
        mcs_message = self.mcs_message(torch.cat((mcs_h, mcs_rel), -1))
        mcs_context = (mcs_mask * mcs_gate * mcs_message).sum(dim=1)

        fcs_mask = fcs_raw[..., :1]
        fcs_h = self.fcs_encoder(fcs_raw[..., 1:1 + FCS_DIM])
        fcs_rel = self.qf_rel_encoder(fcs_raw[..., 1 + FCS_DIM:])
        center_f = center_h.unsqueeze(1).expand_as(fcs_h)
        fcs_gate = self.fcs_gate(torch.cat((center_f, fcs_h, fcs_rel), -1))
        fcs_message = self.fcs_message(torch.cat((fcs_h, fcs_rel), -1))
        fcs_context = (fcs_mask * fcs_gate * fcs_message).sum(dim=1)
        edge = self.edge_mlp(torch.cat((center_h, mcs_context, fcs_context, counts), -1))
        edge = edge * edge_present
        self._record_diagnostics(
            edge, edge_present, mcs_gate, mcs_mask, fcs_gate, fcs_mask,
            (time.perf_counter() - started) * 1000.0,
        )
        return (
            candidate_h.reshape(*shape, self.hidden_dim),
            edge.reshape(*shape, self.hidden_dim),
            edge_present.reshape(*shape, 1),
        )


class CompetitionActor(nn.Module):
    def __init__(self, self_dim, hidden_dim=128, hg_hidden_dim=64):
        super().__init__()
        self.self_encoder = _MLP(self_dim, hidden_dim)
        self.hg_encoder = CompetitionHyperedgeEncoder(hg_hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + 2 * hg_hidden_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self._logit_rows: list[dict[str, float]] = []

    def logits(self, self_state, candidates):
        packed = candidates[
            ..., BASE_CANDIDATE_DIM:BASE_CANDIDATE_DIM + HG_CONTEXT_DIM
        ]
        candidate_h, edge_h, _ = self.hg_encoder(packed)
        self_h = self.self_encoder(self_state).unsqueeze(1).expand(
            -1, candidates.shape[1], -1
        )
        distance = candidates[..., 1:2]
        controls = candidates[..., -CONTROL_DIM:]
        logits = self.scorer(torch.cat(
            (self_h, candidate_h, edge_h, distance, controls), -1
        )).squeeze(-1)
        with torch.no_grad():
            finite = logits[torch.isfinite(logits)]
            self._logit_rows.append({
                'mean': float(finite.mean().item()) if finite.numel() else 0.0,
                'min': float(finite.min().item()) if finite.numel() else 0.0,
                'max': float(finite.max().item()) if finite.numel() else 0.0,
                'nan_count': float(torch.isnan(logits).sum().item()),
                'inf_count': float(torch.isinf(logits).sum().item()),
            })
        return logits

    def pop_logit_diagnostics(self):
        rows, self._logit_rows = self._logit_rows, []
        result = {}
        for name in ('mean', 'min', 'max', 'nan_count', 'inf_count'):
            result[f'actor_logit_{name}'] = (
                float(np.mean([row[name] for row in rows])) if rows else 0.0
            )
        return result

    def distribution(self, self_state, candidates, mask):
        logits = self.logits(self_state, candidates)
        if not torch.all(mask.any(dim=-1)):
            raise RuntimeError('Every Actor row must have at least one legal action')
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


class CompetitionCritic(nn.Module):
    def __init__(self, original_global_dim, self_dim, hidden_dim=128, hg_hidden_dim=64):
        super().__init__()
        self.original_global_dim = int(original_global_dim)
        self.hg_encoder = CompetitionHyperedgeEncoder(hg_hidden_dim)
        self.self_encoder = _MLP(self_dim, hidden_dim)
        self.value = nn.Sequential(
            nn.Linear(
                original_global_dim + 2 * hg_hidden_dim + 1 + hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state, self_state, candidates, mask):
        original = global_state[..., :self.original_global_dim]
        graph = global_state[..., self.original_global_dim:].reshape(
            -1, MAX_GLOBAL_EDGES, HG_CONTEXT_DIM
        )
        edge_candidate_h, edge_h, edge_present = self.hg_encoder(graph)
        del edge_candidate_h
        edge_mask = edge_present.squeeze(-1) > 0.5
        count = edge_mask.sum(dim=1, keepdim=True)
        denominator = count.clamp_min(1).to(edge_h.dtype)
        mean = (edge_h * edge_mask.unsqueeze(-1)).sum(dim=1) / denominator
        masked = edge_h.masked_fill(~edge_mask.unsqueeze(-1), -torch.inf)
        maximum = masked.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        log_count = torch.log1p(count.to(edge_h.dtype))
        # Preserve the original per-agent state path. Candidate information is
        # intentionally represented through the global HG rather than pooled
        # artificial competition columns.
        own = self.self_encoder(self_state)
        return self.value(torch.cat((original, mean, maximum, log_count, own), -1)).squeeze(-1)
