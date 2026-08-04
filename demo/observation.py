import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from config import *
from core import EV, MCS, FCS, euclidean_distance

MCS_HIGH_FEATURE_NAMES = (
    'remain_ratio',
    'nearest_fcs_distance_ratio',
    'recharge_margin_ratio',
    'local_available_slot_ratio',
    'local_need_supply_ratio',
    'near_quasi_count_ratio',
    'near_iev_demand_ratio',
)
MCS_LOW_SELF_FEATURE_NAMES = (
    'remain_ratio',
    'local_need_supply_ratio',
    'near_quasi_count_ratio',
)
MCS_LOW_CANDIDATE_FEATURE_NAMES = (
    'need_power_ratio',
    'distance_ratio',
    'attraction_ratio',
    'competition_mcs_ratio',
    'competition_fcs_ratio',
)


# ============================================================
# ObservationBuilder
# ============================================================
class ObservationBuilder:
    """观测构建器 (Target-based, GNN 占位)
    为 MCS / IEV 智能体构建结构化观测 dict，
    结构兼容 env/world.py 的 get_agent_obs()。
    """

    def obs_mcs(self, mcs: MCS):
        """Build normalized local observations for the MCS high/low actors.

        Mask conventions intentionally differ from the legacy ``mask`` field:
        ``high_action_mask`` and ``low_candidate_mask`` use True for a valid
        choice.  The legacy ``mask`` alias keeps True for padding so existing
        callers are not broken.
        """
        eps = 1e-8
        candidates: List[EV] = [ev for ev in mcs.near_quasi if ev.is_quasi]
        candidates.sort(
            key=lambda ev: euclidean_distance(
                mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1]
            )
        )
        candidates = candidates[:TOP_K_MCS_CANDIDATES]

        # Only local, physically valid FCSs may enable the Recharge action.
        reachable_fcs = []
        for fcs in mcs.near_available_fcs:
            dist_km = euclidean_distance(
                mcs.pos[0], mcs.pos[1], fcs.pos[0], fcs.pos[1]
            ) / 1000.0
            required_energy = dist_km * POWER_UNIT
            if dist_km <= COMM_RANGE and mcs.remain > required_energy:
                reachable_fcs.append((fcs, dist_km, required_energy))

        need = sum(ev.need_power for ev in mcs.near_iev)
        for ev in mcs.near_quasi:
            remain = max(float(ev.remain), eps)
            need += min(EV_LOW_POWER_THRESHOLD / remain, 1.0) * ev.need_power

        support = sum(
            MAX_CHARGE_PER_SESSION_KWH * fcs.available_slots
            for fcs in mcs.near_available_fcs
        )
        support += sum(
            min(other.remain, MAX_CHARGE_PER_SESSION_KWH)
            for other in mcs.near_idle_mcs
        )
        local_need_supply_ratio = need / (need + support + eps)

        remain_ratio = float(np.clip(
            mcs.remain / max(MCS_BATTERY_CAPACITY, eps), 0.0, 1.0
        ))
        if reachable_fcs:
            _, nearest_dist_km, required_energy = min(
                reachable_fcs, key=lambda item: item[1]
            )
            nearest_fcs_distance_ratio = nearest_dist_km / max(COMM_RANGE, eps)
            recharge_margin_ratio = (
                mcs.remain - required_energy
            ) / max(MCS_BATTERY_CAPACITY, eps)
        else:
            nearest_fcs_distance_ratio = 1.0
            recharge_margin_ratio = 0.0

        local_slot_capacity = max(
            len(mcs.near_available_fcs) * FCS_SLOTS_PER_STATION, 1
        )
        local_available_slots = sum(
            fcs.available_slots for fcs in mcs.near_available_fcs
        )
        near_iev_demand = sum(ev.need_power for ev in mcs.near_iev)

        high_state = np.asarray([
            remain_ratio,
            np.clip(nearest_fcs_distance_ratio, 0.0, 1.0),
            np.clip(recharge_margin_ratio, 0.0, 1.0),
            np.clip(local_available_slots / local_slot_capacity, 0.0, 1.0),
            np.clip(local_need_supply_ratio, 0.0, 1.0),
            np.clip(len(mcs.near_quasi) / max(TOP_K_MCS_CANDIDATES, 1), 0.0, 1.0),
            np.clip(
                near_iev_demand /
                max(TOP_K_MCS_CANDIDATES * MAX_CHARGE_PER_SESSION_KWH, eps),
                0.0,
                1.0,
            ),
        ], dtype=np.float32)
        high_action_mask = np.asarray([
            bool(candidates),       # 0 = Serve
            bool(reachable_fcs),    # 1 = Recharge
            True,                   # 2 = Wait
        ], dtype=bool)

        candidate_features = []
        candidate_ids = []
        for quasi in candidates:
            dist_km = euclidean_distance(
                mcs.pos[0], mcs.pos[1], quasi.pos[0], quasi.pos[1]
            ) / 1000.0
            attraction = 0.0
            competition_mcs = 0.0
            competition_fcs = 0.0

            for other in quasi.near_quasi + quasi.near_iev:
                other_dist_km = euclidean_distance(
                    quasi.pos[0], quasi.pos[1], other.pos[0], other.pos[1]
                ) / 1000.0
                weight = 1.5 if other.is_iev else 1.0
                attraction += weight * other.need_power / (other_dist_km + 1.0)

            for other in quasi.near_idle_mcs:
                if other.id == mcs.id:
                    continue
                other_dist_km = euclidean_distance(
                    quasi.pos[0], quasi.pos[1], other.pos[0], other.pos[1]
                ) / 1000.0
                competition_mcs += (
                    min(other.remain, MAX_CHARGE_PER_SESSION_KWH) /
                    max(MAX_CHARGE_PER_SESSION_KWH, eps) /
                    (other_dist_km + 1.0)
                )

            for fcs in quasi.near_available_fcs:
                other_dist_km = euclidean_distance(
                    quasi.pos[0], quasi.pos[1], fcs.pos[0], fcs.pos[1]
                ) / 1000.0
                competition_fcs += fcs.available_slots / (other_dist_km + 1.0)

            candidate_features.append([
                np.clip(quasi.need_power / max(MAX_CHARGE_PER_SESSION_KWH, eps), 0.0, 1.0),
                np.clip(dist_km / max(COMM_RANGE, eps), 0.0, 1.0),
                np.clip(attraction / (attraction + MAX_CHARGE_PER_SESSION_KWH + eps), 0.0, 1.0),
                np.clip(competition_mcs / (competition_mcs + 1.0), 0.0, 1.0),
                np.clip(competition_fcs / (competition_fcs + 1.0), 0.0, 1.0),
            ])
            candidate_ids.append(int(quasi.id))

        valid_count = len(candidate_features)
        while len(candidate_features) < TOP_K_MCS_CANDIDATES:
            candidate_features.append([0.0] * MCS_FEAT_DIM_tgt)
            candidate_ids.append(-1)

        low_candidates = np.asarray(candidate_features, dtype=np.float32)
        low_candidate_mask = np.zeros(TOP_K_MCS_CANDIDATES, dtype=bool)
        low_candidate_mask[:valid_count] = True
        low_self_state = np.asarray([
            remain_ratio,
            np.clip(local_need_supply_ratio, 0.0, 1.0),
            np.clip(len(mcs.near_quasi) / max(TOP_K_MCS_CANDIDATES, 1), 0.0, 1.0),
        ], dtype=np.float32)

        return {
            'high_state': high_state,
            'high_action_mask': high_action_mask,
            'low_self_state': low_self_state,
            'low_candidates': low_candidates,
            'low_candidate_mask': low_candidate_mask,
            'candidate_ids': np.asarray(candidate_ids, dtype=np.int64),
            # Backward-compatible aliases used by the current baseline script.
            'obs_self': low_self_state,
            'obs_tgt': low_candidates,
            'mask': np.logical_not(low_candidate_mask),
            'done': False,
        }

    def obs_iev(self, iev: EV):
        """
        为 IEV 构建观测。
        """
        candidates: List[Optional[Union[FCS, MCS]]] = (iev.near_task_mcs or []) + (iev.near_busy_fcs or [])
        # 按照规则排序
        if TOP_K_IEV_CANDIDATES and len(candidates) > 1:
            candidates.sort(key=lambda target: euclidean_distance(
                iev.pos[0], iev.pos[1], target.pos[0], target.pos[1]))
        # 取Top-K
        if TOP_K_IEV_CANDIDATES and len(candidates) > TOP_K_IEV_CANDIDATES:
            candidates = candidates[:TOP_K_IEV_CANDIDATES]

        # 构建观测特征矩阵obs_tgt:
        features = []  # 特征矩阵   k_max * d_in_target(4)
        mask = []  # mask向量  k_max
        for target in candidates:
            # 1.剩余充电时间
            if isinstance(target, MCS):
                remain_time_min = target.charge_time_remain_min
            else:
                non_zero_remain = [x for x in target.slot_charge_remain_min if x != 0]
                remain_time_min = min(non_zero_remain) if non_zero_remain else 99
            # 2.二者距离
            dist_km = euclidean_distance(
                iev.pos[0], iev.pos[1],
                target.pos[0], target.pos[1]) / 1000.0
            # 3.附近 竞争 与 吸引
            competition: float = 0.0
            attraction: float = 0.0
            attractors: List[Optional[Union[MCS, FCS]]] = target.near_idle_mcs + target.near_available_fcs
            # 吸引: 目的地周围的 idle MCS 和 avail FCS
            for other in attractors:
                d_km = euclidean_distance(
                    target.pos[0], target.pos[1],
                    other.pos[0], other.pos[1]) / 1000.0
                if isinstance(other, MCS):  # idle MCS更优
                    attraction += 1.2 * min(other.remain, MAX_CHARGE_PER_SESSION_KWH) / (d_km + 1)
                else:
                    attraction += MAX_CHARGE_PER_SESSION_KWH * other.available_slots / (d_km + 1)
            # 竞争: 目的地周围的 iev
            for cpt in target.near_iev:
                d_km = euclidean_distance(
                    target.pos[0], target.pos[1],
                    cpt.pos[0], cpt.pos[1]) / 1000.0
                competition += cpt.need_power / (d_km + 1)
            features.append([remain_time_min, dist_km, attraction, competition])
            mask.append(False)
        obs_tgt = np.array(features)

        # IEV智能体自身的特征向量obs_self:
        obs_self = np.array([iev.remain, iev.total_wait_time_min])

        # 候选对象数量不足k_max时，用0补齐，mask矩阵
        if len(candidates) < TOP_K_MCS_CANDIDATES:
            pad_num = TOP_K_MCS_CANDIDATES - len(candidates)
            for _ in range(pad_num):
                zero_feature = [0.0] * MCS_FEAT_DIM_tgt
                features.append(zero_feature)
                mask.append(True)

        mask_arr = np.array(mask, dtype=bool)
        obs = {"obs_self": obs_self,  # 观测特征向量 d_in_self
               "obs_tgt": obs_tgt,  # 观测特征向量 k_max * d_in_target
               "mask": mask_arr,  # 掩码向量 k_max
               "done": False}

        return obs

    def build_global_state(
        self,
        EVs: List[EV],
        MCSs: List[MCS],
        FCSs: List[FCS],
    ) -> np.ndarray:
        """Build a normalized, permutation-invariant 20-D critic state.

        The actor remains local.  The centralized critic receives mean/max
        pooling over seven per-MCS features plus six system summaries, so its
        width is independent of NUM_MCS and no IEV policy output is required.
        """
        eps = 1e-8
        per_mcs = []
        profit_scale = max(
            MAX_CHARGE_PER_SESSION_KWH * CHARGE_PRICE * MAX_STEPS_PER_EPISODE,
            1.0,
        )
        for mcs in MCSs:
            per_mcs.append([
                np.clip(mcs.remain / max(MCS_BATTERY_CAPACITY, eps), 0.0, 1.0),
                float(mcs.is_idle and not mcs.is_broken and not mcs.is_recharging),
                float(mcs.is_task),
                float(mcs.is_recharging),
                float(mcs.is_broken),
                np.clip(
                    mcs.charge_power_kwh /
                    max(MAX_CHARGE_PER_SESSION_KWH, eps),
                    0.0,
                    1.0,
                ),
                np.tanh(mcs.total_profit / profit_scale),
            ])

        if per_mcs:
            mcs_matrix = np.asarray(per_mcs, dtype=np.float32)
            pooled_mcs = np.concatenate(
                (mcs_matrix.mean(axis=0), mcs_matrix.max(axis=0))
            )
        else:
            pooled_mcs = np.zeros(14, dtype=np.float32)

        total_evs = max(len(EVs), 1)
        quasi_ratio = sum(ev.is_quasi for ev in EVs) / total_evs
        iev_ratio = sum(ev.is_iev for ev in EVs) / total_evs
        success_count = sum(
            ev.is_charged and ev.charge_pos is None for ev in EVs
        )
        failure_count = sum(ev.fail_charge for ev in EVs)
        finished_count = success_count + failure_count
        success_rate = success_count / finished_count if finished_count else 0.0
        failure_rate = failure_count / finished_count if finished_count else 0.0
        total_slots = max(sum(fcs.capacity for fcs in FCSs), 1)
        available_slot_ratio = sum(
            fcs.available_slots for fcs in FCSs
        ) / total_slots
        active_demands = [
            ev.need_power for ev in EVs if ev.is_quasi or ev.is_iev
        ]
        mean_demand_ratio = (
            np.mean(active_demands) / max(MAX_CHARGE_PER_SESSION_KWH, eps)
            if active_demands else 0.0
        )
        system_summary = np.asarray([
            np.clip(quasi_ratio, 0.0, 1.0),
            np.clip(iev_ratio, 0.0, 1.0),
            np.clip(available_slot_ratio, 0.0, 1.0),
            np.clip(success_rate, 0.0, 1.0),
            np.clip(failure_rate, 0.0, 1.0),
            np.clip(mean_demand_ratio, 0.0, 1.0),
        ], dtype=np.float32)
        state = np.concatenate((pooled_mcs, system_summary)).astype(np.float32)
        if state.shape != (MCS_GLOBAL_STATE_DIM,):
            raise RuntimeError(f'unexpected critic state shape: {state.shape}')
        return state
