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

    def build_global_state(self, EVs: List[EV], MCSs: List[MCS], FCSs: List[FCS]) -> np.ndarray:
        """
        构造 Critic 使用的 6 维全局状态摘要。
        特征顺序为 IEV 竞争、MCS 竞争、IEV 吸引、MCS 吸引、
        成功充电率和 MCS 平均收益。距离以 km 为单位，统计范围使用
        ``COMM_RANGE``。FCS 的 e 使用当前可服务电量表示。
        """
        region_km = float(2 * COMM_RANGE)
        min_distance_km = 1e-6
        ievs = [ev for ev in EVs if ev.is_iev]
        demand_evs = [ev for ev in EVs if ev.is_iev or ev.is_quasi]
        active_mcss = [mcs for mcs in MCSs if not mcs.is_broken and not mcs.is_recharging and mcs.is_idle]

        def distance_km(obj_a, obj_b) -> float:
            return max(
                euclidean_distance(
                    obj_a.pos[0], obj_a.pos[1],
                    obj_b.pos[0], obj_b.pos[1],
                ) / 1000.0,
                min_distance_km,
            )

        def mean_or_zero(values: List[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        iev_competition_values: List[float] = []
        for iev in ievs:
            competition = 0.0
            for other in ievs:
                if other is iev:
                    continue
                dist = distance_km(iev, other)
                if dist <= region_km:
                    competition += float(other.need_power) / (dist + 1)
            iev_competition_values.append(competition)
        iev_competition = mean_or_zero(iev_competition_values)

        mcs_competition_values: List[float] = []
        for mcs in active_mcss:
            competition = 0.0
            for other in active_mcss:
                if other is mcs:
                    continue
                dist = distance_km(mcs, other)
                if dist <= region_km:
                    competition += float(other.remain) / (dist + 1)
            for fcs in FCSs:
                dist = distance_km(mcs, fcs)
                if dist <= region_km:
                    available_energy = (
                            float(fcs.available_slots)
                            * float(MAX_CHARGE_PER_SESSION_KWH)
                    )
                    competition += available_energy / (dist + 1)
            mcs_competition_values.append(competition)
        mcs_competition = mean_or_zero(mcs_competition_values)

        iev_attraction_values: List[float] = []
        for iev in ievs:
            attraction = 0.0
            for mcs in active_mcss:
                dist = distance_km(iev, mcs)
                if dist <= region_km:
                    attraction += float(mcs.remain) / (dist + 1)
            iev_attraction_values.append(attraction)
        iev_attraction = mean_or_zero(iev_attraction_values)

        mcs_attraction_values: List[float] = []
        for mcs in active_mcss:
            attraction = 0.0
            for ev in demand_evs:
                dist = distance_km(mcs, ev)
                if dist <= region_km:
                    if ev.is_iev:
                        attraction += 1.5 * float(ev.need_power) / (dist + 1)
                    else:
                        attraction += float(ev.need_power) / (dist + 1)
            mcs_attraction_values.append(attraction)
        mcs_attraction = mean_or_zero(mcs_attraction_values)

        n_success = sum(
            1 for ev in EVs if ev.is_charged and ev.charge_pos is None
        )
        n_fail = sum(1 for ev in EVs if ev.fail_charge)
        n_finished = n_success + n_fail
        success_rate = n_success / n_finished if n_finished > 0 else 0.0

        avg_mcs_profit = (
            sum(mcs.total_profit for mcs in MCSs) / len(MCSs)
            if MCSs
            else 0.0
        )

        return np.asarray(
            [
                iev_competition,
                mcs_competition,
                iev_attraction,
                mcs_attraction,
                success_rate,
                avg_mcs_profit,
            ],
            dtype=np.float32,
        )
