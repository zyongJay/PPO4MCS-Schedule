from typing import Any, Dict, List
import numpy as np
from config import *
from core import EV, MCS, FCS, euclidean_distance


# ============================================================
# ObservationBuilder
# ============================================================
class ObservationBuilder:
    """观测构建器 (Target-based, GNN 占位)

    为 MCS / IEV 智能体构建结构化观测 dict，
    结构兼容 env/world.py 的 get_agent_obs()。
    """

    # ──────────────────────────────────────────
    # MCS 观测
    # ──────────────────────────────────────────
    def obs_mcs(self, mcs: MCS):
        """
        针对 MCS-Low-Policy Actor
        为 MCS智能体 构造观测特征矩阵，包括附近K_max个候选对象的特征 以及 自身的特征
        同时返回掩码向量
        """
        features = []  # 特征矩阵   k_max * d_in_target(5)
        mask = []  # mask向量  k_max
        candidates: List[EV] = [ev for ev in mcs.near_quasi]

        # 按照规则排序
        if TOP_K_MCS_CANDIDATES and len(candidates) > 1:
            candidates.sort(key=lambda ev: euclidean_distance(
                mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1]))
        # 取Top-K
        if TOP_K_MCS_CANDIDATES and len(candidates) > TOP_K_MCS_CANDIDATES:
            candidates = candidates[:TOP_K_MCS_CANDIDATES]

        # 构建观测特征矩阵：
        for quasi in candidates:
            # mcs-quasi 交互信息
            dist_km = euclidean_distance(
                mcs.pos[0], mcs.pos[1],
                quasi.pos[0], quasi.pos[1]) / 1000.0
            # 周围环境感知
            attraction: float = 0.0
            competition_mcs: float = 0.0
            competition_fcs: float = 0.0
            attractors: List[EV] = quasi.near_quasi + quasi.near_iev
            # 附近的IEV and quasi
            for other in attractors:
                d_km = euclidean_distance(
                    quasi.pos[0], quasi.pos[1],
                    other.pos[0], other.pos[1]) / 1000.0
                if other.is_iev:  # 附近有IEV更加分
                    attraction += 1.5 * other.need_power / (d_km + 1)
                else:
                    attraction += other.need_power / (d_km + 1)
            # 附近的idle MCS
            for cpt in quasi.near_idle_mcs:
                d_km = euclidean_distance(
                    quasi.pos[0], quasi.pos[1],
                    cpt.pos[0], cpt.pos[1]) / 1000.0
                competition_mcs += cpt.remain / (d_km + 1)
            # 附近的avail FCS
            for cpt in quasi.near_available_fcs:
                d_km = euclidean_distance(
                    quasi.pos[0], quasi.pos[1],
                    cpt.pos[0], cpt.pos[1]) / 1000.0
                competition_fcs += cpt.available_slots / (d_km + 1)
            features.append([quasi.need_power, dist_km, attraction, competition_mcs, competition_fcs])
            mask.append(False)

        # MCS智能体自身的特征向量（d_self(3)）
        need: float = 0.0
        support: float = 0.0
        nearest_dist = 99
        for fcs in mcs.near_available_fcs:
            dist_km = euclidean_distance(
                mcs.pos[0], mcs.pos[1],
                fcs.pos[0], mcs.pos[1]
            ) / 1000.0
            if dist_km < nearest_dist:
                nearest_dist = dist_km
            support += MAX_CHARGE_TIME_MIN * fcs.available_slots
        for m in mcs.near_idle_mcs:
            support += m.remain
        for ev in mcs.near_iev:
            need += ev.need_power
        for ev in mcs.near_quasi:
            weight = min(EV_LOW_POWER_THRESHOLD / ev.remain, 1)
            need += ev.need_power * weight
        # 1.紧急程度
        warn_degree = mcs.remain / (1 + (POWER_UNIT * max(0.01, nearest_dist)))
        # 2.附近电量供求比例
        ns_rate = need / (1 + support)

        # 候选对象数量不足k_max时，用0补齐，mask矩阵
        if len(candidates) < TOP_K_MCS_CANDIDATES:
            pad_num = TOP_K_MCS_CANDIDATES - len(candidates)
            for _ in range(pad_num):
                zero_feature = [0.0] * MCS_FEAT_DIM_tgt
                features.append(zero_feature)
                mask.append(True)

        obs_tgt = np.array(features)
        obs_self = np.array([mcs.remain, warn_degree, ns_rate])
        obs = {"obs_self": obs_self,  # 观测特征向量 d_in_self
               "obs_tgt": obs_tgt,  # 观测特征向量 k_max * d_in_target
               "done": False}
        mask_arr = np.array(mask, dtype=bool)

        return obs, mask_arr

    # ──────────────────────────────────────────
    # IEV 观测
    # ──────────────────────────────────────────

    def build_iev_obs(self, ev: EV) -> Dict[str, Any]:
        """为 IEV 构建观测。
        先只做MCS的调度，后续扩展补充
        """
        pass

    # ──────────────────────────────────────────
    # 全局状态
    # ──────────────────────────────────────────
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

