import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from config import *
from core import EV, MCS, FCS, euclidean_distance
from low_spatial import (
    compute_low_candidate_metrics,
    compute_low_stay_metrics,
    ev_service_opportunity,
    quasi_transition_urgency,
)


# Low 候选 ID：-1 保留给 padding；-2 表示“MCS当前位置”固定动作。
MCS_STAY_CANDIDATE_ID = -2

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
    'local_unserved_opportunity_ratio',
    'near_quasi_count_ratio',
)
MCS_LOW_CANDIDATE_FEATURE_NAMES = (
    'quasi_service_opportunity_ratio',
    'distance_ratio',
    'attraction_ratio',
    'immediate_iev_attraction_ratio',
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

    @staticmethod
    def get_physically_reachable_fcss(
        mcs: MCS,
        all_fcss: List[FCS],
    ) -> List[tuple[FCS, float, float]]:
        """返回严格能量可达的全图物理 FCS，不考虑当前槽位。"""
        reachable: List[tuple[FCS, float, float]] = []
        for fcs in all_fcss:
            distance_km = euclidean_distance(
                mcs.pos[0], mcs.pos[1], fcs.pos[0], fcs.pos[1]
            ) / 1000.0
            required_energy = distance_km * POWER_UNIT
            if float(mcs.remain) > required_energy:
                reachable.append((fcs, distance_km, required_energy))
        reachable.sort(key=lambda item: item[1])
        return reachable

    @staticmethod
    def get_reachable_serve_candidates(
        mcs: MCS,
        all_fcss: List[FCS],
        apply_topk: bool = True,
    ) -> List[EV]:
        """返回安全可达且补电紧急度最高的至多 TopK-1 个 quasi。

        Serve 候选的能量硬约束同时覆盖两段移动：
        ``MCS当前位置 -> quasi -> 距离该quasi最近的FCS``。最近 FCS 从全图
        物理站点中选择，不要求当前存在空闲槽；空闲槽属于未来 Recharge
        匹配条件，不应改变当前 Serve 选位的返程安全性。

        环境执行 Serve 移动时，若剩余电量小于或等于到达目标所需电量，
        MCS 会在途中耗尽电量并进入 broken。因此这里对两段总耗电采用严格
        大于关系，并在紧急度排序与 TopK 截断前过滤。合法候选优先按
        “剩余电量接近 IEV 阈值”的程度排序，再用距离和 ID 打破平局。
        """
        reachable_candidates: List[EV] = []
        if not all_fcss:
            return reachable_candidates

        for quasi in mcs.near_quasi:
            if not quasi.is_quasi:
                continue
            distance_to_quasi_km = euclidean_distance(
                mcs.pos[0], mcs.pos[1], quasi.pos[0], quasi.pos[1]
            ) / 1000.0
            nearest_fcs_distance_km = min(
                euclidean_distance(
                    quasi.pos[0], quasi.pos[1], fcs.pos[0], fcs.pos[1]
                ) / 1000.0
                for fcs in all_fcss
            )
            required_energy = (
                distance_to_quasi_km + nearest_fcs_distance_km
            ) * POWER_UNIT
            if float(mcs.remain) > required_energy:
                reachable_candidates.append(quasi)

        # Top-K 优先保留最接近转变为 IEV 的 quasi。距离只作为同紧急度
        # 下的次级排序，ID 用于保证完全相同时结果可复现。
        reachable_candidates.sort(key=lambda quasi: (
            -quasi_transition_urgency(quasi),
            euclidean_distance(
                mcs.pos[0], mcs.pos[1], quasi.pos[0], quasi.pos[1]
            ),
            int(quasi.id),
        ))
        if not apply_topk:
            return reachable_candidates
        # Low 候选空间始终为“MCS当前位置”预留一个固定槽位。
        quasi_capacity = max(int(TOP_K_MCS_CANDIDATES) - 1, 0)
        return reachable_candidates[:quasi_capacity]

    @staticmethod
    def get_reachable_recharge_candidates(
        mcs: MCS,
        all_fcss: List[FCS],
    ) -> List[tuple[FCS, float, float]]:
        """返回全图中当前有空闲槽且电量能够到达的 FCS。

        Recharge 是全图规划动作，不受 MCS 通信范围限制；通信范围仍只
        用于局部空间观测和竞争特征。本函数只返回当前有空闲槽的
        可执行匹配候选；High Recharge mask 则使用全部物理可达 FCS，
        允许 Recharge option 在槽位竞争失败时保持并重试。
        """
        return [
            item for item in ObservationBuilder.get_physically_reachable_fcss(
                mcs, all_fcss
            )
            if item[0].available_slots > 0
        ]

    def obs_mcs(self, mcs: MCS, all_fcss: Optional[List[FCS]] = None):
        """Build normalized local observations for the MCS high/low actors.

        Mask conventions intentionally differ from the legacy ``mask`` field:
        ``high_action_mask`` and ``low_candidate_mask`` use True for a valid
        choice.  The legacy ``mask`` alias keeps True for padding so existing
        callers are not broken.
        """
        eps = 1e-8
        # Serve 的返程安全检查使用全图物理 FCS；独立调用未传入全图列表时，
        # 兼容性回退到当前已知的局部 available/busy FCS。
        serve_safety_fcss = (
            list(all_fcss)
            if all_fcss is not None
            else list(mcs.near_available_fcs) + list(mcs.near_busy_fcs)
        )
        # 先剔除无法依次到达 quasi 和最近 FCS 的候选，再排序并选择 TopK。
        ranked_safe_candidates = self.get_reachable_serve_candidates(
            mcs, serve_safety_fcss, apply_topk=False
        )
        quasi_capacity = max(int(TOP_K_MCS_CANDIDATES) - 1, 0)
        candidates = ranked_safe_candidates[:quasi_capacity]
        raw_quasi_count = sum(ev.is_quasi for ev in mcs.near_quasi)
        safe_quasi_count = len(ranked_safe_candidates)
        topk_truncated_count = max(safe_quasi_count - quasi_capacity, 0)

        # Recharge 在全图 FCS 上规划；未显式传入时回退到当前已知的全部
        # 局部物理 FCS。候选函数会进一步排除没有空闲槽的 busy FCS。
        recharge_fcss = (
            list(all_fcss)
            if all_fcss is not None
            else list(mcs.near_available_fcs) + list(mcs.near_busy_fcs)
        )
        physically_reachable_fcss = self.get_physically_reachable_fcss(
            mcs, recharge_fcss
        )

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

        # Low 只使用按 EV 数量和紧迫度构造的局部服务缺口，不使用 kWh。
        # High 仍保留上面的能量供需比，用于模式与能源规划，两层语义独立。
        low_opportunity = sum(
            ev_service_opportunity(ev) for ev in mcs.near_quasi
            if ev.is_quasi
        )
        low_alternative_capacity = float(
            len([
                other for other in mcs.near_idle_mcs
                if not other.is_broken
                and not getattr(other, 'is_energy_stranded', False)
            ])
            + sum(fcs.available_slots for fcs in mcs.near_available_fcs)
        )
        low_unserved_opportunity_ratio = (
            low_opportunity
            / (low_opportunity + low_alternative_capacity + eps)
        )

        remain_ratio = float(np.clip(
            mcs.remain / max(MCS_BATTERY_CAPACITY, eps), 0.0, 1.0
        ))
        if physically_reachable_fcss:
            _, nearest_dist_km, required_energy = physically_reachable_fcss[0]
            # 全图候选可能超过通信范围。用地图对角线归一化，避免所有
            # 远端 FCS 的距离特征都被裁剪为 1。
            map_diagonal_km = euclidean_distance(
                AREA_LON_MIN, AREA_LAT_MIN, AREA_LON_MAX, AREA_LAT_MAX
            ) / 1000.0
            nearest_fcs_distance_ratio = (
                nearest_dist_km / max(map_diagonal_km, eps)
            )
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
        quasi_candidate_capacity = max(TOP_K_MCS_CANDIDATES - 1, 1)

        high_state = np.asarray([
            remain_ratio,
            np.clip(nearest_fcs_distance_ratio, 0.0, 1.0),
            np.clip(recharge_margin_ratio, 0.0, 1.0),
            np.clip(local_available_slots / local_slot_capacity, 0.0, 1.0),
            np.clip(local_need_supply_ratio, 0.0, 1.0),
            np.clip(len(candidates) / quasi_candidate_capacity, 0.0, 1.0),
            np.clip(
                near_iev_demand /
                max(TOP_K_MCS_CANDIDATES * MAX_CHARGE_PER_SESSION_KWH, eps),
                0.0,
                1.0,
            ),
        ], dtype=np.float32)
        recharge_only = bool(
            physically_reachable_fcss
            and not float(mcs.remain) > (
                float(MCS_SERVE_SAFETY_RESERVE_KWH)
                + float(physically_reachable_fcss[0][2])
            )
        )
        high_action_mask = np.asarray([
            not recharge_only,                  # 0 = Serve；只在绝对安全域内可用
            bool(physically_reachable_fcss),    # 1 = Recharge；槽位竞争由匹配器处理
        ], dtype=bool)

        if TOP_K_MCS_CANDIDATES < 1:
            raise RuntimeError('TOP_K_MCS_CANDIDATES 至少为 1，需容纳当前位置动作')

        # index=0 永远是 MCS 当前位置。存在其他 quasi 时选择它表示主动
        # 等待；没有其他 quasi 时它是唯一合法候选，表示被动等待。
        stay_metrics = compute_low_stay_metrics(mcs)
        candidate_features = [[
            stay_metrics['urgency_demand'],
            stay_metrics['distance_ratio'],
            stay_metrics['attraction'],
            stay_metrics['immediate_iev_attraction'],
            stay_metrics['mcs_competition'],
            stay_metrics['fcs_competition'],
        ]]
        candidate_ids = [MCS_STAY_CANDIDATE_ID]
        candidate_is_stay = [True]
        candidate_urgencies = [0.0]
        candidate_desirabilities = [stay_metrics['desirability']]
        candidate_remain_kwh = [-1.0]
        candidate_need_power_kwh = [-1.0]
        for quasi in candidates:
            metrics = compute_low_candidate_metrics(mcs, quasi)
            candidate_features.append([
                metrics['urgency_demand'],
                metrics['distance_ratio'],
                metrics['attraction'],
                metrics['immediate_iev_attraction'],
                metrics['mcs_competition'],
                metrics['fcs_competition'],
            ])
            candidate_ids.append(int(quasi.id))
            candidate_is_stay.append(False)
            candidate_urgencies.append(quasi_transition_urgency(quasi))
            candidate_desirabilities.append(metrics['desirability'])
            candidate_remain_kwh.append(float(quasi.remain))
            candidate_need_power_kwh.append(float(quasi.need_power))

        valid_count = len(candidate_features)
        while len(candidate_features) < TOP_K_MCS_CANDIDATES:
            candidate_features.append([0.0] * MCS_FEAT_DIM_tgt)
            candidate_ids.append(-1)
            candidate_is_stay.append(False)
            candidate_urgencies.append(0.0)
            candidate_desirabilities.append(0.0)
            candidate_remain_kwh.append(-1.0)
            candidate_need_power_kwh.append(-1.0)

        low_candidates = np.asarray(candidate_features, dtype=np.float32)
        low_candidate_mask = np.zeros(TOP_K_MCS_CANDIDATES, dtype=bool)
        low_candidate_mask[:valid_count] = True
        low_self_state = np.asarray([
            remain_ratio,
            np.clip(low_unserved_opportunity_ratio, 0.0, 1.0),
            np.clip(len(candidates) / quasi_candidate_capacity, 0.0, 1.0),
        ], dtype=np.float32)

        return {
            'high_state': high_state,
            'high_action_mask': high_action_mask,
            'high_recharge_only': recharge_only,
            'low_self_state': low_self_state,
            'low_candidates': low_candidates,
            'low_candidate_mask': low_candidate_mask,
            'candidate_ids': np.asarray(candidate_ids, dtype=np.int64),
            'candidate_is_stay': np.asarray(candidate_is_stay, dtype=bool),
            # 以下数组仅用于审计日志，不进入 Actor/Critic 张量。
            'candidate_urgencies': np.asarray(
                candidate_urgencies, dtype=np.float32
            ),
            'candidate_desirabilities': np.asarray(
                candidate_desirabilities, dtype=np.float32
            ),
            'candidate_remain_kwh': np.asarray(
                candidate_remain_kwh, dtype=np.float32
            ),
            'candidate_need_power_kwh': np.asarray(
                candidate_need_power_kwh, dtype=np.float32
            ),
            'raw_quasi_count': int(raw_quasi_count),
            'safe_quasi_count': int(safe_quasi_count),
            'topk_truncated_count': int(topk_truncated_count),
            'quasi_candidate_count': int(len(candidates)),
            'stay_candidate_index': 0,
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
                float(
                    mcs.is_idle
                    and not mcs.is_broken
                    and not mcs.is_energy_stranded
                    and not mcs.is_recharging
                ),
                float(mcs.is_task),
                float(mcs.is_recharging),
                float(mcs.is_broken or mcs.is_energy_stranded),
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
