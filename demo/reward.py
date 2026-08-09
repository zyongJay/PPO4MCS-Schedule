"""
MCS 奖励计算器。

当前奖励分为两个互相独立的层级：

1. step 级奖励：评价本次动作带来的即时状态改善。
   - Serve：评价选位后的需求覆盖、竞争情况和移动能耗。
   - Recharge：评价低电量风险是否下降。
   - Wait：按当前位置可服务需求的大小计算机会成本。
2. event 级奖励：评价离散业务事件。
   - MCS 成功事件只奖励实际服务该 EV 的责任 MCS。
   - 可控失败事件按即时服务能力在可行 MCS 间分配责任惩罚。
   - FCS 成功和无可行 MCS 的外生失败只记录 KPI，不进入 MCS 回报。
   - MCS 与 IEV 的服务匹配奖励只保留计算和日志，不进入实际回报。
   - Recharge 请求匹配成功或失败分别给出小额反馈。
   - MCS 首次 broken 时给出一次性惩罚。

各模式的 step/event 分量分别返回，便于后续为 High Actor 和 Low Actor
拆分 Critic。当前共享 Critic 使用 ``total``；未来可以直接使用
``high_step/high_event`` 和 ``low_step/low_event`` 构造各自的回报。

本文件暂时集中保存全部奖励参数，便于奖励实验时只修改 reward.py。
"""

from typing import Dict, List

import numpy as np

from config import (
    COMM_RANGE,
    EV_LOW_POWER_THRESHOLD,
    MAX_CHARGE_PER_SESSION_KWH,
    MAX_MOVE_PER_STEP,
    MCS_RECHARGE_THRESHOLD,
    MOVE_SPEED,
    POWER_UNIT,
    TOP_K_MCS_CANDIDATES,
)
from core import EV, FCS, MCS, euclidean_distance


# =============================================================================
# 奖励参数
# =============================================================================

# 总奖励中 step/event 两类信号的权重。业务事件占主导，step 奖励只负责
# 提供稠密的选位反馈和缩短信任分配路径。
STEP_REWARD_WEIGHT = 0.20
EVENT_REWARD_WEIGHT = 0.80

# 空间需求与竞争力参数。
ENERGY_REFERENCE_KWH = max(float(MAX_CHARGE_PER_SESSION_KWH), 1e-8)
DISTANCE_DECAY_KM = max(float(COMM_RANGE) / 2.0, 1e-8)
IEV_DEMAND_WEIGHT = 2.0
FCS_COMPETITION_WEIGHT = 1.0
POSITION_DELTA_REFERENCE = 0.25
# 选位塑形只用于缓解团队事件奖励的稀疏性，不应主导训练目标。
POSITION_SHAPING_WEIGHT = 0.25
SERVE_MOVE_PENALTY = 0.10

# 服务事件由“成功服务一辆 IEV”和“服务电量”共同组成。固定成功奖励
# 占比较高，电量项采用平方根压缩，防止策略只追逐高需求量 IEV。
SERVICE_SUCCESS_WEIGHT = 0.60
SERVICE_ENERGY_WEIGHT = 0.40

# 责任事件的系统总量。成功奖励只给实际服务 MCS；可控失败惩罚按责任
# 权重归一化分配。每个事件在全部 MCS 上的权重和不超过 1。
SYSTEM_SUCCESS_EVENT_REWARD = 1.0
SYSTEM_FAILURE_EVENT_PENALTY = 1.1

# Recharge/Wait 是 High Actor 的独立模式，分别保留自己的 step/event
# 奖励，避免将来拆分 Critic 时需要重新定义奖励语义。
RECHARGE_STEP_WEIGHT = 0.20
RECHARGE_MATCH_WEIGHT = 0.20
RECHARGE_MATCH_FAILURE_PENALTY = 0.05

# Wait 惩罚仅针对主动等待。forced Wait 不处罚；主动等待具有固定成本、
# 最佳合法 Serve 候选的机会成本，以及连续主动等待的递增成本。
FORCED_WAIT_TIME_PENALTY = 0.0
WAIT_BASE_PENALTY = 0.03
WAIT_OPPORTUNITY_PENALTY = 0.07
WAIT_STREAK_PENALTY = 0.05
WAIT_STREAK_NORMALIZER = 10

# broken 只在状态首次变为 broken 时处罚一次。
BROKEN_EVENT_PENALTY = 2.0

EPSILON = 1e-8


class RewardBuilder:
    """构建与当前 World 接口兼容的 MCS 奖励。"""

    @staticmethod
    def _distance_km(first, second) -> float:
        """返回两个实体之间的千米距离。"""
        return float(euclidean_distance(
            first.pos[0], first.pos[1], second.pos[0], second.pos[1]
        ) / 1000.0)

    @staticmethod
    def _distance_kernel(distance_km: float) -> float:
        """无量纲距离衰减核，避免直接做“电量/距离”造成量纲混合。"""
        distance_km = max(float(distance_km), 0.0)
        return float(np.exp(-distance_km / DISTANCE_DECAY_KM))

    @staticmethod
    def _energy_ratio(energy_kwh: float, reference_kwh: float) -> float:
        """把电量裁剪并归一化到 [0, 1]。"""
        return float(np.clip(
            max(float(energy_kwh), 0.0) / max(float(reference_kwh), EPSILON),
            0.0,
            1.0,
        ))

    @staticmethod
    def _saturating_energy_ratio(
        energy_kwh: float,
        reference_kwh: float,
    ) -> float:
        """无硬截断的电量归一化，保留大于参考电量后的差异。

        当前仿真中的匹配充电量可能超过 MAX_CHARGE_PER_SESSION_KWH。
        使用 q / (q + q_ref) 可将结果限制在 [0, 1)，同时避免把所有
        大任务都裁剪成完全相同的电量奖励。
        """
        energy_kwh = max(float(energy_kwh), 0.0)
        reference_kwh = max(float(reference_kwh), EPSILON)
        return float(energy_kwh / (energy_kwh + reference_kwh))

    @staticmethod
    def _mcs_service_capacity(mcs: MCS) -> float:
        """MCS 扣除安全储备后，可用于服务的归一化电量。"""
        usable_energy = max(
            float(mcs.remain) - float(MCS_RECHARGE_THRESHOLD),
            0.0,
        )
        return RewardBuilder._energy_ratio(
            usable_energy, ENERGY_REFERENCE_KWH
        )

    @staticmethod
    def _battery_risk(remain_kwh: float) -> float:
        """低电量风险；达到两倍强制补电阈值后风险为 0。"""
        safe_energy = max(2.0 * float(MCS_RECHARGE_THRESHOLD), EPSILON)
        return float(np.clip(
            (safe_energy - float(remain_kwh)) / safe_energy,
            0.0,
            1.0,
        ))

    def _quasi_demand(self, quasi: EV) -> Dict[str, float]:
        """计算一个 quasi 节点的潜在需求和附近 IEV 的即时需求。

        quasi 自身需求乘以低电量紧迫度；附近 IEV 的即时需求具有更高
        权重。所有电量和距离均先转换成无量纲量。
        """
        quasi_need = self._saturating_energy_ratio(
            quasi.need_power, ENERGY_REFERENCE_KWH
        )
        quasi_urgency = float(np.clip(
            float(EV_LOW_POWER_THRESHOLD) /
            max(float(quasi.remain), EPSILON),
            0.0,
            1.0,
        ))
        potential_demand = quasi_need * quasi_urgency

        immediate_demand = 0.0
        # 使用环境已经构建好的 quasi.near_iev，既符合局部信息边界，也
        # 避免在每次奖励计算中重复执行全体 EV 的两两距离搜索。
        seen_iev_ids = set()
        for iev in getattr(quasi, 'near_iev', []):
            if not iev.is_iev or iev.id in seen_iev_ids:
                continue
            seen_iev_ids.add(iev.id)
            distance_km = self._distance_km(quasi, iev)
            if distance_km > COMM_RANGE:
                continue
            immediate_demand += (
                IEV_DEMAND_WEIGHT
                * self._saturating_energy_ratio(
                    iev.need_power, ENERGY_REFERENCE_KWH
                )
                * self._distance_kernel(distance_km)
            )

        return {
            'potential': float(potential_demand),
            'immediate': float(immediate_demand),
            'total': float(potential_demand + immediate_demand),
        }

    def compute_mcs_spatial_features(
        self,
        mcs: MCS,
        all_evs: List[EV],
        all_mcss: List[MCS],
        all_fcss: List[FCS],
    ) -> Dict[str, float]:
        """计算 MCS 当前选位的无量纲需求力、竞争力和综合潜力。

        World 在动作执行前后都会调用本函数。函数只使用通信范围内的
        quasi 节点，并利用需求节点附近的空闲 MCS/FCS 估计竞争供给。
        返回值全部有界，便于和事件奖励设置稳定、可解释的权重。
        """
        attraction_raw = 0.0
        competition_raw = 0.0
        potential_raw = 0.0
        quasi_demand_raw = 0.0
        immediate_iev_demand_raw = 0.0

        own_capacity = self._mcs_service_capacity(mcs)

        for quasi in all_evs:
            if not quasi.is_quasi:
                continue

            mcs_distance_km = self._distance_km(mcs, quasi)
            if mcs_distance_km > COMM_RANGE:
                continue

            demand = self._quasi_demand(quasi)
            total_demand = demand['total']
            if total_demand <= 0.0:
                continue

            quasi_demand_raw += demand['potential']
            immediate_iev_demand_raw += demand['immediate']

            own_access = (
                own_capacity * self._distance_kernel(mcs_distance_km)
            )

            competitor_access = 0.0
            # 优先使用环境局部邻居；若调用方构造的是最小测试对象且尚未
            # 建立邻居列表，则回退到传入的全体实体并执行相同距离过滤。
            local_mcss = getattr(quasi, 'near_idle_mcs', None)
            competitor_mcss = (
                local_mcss if local_mcss is not None else all_mcss
            )
            for other in competitor_mcss:
                if (
                    other is mcs
                    or other.is_broken
                    or other.is_recharging
                    or not other.is_idle
                ):
                    continue
                distance_km = self._distance_km(other, quasi)
                if distance_km > COMM_RANGE:
                    continue
                competitor_access += (
                    self._mcs_service_capacity(other)
                    * self._distance_kernel(distance_km)
                )

            local_fcss = getattr(quasi, 'near_available_fcs', None)
            competitor_fcss = (
                local_fcss if local_fcss is not None else all_fcss
            )
            for fcs in competitor_fcss:
                if not fcs.has_available_slot():
                    continue
                distance_km = self._distance_km(fcs, quasi)
                if distance_km > COMM_RANGE:
                    continue
                slot_ratio = (
                    float(fcs.available_slots) / max(float(fcs.capacity), 1.0)
                )
                competitor_access += (
                    FCS_COMPETITION_WEIGHT
                    * slot_ratio
                    * self._distance_kernel(distance_km)
                )

            attraction_raw += total_demand * own_access
            competition_raw += total_demand * competitor_access
            # 分母中的 1 表示未建模的外部服务机会，并确保距离很远时
            # own_access 很小，即使没有竞争者也不会获得高潜力。
            potential_raw += (
                total_demand
                * own_access
                / (1.0 + own_access + competitor_access)
            )

        # 饱和映射保留稀疏区域的分辨率，同时把特征限制在 [0, 1)。
        attraction = 1.0 - np.exp(-attraction_raw)
        competition = 1.0 - np.exp(-competition_raw)
        potential = 1.0 - np.exp(-potential_raw)
        quasi_demand = 1.0 - np.exp(-quasi_demand_raw)
        immediate_iev_demand = 1.0 - np.exp(-immediate_iev_demand_raw)

        return {
            'quasi_demand': float(np.clip(quasi_demand, 0.0, 1.0)),
            'immediate_iev_demand': float(np.clip(
                immediate_iev_demand, 0.0, 1.0
            )),
            'attraction': float(np.clip(attraction, 0.0, 1.0)),
            'competition': float(np.clip(competition, 0.0, 1.0)),
            'potential': float(np.clip(potential, 0.0, 1.0)),
        }

    def compute_best_available_serve_potential(
        self,
        mcs: MCS,
        all_mcss: List[MCS],
        all_fcss: List[FCS],
    ) -> float:
        """返回 Low Actor 当前合法 Serve 候选中的最佳无量纲潜力。

        这里只检查观测中实际可选的最近 ``TOP_K`` 个 quasi 候选，避免
        Wait 惩罚引用 Actor 看不到的全局需求。潜力沿用选位塑形中的
        需求、距离和竞争定义，而不是使用 MCS 当前所在地的吸引力。
        """
        candidates = [
            ev for ev in getattr(mcs, 'near_quasi', [])
            if ev.is_quasi and self._distance_km(mcs, ev) <= COMM_RANGE
        ]
        candidates.sort(key=lambda ev: self._distance_km(mcs, ev))
        candidates = candidates[:TOP_K_MCS_CANDIDATES]

        own_capacity = self._mcs_service_capacity(mcs)
        best_potential = 0.0
        for quasi in candidates:
            demand = self._quasi_demand(quasi)['total']
            if demand <= 0.0:
                continue

            own_access = (
                own_capacity
                * self._distance_kernel(self._distance_km(mcs, quasi))
            )
            competitor_access = 0.0
            for other in getattr(quasi, 'near_idle_mcs', all_mcss):
                if (
                    other is mcs
                    or other.is_broken
                    or other.is_recharging
                    or not other.is_idle
                ):
                    continue
                distance_km = self._distance_km(other, quasi)
                if distance_km <= COMM_RANGE:
                    competitor_access += (
                        self._mcs_service_capacity(other)
                        * self._distance_kernel(distance_km)
                    )

            for fcs in getattr(quasi, 'near_available_fcs', all_fcss):
                if not fcs.has_available_slot():
                    continue
                distance_km = self._distance_km(fcs, quasi)
                if distance_km <= COMM_RANGE:
                    competitor_access += (
                        FCS_COMPETITION_WEIGHT
                        * float(fcs.available_slots)
                        / max(float(fcs.capacity), 1.0)
                        * self._distance_kernel(distance_km)
                    )

            potential_raw = (
                demand * own_access
                / (1.0 + own_access + competitor_access)
            )
            candidate_potential = 1.0 - np.exp(-potential_raw)
            best_potential = max(best_potential, float(candidate_potential))

        return float(np.clip(best_potential, 0.0, 1.0))

    def compute_failure_responsibility_weights(
        self,
        ev: EV,
        all_mcss: List[MCS],
    ) -> Dict[int, float]:
        """计算一次 EV 失败事件的可控 MCS 责任权重。

        责任集合只包含失败时点能够立即进入匹配的 idle MCS。现有任务、
        补电和 broken 状态在该时点都不可被匹配，因此不处罚。候选还须
        满足与 ImmediateMatcher 一致的通信距离、EV/MCS 能量硬约束，
        并能在一个环境 step 内到达中点。原始权重为
        ``可用性 × 距离衰减 × 服务余量``，最后按事件归一化到权重和 1。
        """
        raw_weights: Dict[int, float] = {}
        for mcs in all_mcss:
            if mcs.is_broken or mcs.is_recharging or not mcs.is_idle:
                continue

            distance_m = float(euclidean_distance(
                ev.pos[0], ev.pos[1], mcs.pos[0], mcs.pos[1]
            ))
            distance_km = distance_m / 1000.0
            if distance_km > COMM_RANGE:
                continue

            # MCS 与 EV 前往中点；显式检查一个 step 内的时间可达性。
            rendezvous_time_min = (
                distance_m / 2.0 / max(float(MOVE_SPEED), EPSILON) / 60.0
            )
            max_step_time_min = (
                float(MAX_MOVE_PER_STEP)
                / max(float(MOVE_SPEED), EPSILON)
                / 60.0
            )
            if rendezvous_time_min > max_step_time_min + EPSILON:
                continue

            charge_pos = [
                (float(ev.pos[0]) + float(mcs.pos[0])) / 2.0,
                (float(ev.pos[1]) + float(mcs.pos[1])) / 2.0,
            ]
            if getattr(ev, 'track', None):
                if ev.track_index < len(ev.track) - 1:
                    next_track_pos = ev.track[ev.track_index + 1]
                else:
                    next_track_pos = ev.track[-1]
            else:
                next_track_pos = getattr(ev, 'destination', ev.pos) or ev.pos

            detour_dist_km = max(float(
                (
                    euclidean_distance(
                        ev.pos[0], ev.pos[1], charge_pos[0], charge_pos[1]
                    )
                    + euclidean_distance(
                        charge_pos[0], charge_pos[1],
                        next_track_pos[0], next_track_pos[1]
                    )
                    - euclidean_distance(
                        ev.pos[0], ev.pos[1],
                        next_track_pos[0], next_track_pos[1]
                    )
                ) / 1000.0
            ), 0.0)
            rendezvous_energy = distance_km / 2.0 * float(POWER_UNIT)
            required_mcs_energy = (
                max(float(ev.need_power), 0.0)
                + detour_dist_km * float(POWER_UNIT)
                + rendezvous_energy
            )
            if float(mcs.remain) + EPSILON < required_mcs_energy:
                continue
            if float(ev.remain) + EPSILON < rendezvous_energy:
                continue

            energy_margin = max(float(mcs.remain) - required_mcs_energy, 0.0)
            service_ability = 0.5 + 0.5 * self._energy_ratio(
                energy_margin, ENERGY_REFERENCE_KWH
            )
            raw_weights[mcs.id] = (
                self._distance_kernel(distance_km) * service_ability
            )

        total_weight = float(sum(raw_weights.values()))
        if total_weight <= EPSILON:
            return {}
        return {
            mcs_id: float(weight / total_weight)
            for mcs_id, weight in raw_weights.items()
        }

    def compute_mcs_reward(
        self,
        mcs: MCS,
        event: Dict,
        MCSState=None,
    ) -> Dict[str, float]:
        """返回模式独立、层级独立的 MCS 奖励分量。

        ``MCSState`` 为旧接口保留，目前不参与计算。返回的 ``total``
        与当前共享 Critic/rollout 接口兼容；其余 high/low 与模式字段为
        后续拆分 Critic 预留。
        """
        del MCSState

        requested_mode = str(event.get('requested_mode', ''))
        is_serve = requested_mode == 'Serve'
        is_recharge = requested_mode in ('Recharge', 'RechargeProgress')
        is_wait = requested_mode == 'Wait'

        previous_potential = float(np.clip(
            event.get('previous_spatial_potential', 0.0), 0.0, 1.0
        ))
        post_potential = float(np.clip(
            event.get('post_spatial_potential', 0.0), 0.0, 1.0
        ))
        post_attraction = float(np.clip(
            event.get('post_attraction', 0.0), 0.0, 1.0
        ))

        movement_energy = max(
            float(event.get('movement_energy_kwh', 0.0)), 0.0
        )
        max_move_energy = max(
            (float(MAX_MOVE_PER_STEP) / 1000.0) * float(POWER_UNIT),
            EPSILON,
        )
        movement_ratio = float(np.clip(
            movement_energy / max_move_energy, 0.0, 1.0
        ))

        matched_service = bool(event.get('became_task', False))

        # -----------------------------------------------------------------
        # Serve：Low Actor 的选位 step 奖励
        # -----------------------------------------------------------------
        position_gain = 0.0
        movement_cost = 0.0
        if is_serve:
            # 匹配成功时，匹配阶段会移除/改变需求对象，导致 post potential
            # 下降。该情况下用独立 event 奖励评价选位，不再用位置差分
            # 重复评价，避免“成功匹配反而得到负 step 奖励”。
            if not matched_service:
                potential_delta = post_potential - previous_potential
                normalized_position_gain = float(np.clip(
                    potential_delta / POSITION_DELTA_REFERENCE,
                    -1.0,
                    1.0,
                ))
                position_gain = (
                    POSITION_SHAPING_WEIGHT * normalized_position_gain
                )
            movement_cost = -SERVE_MOVE_PENALTY * movement_ratio

        serve_step = float(np.clip(
            position_gain + movement_cost, -1.0, 1.0
        ))

        # -----------------------------------------------------------------
        # Serve：匹配成功事件奖励
        # -----------------------------------------------------------------
        service_success = 0.0
        service_energy = 0.0
        matched_service_reward = 0.0
        if matched_service:
            # 匹配成功后 mcs.current_target 为 IEV，charge_power_kwh 是通过
            # 物理约束检查后的计划充电量。根据当前仿真假设，该任务必然
            # 完成，因此在 is_charged=True 的匹配时刻立即发放事件奖励。
            planned_service_kwh = max(
                float(getattr(mcs, 'charge_power_kwh', 0.0)), 0.0
            )
            service_ratio = self._saturating_energy_ratio(
                planned_service_kwh, ENERGY_REFERENCE_KWH
            )
            service_success = SERVICE_SUCCESS_WEIGHT
            service_energy = (
                SERVICE_ENERGY_WEIGHT * float(np.sqrt(service_ratio))
            )
            matched_service_reward = service_success + service_energy

        # 即时匹配会在动作执行后对所有空闲 MCS 运行。因此选择 Wait 的
        # MCS 也可能原地被 IEV 匹配。业务事件应归属于真实的 High 模式；
        # 只有 Serve 匹配事件同时属于 Low Actor 的目标选择结果。
        if matched_service and is_wait:
            serve_event = 0.0
            matched_wait_event = matched_service_reward
        else:
            serve_event = matched_service_reward
            matched_wait_event = 0.0

        # -----------------------------------------------------------------
        # Recharge：High Actor 独立的 step/event 奖励
        # -----------------------------------------------------------------
        previous_remain = float(
            event.get('previous_remain_kwh', mcs.remain)
        )
        current_remain = float(
            event.get('current_remain_kwh', mcs.remain)
        )
        previous_risk = self._battery_risk(previous_remain)
        current_risk = self._battery_risk(current_remain)

        recharge_step = 0.0
        recharge_match = 0.0
        recharge_match_failure = 0.0
        if is_recharge:
            recharge_step = (
                RECHARGE_STEP_WEIGHT
                * max(previous_risk - current_risk, 0.0)
            )
            if event.get('recharge_matched', False):
                # 只在确实存在补电紧迫度时奖励匹配，防止高电量反复补电。
                recharge_match = RECHARGE_MATCH_WEIGHT * previous_risk
            elif requested_mode == 'Recharge':
                recharge_match_failure = -RECHARGE_MATCH_FAILURE_PENALTY

        recharge_event = float(recharge_match + recharge_match_failure)

        # -----------------------------------------------------------------
        # Wait：High Actor 独立的三段式 step 惩罚
        # -----------------------------------------------------------------
        forced_wait = bool(event.get('forced_wait', False)) and is_wait
        voluntary_wait = bool(event.get('voluntary_wait', False)) and is_wait
        wait_duration_steps = max(
            int(event.get('wait_duration_steps', 1 if is_wait else 0)), 0
        )
        best_serve_potential = float(np.clip(
            event.get('best_available_serve_potential', 0.0), 0.0, 1.0
        ))
        voluntary_wait_streak = max(
            int(event.get('consecutive_voluntary_wait_steps', 0)), 0
        )

        wait_base = 0.0
        wait_opportunity = 0.0
        wait_streak = 0.0
        wait_forced_time = 0.0
        if forced_wait:
            wait_forced_time = (
                -FORCED_WAIT_TIME_PENALTY * wait_duration_steps
            )
        elif voluntary_wait:
            duration_scale = max(wait_duration_steps, 1)
            wait_base = -WAIT_BASE_PENALTY * duration_scale
            wait_opportunity = (
                -WAIT_OPPORTUNITY_PENALTY
                * best_serve_potential
                * duration_scale
            )
            wait_streak = (
                -WAIT_STREAK_PENALTY
                * min(
                    voluntary_wait_streak
                    / max(float(WAIT_STREAK_NORMALIZER), 1.0),
                    1.0,
                )
                * duration_scale
            )
        wait_step = float(
            wait_base + wait_opportunity + wait_streak + wait_forced_time
        )
        wait_event = float(matched_wait_event)

        # -----------------------------------------------------------------
        # 责任加权的系统业务事件
        # -----------------------------------------------------------------
        # MCS 成功只奖励实际 provider；可控失败按 World 计算的归一化
        # 责任权重处罚。FCS 成功和外生失败的学习奖励恒为 0。
        attributed_success_count = max(
            int(event.get('attributed_mcs_success_count', 0)), 0
        )
        attributed_success_weight = max(float(
            event.get(
                'attributed_mcs_success_weight', attributed_success_count
            )
        ), 0.0)
        controllable_failure_weight = max(float(
            event.get('controllable_failure_weight', 0.0)
        ), 0.0)
        controllable_failure_count = max(
            int(event.get('controllable_failure_count', 0)), 0
        )
        uncontrollable_failure_count = max(
            int(event.get('uncontrollable_failure_count', 0)), 0
        )
        fcs_success_kpi_count = max(
            int(event.get('fcs_success_kpi_count', 0)), 0
        )

        attributed_mcs_success = (
            SYSTEM_SUCCESS_EVENT_REWARD
            * attributed_success_weight
        )
        controllable_failure = (
            -SYSTEM_FAILURE_EVENT_PENALTY
            * controllable_failure_weight
        )
        # 两个 KPI 分量明确保留为 0，防止误接入 High/Low 任一回报。
        uncontrollable_failure = 0.0
        fcs_success_kpi = 0.0
        attributable_system_event = float(
            attributed_mcs_success + controllable_failure
        )

        # broken 为一次性安全事件，不混入任何普通 step 奖励。
        broken_event = (
            -BROKEN_EVENT_PENALTY
            if event.get('newly_broken', False)
            else 0.0
        )

        # 当前每个 step 只会有一个决策模式生效，因此模式分量相加不会
        # 重复计奖。broken 是跨模式的一次性公共安全事件。
        step_total = float(serve_step + recharge_step + wait_step)
        # 继续关闭旧 MCS 服务匹配事件对实际回报的贡献。
        # 保留 serve_event、wait_event 和 matched_service_reward 的完整
        # 计算及日志输出；责任系统事件取代它成为主导信号。
        # event_total = float(
        #     serve_event + recharge_event + wait_event + broken_event
        # )
        event_total = float(
            attributable_system_event + recharge_event + broken_event
        )
        total = float(
            STEP_REWARD_WEIGHT * step_total
            + EVENT_REWARD_WEIGHT * event_total
        )

        # High/Low 是“奖励视图”，不是额外奖励，不能再次加到 total 中。
        high_step = step_total
        high_event = event_total
        low_step = serve_step if is_serve else 0.0
        # Low 只在 Serve 模式接收可归因系统事件；FCS/外生失败均为 0。
        # 旧服务匹配奖励仍保持关闭。
        # low_event = serve_event + broken_event if is_serve else 0.0
        low_event = (
            attributable_system_event + broken_event if is_serve else 0.0
        )

        components = {
            # 各模式独立的 step/event 奖励。
            'serve_step': serve_step,
            'serve_event': serve_event,
            'recharge_step': float(recharge_step),
            'recharge_event': recharge_event,
            'wait_step': float(wait_step),
            'wait_event': wait_event,
            'broken_event': float(broken_event),
            # 为后续拆分 High/Low Critic 预留的独立视图。
            'high_step': high_step,
            'high_event': high_event,
            'low_step': float(low_step),
            'low_event': float(low_event),
            'step_total': step_total,
            'event_total': event_total,
            # 可审计的原子分量。
            'position_gain': float(position_gain),
            'service_success': float(service_success),
            'service_energy': float(service_energy),
            'recharge_match': float(recharge_match),
            'attributed_mcs_success': float(attributed_mcs_success),
            'controllable_failure': float(controllable_failure),
            'uncontrollable_failure': float(uncontrollable_failure),
            'fcs_success_kpi': float(fcs_success_kpi),
            'attributed_mcs_success_weight': float(
                attributed_success_weight
            ),
            'controllable_failure_weight': float(
                controllable_failure_weight
            ),
            'attributed_mcs_success_count': float(
                attributed_success_count
            ),
            'controllable_failure_count': float(
                controllable_failure_count
            ),
            'uncontrollable_failure_count': float(
                uncontrollable_failure_count
            ),
            'fcs_success_kpi_count': float(fcs_success_kpi_count),
            # 兼容旧日志字段，但语义已经改为“可归因/可控”。
            'system_success': float(attributed_mcs_success),
            'system_failure': float(controllable_failure),
            'system_team_event': attributable_system_event,
            'wait_base': float(wait_base),
            'wait_opportunity': float(wait_opportunity),
            'wait_streak': float(wait_streak),
            'wait_forced_time': float(wait_forced_time),
            'forced_wait_count': float(forced_wait),
            'voluntary_wait_count': float(voluntary_wait),
            'best_available_serve_potential': best_serve_potential,
            # 与当前 train.py 日志字段保持兼容的别名。
            'service': float(matched_service_reward),
            'serve_attraction': (
                post_attraction if is_serve and not matched_service else 0.0
            ),
            'serve_competition': (
                -float(event.get('post_competition', 0.0))
                if is_serve and not matched_service else 0.0
            ),
            'serve_potential_improvement': float(position_gain),
            'recharge': float(recharge_step + recharge_match),
            'movement': float(movement_cost),
            'wait': float(wait_step),
            'recharge_match_failure': float(recharge_match_failure),
            'broken': float(broken_event),
            'battery_potential': float(recharge_step),
            'total': total,
        }
        return {name: float(value) for name, value in components.items()}

    def compute_iev_reward(self, ev: EV, info: Dict) -> float:
        """当前仅训练 MCS，IEV 奖励保持为 0 以兼容 World 接口。"""
        del ev, info
        return 0.0
