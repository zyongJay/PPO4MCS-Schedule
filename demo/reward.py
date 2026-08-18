"""
MCS 奖励计算器。

当前奖励分为两个互相独立的层级：

1. step 级奖励：评价本次动作带来的即时状态改善。
   - High Serve：只评价是否存在合法 Serve 候选，不读取具体吸引力/竞争力。
   - High Recharge：只评价低电量风险是否下降。
   - Low 当前位置固定候选承载主动/被动等待语义，High 不再拥有 Wait。
   - Low Serve：独立评价选位后的需求覆盖、竞争情况和移动能耗。
2. event 级奖励：评价离散业务事件。
   - MCS 成功事件只奖励实际服务该 EV 的责任 MCS。
   - 可控失败事件仅进入 Low 的空间责任回报，不进入 High 回报。
   - FCS 成功和无可行 MCS 的外生失败只记录 KPI，不进入 MCS 回报。
   - MCS 与 IEV 的服务匹配奖励只保留计算和日志，不进入实际回报。
   - MCS 首次 broken、首次 energy-stranded 时给出 High 独占惩罚。

各模式的 step/event 分量分别返回，并显式形成 ``high_total`` 与
``low_total``。训练只允许 High/Low 各自消费对应回报；``total`` 仅作为
旧日志和环境 reward_n 的兼容字段。

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
from low_spatial import (
    compute_low_candidate_metrics,
    quasi_transition_urgency,
)


# =============================================================================
# 奖励参数
# =============================================================================

# High/Low 各自拥有独立权重；训练不得再使用兼容字段 total 混合更新。
HIGH_STEP_REWARD_WEIGHT = 0.20
HIGH_EVENT_REWARD_WEIGHT = 0.80
LOW_STEP_REWARD_WEIGHT = 0.25
LOW_EVENT_REWARD_WEIGHT = 0.75
# 兼容旧日志/外部分析脚本。
STEP_REWARD_WEIGHT = HIGH_STEP_REWARD_WEIGHT
EVENT_REWARD_WEIGHT = HIGH_EVENT_REWARD_WEIGHT

# 空间服务机会与竞争力参数。
ENERGY_REFERENCE_KWH = max(float(MAX_CHARGE_PER_SESSION_KWH), 1e-8)
DISTANCE_DECAY_KM = max(float(COMM_RANGE) / 2.0, 1e-8)
IEV_DEMAND_WEIGHT = 2.0
FCS_COMPETITION_WEIGHT = 1.0
POSITION_DELTA_REFERENCE = 0.25
# 选位塑形只使用一个综合的“计数型边际服务机会改善”，避免 attraction、
# MCS competition、FCS competition 在综合 desirability 内外重复计奖。
LOW_SPATIAL_OPPORTUNITY_WEIGHT = 0.20
# 兼容旧参数名；实际含义已变为综合机会改善权重。
POSITION_SHAPING_WEIGHT = LOW_SPATIAL_OPPORTUNITY_WEIGHT
SERVE_MOVE_PENALTY = 0.08
# High 只判断“当前是否存在合法 Serve 候选”。具体候选的吸引力、竞争力
# 和位置质量全部交给 Low Actor，避免两层重复优化空间目标。
HIGH_SERVE_FEASIBILITY_REWARD = 0.02

# Low 成功事件比 High 更严格地衡量系统边际贡献。High 仍沿用原来的
# FCS 可替代 slot 权重；Low 额外考虑其他可行 MCS，避免通过选位抢占
# 已有资源能够完成的订单。
# FCS 一个可行槽位比另一辆 MCS 更稳定，但不应把仍然增加全局成功数的
# MCS 服务压到接近零，因此采用 2:1 的可替代性系数，而非旧版 4:1。
LOW_FCS_ALTERNATIVE_PENALTY = 2.0
LOW_MCS_ALTERNATIVE_PENALTY = 1.0

# 服务事件由“成功服务一辆 IEV”和“服务电量”共同组成。固定成功奖励
# 占比较高，电量项采用平方根压缩，防止策略只追逐高需求量 IEV。
SERVICE_SUCCESS_WEIGHT = 0.60
SERVICE_ENERGY_WEIGHT = 0.40

# 责任事件的系统总量。成功奖励只给实际服务 MCS；可控失败惩罚按责任
# 权重归一化分配。每个事件在全部 MCS 上的权重和不超过 1。
SYSTEM_SUCCESS_EVENT_REWARD = 1.0
SYSTEM_FAILURE_EVENT_PENALTY = 1.1

# Recharge 只奖励真实发生的电量风险改善，不再对匹配成功/失败重复计奖。
RECHARGE_STEP_WEIGHT = 0.20

# 等待惩罚属于 Low 的固定当前位置动作；High Actor 已不再拥有 Wait。
# 被动等待没有替代 quasi，因此不处罚；主动等待按基础、机会和连续时长
# 三部分处罚，防止 Low 在存在可移动候选时长期原地不动。
FORCED_WAIT_TIME_PENALTY = 0.0
WAIT_BASE_PENALTY = 0.03
WAIT_OPPORTUNITY_PENALTY = 0.07
WAIT_STREAK_PENALTY = 0.05
WAIT_STREAK_NORMALIZER = 10

# broken/energy-stranded 都只在状态首次进入时处罚一次。后者表示空闲 MCS
# 的当前电量已经无法到达任何物理 FCS，不包含“FCS 暂时无空槽”等外因。
BROKEN_EVENT_PENALTY = 2.0
ENERGY_STRANDED_EVENT_PENALTY = BROKEN_EVENT_PENALTY

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
        mcs_competition_raw = 0.0
        fcs_competition_raw = 0.0
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
            competitor_mcs_access = 0.0
            competitor_fcs_access = 0.0
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
                    or getattr(other, 'is_energy_stranded', False)
                    or other.is_recharging
                    or not other.is_idle
                ):
                    continue
                distance_km = self._distance_km(other, quasi)
                if distance_km > COMM_RANGE:
                    continue
                value = (
                    self._mcs_service_capacity(other)
                    * self._distance_kernel(distance_km)
                )
                competitor_access += value
                competitor_mcs_access += value

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
                value = (
                    FCS_COMPETITION_WEIGHT
                    * slot_ratio
                    * self._distance_kernel(distance_km)
                )
                competitor_access += value
                competitor_fcs_access += value

            attraction_raw += total_demand * own_access
            competition_raw += total_demand * competitor_access
            mcs_competition_raw += total_demand * competitor_mcs_access
            fcs_competition_raw += total_demand * competitor_fcs_access
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
        mcs_competition = 1.0 - np.exp(-mcs_competition_raw)
        fcs_competition = 1.0 - np.exp(-fcs_competition_raw)
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
            'mcs_competition': float(np.clip(
                mcs_competition, 0.0, 1.0
            )),
            'fcs_competition': float(np.clip(
                fcs_competition, 0.0, 1.0
            )),
            'potential': float(np.clip(potential, 0.0, 1.0)),
        }

    def compute_low_spatial_features(
        self,
        mcs: MCS,
        all_evs: List[EV],
        all_fcss: List[FCS] | None = None,
    ) -> Dict[str, float]:
        """使用与 Low candidate observation 相同的公式评价当前位置。

        只检查 MCS 通信范围、返程能量安全且按紧急度进入 Top-K 的 quasi。
        每个 quasi 的吸引/竞争均来自
        ``low_spatial.compute_low_candidate_metrics``，因此 Actor 所见特征
        与 Low step reward 的语义和数值尺度保持一致。当前位置取局部
        desirability 最高的候选，表示 MCS 在该位置能够获得的最佳局部
        调度机会，而不是读取任何远处全图需求。
        """
        best = {
            'attraction': 0.0,
            'immediate_iev_attraction': 0.0,
            'mcs_competition': 0.0,
            'fcs_competition': 0.0,
            'desirability': 0.0,
        }
        candidates = [
            quasi for quasi in all_evs
            if quasi.is_quasi
            and self._distance_km(mcs, quasi) <= COMM_RANGE
        ]
        if all_fcss is not None:
            if not all_fcss:
                candidates = []
            else:
                candidates = [
                    quasi for quasi in candidates
                    if float(mcs.remain) > (
                        self._distance_km(mcs, quasi)
                        + min(
                            self._distance_km(quasi, fcs)
                            for fcs in all_fcss
                        )
                    ) * float(POWER_UNIT)
                ]
        candidates.sort(key=lambda quasi: (
            -quasi_transition_urgency(quasi),
            self._distance_km(mcs, quasi),
            int(quasi.id),
        ))
        candidates = candidates[:max(TOP_K_MCS_CANDIDATES - 1, 0)]

        for quasi in candidates:
            metrics = compute_low_candidate_metrics(mcs, quasi)
            if metrics['desirability'] > best['desirability']:
                best = {
                    'attraction': metrics['attraction'],
                    'immediate_iev_attraction': metrics[
                        'immediate_iev_attraction'
                    ],
                    'mcs_competition': metrics['mcs_competition'],
                    'fcs_competition': metrics['fcs_competition'],
                    'desirability': metrics['desirability'],
                }
        return {name: float(value) for name, value in best.items()}

    @staticmethod
    def compute_low_success_marginal_weight(
        feasible_fcs_slot_count: int,
        feasible_mcs_count: int,
    ) -> float:
        """返回 Low 成功事件的系统边际贡献代理。

        ``feasible_mcs_count`` 包含最终 provider 本身，因此只把其余 MCS
        视为替代资源。无 FCS 且无其他 MCS 时权重为 1；每增加一个可行
        FCS slot 或其他 MCS 都会降低 Low 的成功收益。该函数不改变 High
        原有的成功权重。
        """
        fcs_count = max(int(feasible_fcs_slot_count), 0)
        other_mcs_count = max(int(feasible_mcs_count) - 1, 0)
        denominator = (
            1.0
            + LOW_FCS_ALTERNATIVE_PENALTY * fcs_count
            + LOW_MCS_ALTERNATIVE_PENALTY * other_mcs_count
        )
        return float(np.clip(1.0 / denominator, 0.0, 1.0))

    def compute_best_available_serve_potential(
        self,
        mcs: MCS,
        all_mcss: List[MCS],
        all_fcss: List[FCS],
    ) -> float:
        """返回 Low Actor 当前合法 Serve 候选中的最佳无量纲潜力。

        这里只检查观测中实际可选、补电紧急度最高的 ``TOP_K-1`` 个
        quasi 候选，避免
        Wait 惩罚引用 Actor 看不到的全局需求。潜力沿用选位塑形中的
        需求、距离和竞争定义，而不是使用 MCS 当前所在地的吸引力。
        """
        del all_mcss  # 签名保留接口兼容，竞争已由候选局部特征统一计算。
        candidates = []
        if all_fcss:
            for ev in getattr(mcs, 'near_quasi', []):
                if not ev.is_quasi or self._distance_km(mcs, ev) > COMM_RANGE:
                    continue
                route_distance_km = self._distance_km(mcs, ev) + min(
                    self._distance_km(ev, fcs) for fcs in all_fcss
                )
                if float(mcs.remain) > route_distance_km * float(POWER_UNIT):
                    candidates.append(ev)
        candidates.sort(key=lambda ev: (
            -quasi_transition_urgency(ev),
            self._distance_km(mcs, ev),
            int(ev.id),
        ))
        candidates = candidates[:max(TOP_K_MCS_CANDIDATES - 1, 0)]

        best_potential = 0.0
        for quasi in candidates:
            candidate_potential = compute_low_candidate_metrics(
                mcs, quasi
            )['desirability']
            best_potential = max(
                best_potential, float(candidate_potential)
            )

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
            if (
                mcs.is_broken
                or getattr(mcs, 'is_energy_stranded', False)
                or mcs.is_recharging
                or not mcs.is_idle
            ):
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

        ``MCSState`` 为旧接口保留，目前不参与计算。High/Low 训练分别
        使用 ``high_total``/``low_total``；``total`` 仅兼容旧日志/API。
        """
        del MCSState

        requested_mode = str(event.get('requested_mode', ''))
        is_serve = requested_mode == 'Serve'
        is_recharge = requested_mode in ('Recharge', 'RechargeProgress')
        is_wait = requested_mode == 'Wait'
        low_stay_selected = bool(event.get('low_stay_selected', False))
        low_forced_stay = bool(event.get('low_forced_stay', False))

        previous_potential = float(np.clip(
            event.get(
                'previous_low_spatial_desirability',
                event.get('previous_spatial_potential', 0.0),
            ), 0.0, 1.0
        ))
        post_potential = float(np.clip(
            event.get(
                'post_low_spatial_desirability',
                event.get('post_spatial_potential', 0.0),
            ), 0.0, 1.0
        ))
        post_attraction = float(np.clip(
            event.get(
                'post_low_attraction', event.get('post_attraction', 0.0)
            ), 0.0, 1.0
        ))
        post_immediate_demand = float(np.clip(
            event.get(
                'post_low_immediate_iev_attraction',
                event.get('post_immediate_iev_demand', 0.0),
            ), 0.0, 1.0
        ))
        post_mcs_competition = float(np.clip(
            event.get(
                'post_low_mcs_competition',
                event.get('post_mcs_competition', 0.0),
            ), 0.0, 1.0
        ))
        post_fcs_competition = float(np.clip(
            event.get(
                'post_low_fcs_competition',
                event.get('post_fcs_competition', 0.0),
            ), 0.0, 1.0
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

        resource_gap_improvement = float(position_gain)
        # 旧三项保留为零值兼容字段。吸引力与两类竞争已在 desirability
        # 中组合，再单独加减会重复计奖并放大局部塑形。
        attraction_reward = 0.0
        mcs_cluster_penalty = 0.0
        fcs_redundancy_penalty = 0.0
        serve_step = float(np.clip(
            resource_gap_improvement
            + movement_cost,
            -1.0,
            1.0,
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
        # Recharge：High Actor 只保留实际风险改善 step 奖励
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
        # 以下变量仅保留旧日志兼容；不再进入任何 High 奖励。
        recharge_event = 0.0

        # -----------------------------------------------------------------
        # Wait：Low 固定当前位置动作的三段式 step 惩罚
        # -----------------------------------------------------------------
        # 旧 requested_mode=Wait 仅用于兼容旧调用方。新训练流程中，
        # Low 的 stay 候选决定主动/被动等待，惩罚不会进入 High reward。
        forced_wait = bool(event.get('forced_wait', False)) and (
            low_stay_selected or is_wait
        )
        voluntary_wait = bool(event.get('voluntary_wait', False)) and (
            low_stay_selected or is_wait
        )
        wait_duration_steps = max(
            int(event.get('wait_duration_steps', 1 if is_wait else 0)), 0
        )
        best_serve_potential = float(np.clip(
            event.get('best_available_serve_potential', 0.0), 0.0, 1.0
        ))
        serve_available = bool(event.get('serve_available', False))
        has_quasi_candidate = bool(event.get('has_quasi_candidate', False))
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
            # Low 只需知道存在其他 quasi，不直接把具体吸引力/竞争力
            # 重复塞进等待惩罚；具体空间质量仍由 Serve 塑形负责。
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
        low_attributed_success_weight = max(float(
            event.get('low_attributed_mcs_success_weight', 0.0)
        ), 0.0)
        controllable_failure_weight = max(float(
            event.get('controllable_failure_weight', 0.0)
        ), 0.0)
        low_controllable_failure_weight = max(float(
            event.get('low_controllable_failure_weight', 0.0)
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
        low_attributed_mcs_success = (
            SYSTEM_SUCCESS_EVENT_REWARD * low_attributed_success_weight
        )
        low_controllable_failure = (
            -SYSTEM_FAILURE_EVENT_PENALTY
            * low_controllable_failure_weight
        )

        # broken 为一次性安全事件，不混入任何普通 step 奖励。
        broken_event = (
            -BROKEN_EVENT_PENALTY
            if event.get('newly_broken', False)
            else 0.0
        )

        energy_stranded_event = (
            -ENERGY_STRANDED_EVENT_PENALTY
            if event.get('newly_energy_stranded', False)
            else 0.0
        )

        # High 只评价模式选择，不接收具体候选的空间塑形和移动成本。
        high_serve_suitability = (
            HIGH_SERVE_FEASIBILITY_REWARD
            if is_serve and has_quasi_candidate else 0.0
        )
        high_step = float(
            high_serve_suitability + recharge_step
        )
        # High event 只保留其职责直接对应的三类结果：实际 MCS 成功、
        # broken 和 energy-stranded。EV 可控失败继续保留给 Low 的
        # 空间责任链，但不再进入 High 回报。
        high_event = float(
            attributed_mcs_success + broken_event + energy_stranded_event
        )
        # Low step 只属于当前 Serve 决策；TaskProgress 可携带延迟事件，
        # 但不能再次产生选位塑形。
        if is_serve and low_stay_selected:
            # 原地动作不获取位置吸引力/竞争力塑形，避免通过停留刷取
            # 正 step 奖励。被动等待的 wait_step 为 0，主动等待为负。
            low_step = float(wait_step)
        else:
            low_step = float(serve_step if is_serve else 0.0)
        has_low_responsibility = int(event.get('low_decision_id', -1)) >= 0
        low_event = float(
            low_attributed_mcs_success + low_controllable_failure
            if has_low_responsibility else 0.0
        )

        high_total = float(
            HIGH_STEP_REWARD_WEIGHT * high_step
            + HIGH_EVENT_REWARD_WEIGHT * high_event
        )
        low_total = float(
            LOW_STEP_REWARD_WEIGHT * low_step
            + LOW_EVENT_REWARD_WEIGHT * low_event
        )

        # 以下三项仅用于兼容旧日志/API，不得作为双层训练回报。
        step_total = float(high_step + low_step)
        # 继续关闭旧 MCS 服务匹配事件对实际回报的贡献。
        # 保留 serve_event、wait_event 和 matched_service_reward 的完整
        # 计算及日志输出；责任系统事件取代它成为主导信号。
        # event_total = float(
        #     serve_event + recharge_event + wait_event + broken_event
        # )
        event_total = float(high_event + low_event)
        total = float(high_total + low_total)

        components = {
            # 各模式独立的 step/event 奖励。
            'serve_step': serve_step,
            'serve_event': serve_event,
            'recharge_step': float(recharge_step),
            'recharge_event': recharge_event,
            'wait_step': float(wait_step),
            'wait_event': wait_event,
            'broken_event': float(broken_event),
            'energy_stranded_event': float(energy_stranded_event),
            # 为后续拆分 High/Low Critic 预留的独立视图。
            'high_step': high_step,
            'high_event': high_event,
            'low_step': float(low_step),
            'low_event': float(low_event),
            'high_total': high_total,
            'low_total': low_total,
            'low_step_weighted': float(
                LOW_STEP_REWARD_WEIGHT * low_step
            ),
            'low_event_weighted': float(
                LOW_EVENT_REWARD_WEIGHT * low_event
            ),
            'step_total': step_total,
            'event_total': event_total,
            # 可审计的原子分量。
            'position_gain': float(position_gain),
            'low_attraction_reward': float(attraction_reward),
            'low_immediate_iev_attraction': float(post_immediate_demand),
            'service_success': float(service_success),
            'service_energy': float(service_energy),
            'recharge_match': float(recharge_match),
            'attributed_mcs_success': float(attributed_mcs_success),
            'low_attributed_mcs_success': float(
                low_attributed_mcs_success
            ),
            'controllable_failure': float(controllable_failure),
            'low_controllable_failure': float(
                low_controllable_failure
            ),
            'uncontrollable_failure': float(uncontrollable_failure),
            'fcs_success_kpi': float(fcs_success_kpi),
            'attributed_mcs_success_weight': float(
                attributed_success_weight
            ),
            'controllable_failure_weight': float(
                controllable_failure_weight
            ),
            'low_attributed_mcs_success_weight': float(
                low_attributed_success_weight
            ),
            'low_controllable_failure_weight': float(
                low_controllable_failure_weight
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
            'low_active_wait_count': float(
                low_stay_selected and not low_forced_stay
            ),
            'low_passive_wait_count': float(
                low_stay_selected and low_forced_stay
            ),
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
            'resource_gap_improvement': resource_gap_improvement,
            'low_spatial_opportunity_gain': resource_gap_improvement,
            'low_attraction_metric': post_attraction,
            'low_mcs_competition_metric': post_mcs_competition,
            'low_fcs_competition_metric': post_fcs_competition,
            # 兼容旧日志列；当前语义改为候选局部聚合吸引奖励。
            'urgency_coverage': float(attraction_reward),
            'mcs_cluster_penalty': float(mcs_cluster_penalty),
            'fcs_redundancy_penalty': float(fcs_redundancy_penalty),
            'high_serve_suitability': float(high_serve_suitability),
            'recharge': float(recharge_step + recharge_match),
            'movement': float(movement_cost),
            'wait': float(wait_step),
            'recharge_match_failure': float(recharge_match_failure),
            'broken': float(broken_event),
            'energy_stranded': float(energy_stranded_event),
            'battery_potential': float(recharge_step),
            'total': total,
        }
        return {name: float(value) for name, value in components.items()}

    def compute_iev_reward(self, ev: EV, info: Dict) -> float:
        """当前仅训练 MCS，IEV 奖励保持为 0 以兼容 World 接口。"""
        del ev, info
        return 0.0
