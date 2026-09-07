"""Low Actor 选位使用的局部候选特征。

Low 决策发生在即时 IEV 匹配之后。参与决策的 idle MCS 只直接选择
通信范围内的 quasi，但候选 quasi 可以提供其局部邻域的聚合信息。
这里不暴露远处全图状态，也不暴露具体 IEV 身份，只构造候选点的：

1. 自身计数型服务机会与失败紧迫度；
2. MCS 到候选的移动距离；
3. 候选附近需求吸引力；
4. 候选附近 MCS 资源竞争；
5. 候选附近 FCS 资源竞争。

Observation 与 Low reward 共用本模块，避免两侧使用不同公式和尺度。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from config import (
    COMM_RANGE,
    EV_LOW_POWER_THRESHOLD,
    MAX_CHARGE_PER_SESSION_KWH,
    MAX_WAIT_TIME_STEPS,
)
from core import EV, MCS, euclidean_distance


EPSILON = 1e-8
DISTANCE_DECAY_KM = max(float(COMM_RANGE) / 2.0, EPSILON)

# 所有吸引力均以“一辆 EV 是否可能形成一次成功服务”计数，不再乘以
# need_power。这样大电量订单不会仅因 kWh 更高而获得更大选位吸引力。
QUASI_BASE_OPPORTUNITY = 0.50
QUASI_URGENCY_BONUS = 0.50
IEV_BASE_OPPORTUNITY = 1.00
IEV_WAIT_URGENCY_BONUS = 0.50

# 只用于 EV_attraction 聚合。IEV 是已经形成的即时充电需求，必须比
# 仍处于潜在需求阶段的 quasi 获得更高权重；候选排序和单车紧急度仍
# 使用上面的 opportunity，避免重复放大。
QUASI_ATTRACTION_WEIGHT = 1.00
IEV_ATTRACTION_WEIGHT = 1.50

# task MCS 和 busy FCS 当前不能立即参与匹配，但它们仍表示候选区域已有
# 服务资源。权重低于当前可用资源，避免把“繁忙”误判为完全资源富余。
TASK_MCS_COMPETITION_WEIGHT = 0.25
BUSY_FCS_COMPETITION_WEIGHT = 0.10

# desirability 是唯一进入 Low step 的综合空间质量。竞争只削弱有需求的
# 候选，不会让“没有需求但没有竞争”的区域获得正奖励。FCS 以实际可用
# 槽位数计数，因此无需再用过大的手工站点常数。
MCS_COMPETITION_IMPORTANCE = 1.0
FCS_COMPETITION_IMPORTANCE = 1.5

# 计数信号的半饱和参考值：3 个加权 EV 机会、1 辆可替代 MCS、2 个
# FCS 可用槽分别映射到 0.5。相比旧 1-exp(-x)，在多车区域保留更多
# 分辨率，防止 attraction 很快全部接近 1。
ATTRACTION_COUNT_REFERENCE = 3.0
IMMEDIATE_IEV_COUNT_REFERENCE = 2.0
MCS_COMPETITION_REFERENCE = 1.0
FCS_COMPETITION_REFERENCE = 2.0


def distance_km(first, second) -> float:
    """返回两个实体之间的千米距离。"""
    return float(euclidean_distance(
        first.pos[0], first.pos[1], second.pos[0], second.pos[1]
    ) / 1000.0)


def distance_kernel(value_km: float) -> float:
    """将距离映射为 [0, 1] 内的局部影响强度。"""
    return float(np.exp(-max(float(value_km), 0.0) / DISTANCE_DECAY_KM))


def saturating_count(value: float, reference: float) -> float:
    """按明确的半饱和计数参考值映射到 [0, 1)。"""
    value = max(float(value), 0.0)
    reference = max(float(reference), EPSILON)
    return float(value / (value + reference))


def quasi_transition_urgency(quasi: EV) -> float:
    """quasi 距离转变为 IEV 的无量纲紧急程度。

    当剩余电量接近 ``EV_LOW_POWER_THRESHOLD`` 时趋近 1；剩余电量越高
    越接近 0。该指标只用于候选排序和潜在订单紧迫度，不读取充电需求
    kWh，因此不会重新引入偏好大电量订单的收益信号。
    """
    threshold = max(float(EV_LOW_POWER_THRESHOLD), EPSILON)
    remain = max(float(quasi.remain), threshold)
    return float(np.clip(threshold / remain, 0.0, 1.0))


def ev_service_opportunity(ev: EV) -> float:
    """返回与充电电量无关的单车服务机会权重。

    quasi 至少贡献半个潜在订单，并随接近形成需求的程度提升至 1；已经
    形成需求的 IEV 至少贡献 1，并只按等待进度增加失败紧迫度。该值不读取
    ``need_power``，从根源上避免 Low 偏向大电量订单。
    """
    if ev.is_iev:
        wait_urgency = float(np.clip(
            float(getattr(ev, 'wait_time_steps', 0)) /
            max(float(MAX_WAIT_TIME_STEPS), 1.0),
            0.0,
            1.0,
        ))
        return float(
            IEV_BASE_OPPORTUNITY
            + IEV_WAIT_URGENCY_BONUS * wait_urgency
        )
    return float(
        QUASI_BASE_OPPORTUNITY
        + QUASI_URGENCY_BONUS * quasi_transition_urgency(ev)
    )


def quasi_urgency_demand(quasi: EV) -> float:
    """兼容旧调用名；现表示单车计数型服务机会，不含需求电量。"""
    return ev_service_opportunity(quasi)


def _mcs_capacity_ratio(mcs: MCS) -> float:
    """只用于竞争估计的单个 MCS 可替代服务能力。"""
    energy_ratio = float(np.clip(
        min(max(float(mcs.remain), 0.0), MAX_CHARGE_PER_SESSION_KWH)
        / max(float(MAX_CHARGE_PER_SESSION_KWH), EPSILON),
        0.0,
        1.0,
    ))
    # 一个可用 MCS 首先代表一个潜在服务者；电量只修正其可行性，不把
    # 它换算成可获取的收益或订单电量。
    return float(0.25 + 0.75 * energy_ratio)


def compute_low_candidate_metrics(mcs: MCS, quasi: EV) -> Dict[str, float]:
    """计算一个合法 quasi 候选的 Low 局部特征。

    ``near_iev`` 只以聚合吸引力进入特征，不向 Actor 暴露 IEV 的身份、
    位置或全图状态。idle/task MCS 与 available/busy FCS 均计入竞争，
    其中当前不可立即服务的 task/busy 资源采用较小权重。
    """
    candidate_distance_km = distance_km(mcs, quasi)
    urgency_demand = ev_service_opportunity(quasi)

    attraction_raw = QUASI_ATTRACTION_WEIGHT * urgency_demand
    immediate_iev_raw = 0.0
    seen_ev_ids = {int(quasi.id)}
    for other in getattr(quasi, 'near_quasi', []):
        if int(other.id) in seen_ev_ids or not other.is_quasi:
            continue
        seen_ev_ids.add(int(other.id))
        attraction_raw += (
            QUASI_ATTRACTION_WEIGHT
            * ev_service_opportunity(other)
            * distance_kernel(distance_km(quasi, other))
        )
    for iev in getattr(quasi, 'near_iev', []):
        if int(iev.id) in seen_ev_ids or not iev.is_iev:
            continue
        seen_ev_ids.add(int(iev.id))
        value = (
            IEV_ATTRACTION_WEIGHT * ev_service_opportunity(iev)
            * distance_kernel(distance_km(quasi, iev))
        )
        attraction_raw += value
        immediate_iev_raw += value

    mcs_competition_raw = 0.0
    seen_mcs_ids = {int(mcs.id)}
    for other in getattr(quasi, 'near_idle_mcs', []):
        if (
            int(other.id) in seen_mcs_ids
            or other.is_broken
            or getattr(other, 'is_energy_stranded', False)
        ):
            continue
        seen_mcs_ids.add(int(other.id))
        mcs_competition_raw += (
            _mcs_capacity_ratio(other)
            * distance_kernel(distance_km(quasi, other))
        )
    for other in getattr(quasi, 'near_task_mcs', []):
        if (
            int(other.id) in seen_mcs_ids
            or other.is_broken
            or getattr(other, 'is_energy_stranded', False)
        ):
            continue
        seen_mcs_ids.add(int(other.id))
        mcs_competition_raw += (
            TASK_MCS_COMPETITION_WEIGHT
            * _mcs_capacity_ratio(other)
            * distance_kernel(distance_km(quasi, other))
        )

    fcs_competition_raw = 0.0
    seen_fcs_ids = set()
    for fcs in getattr(quasi, 'near_available_fcs', []):
        if int(fcs.id) in seen_fcs_ids:
            continue
        seen_fcs_ids.add(int(fcs.id))
        # 每个空闲槽代表一个能够并行完成的替代订单。
        availability = float(fcs.available_slots)
        fcs_competition_raw += (
            availability * distance_kernel(distance_km(quasi, fcs))
        )
    for fcs in getattr(quasi, 'near_busy_fcs', []):
        if int(fcs.id) in seen_fcs_ids:
            continue
        seen_fcs_ids.add(int(fcs.id))
        fcs_competition_raw += (
            BUSY_FCS_COMPETITION_WEIGHT
            * distance_kernel(distance_km(quasi, fcs))
        )

    attraction = saturating_count(
        attraction_raw, ATTRACTION_COUNT_REFERENCE
    )
    immediate_iev_attraction = saturating_count(
        immediate_iev_raw, IMMEDIATE_IEV_COUNT_REFERENCE
    )
    mcs_competition = saturating_count(
        mcs_competition_raw, MCS_COMPETITION_REFERENCE
    )
    fcs_competition = saturating_count(
        fcs_competition_raw, FCS_COMPETITION_REFERENCE
    )

    # 只作为 reward 中的局部选位质量，不直接替 Actor 手工排序。
    # FCS 竞争比 MCS 竞争具有更强的冗余含义。
    access = distance_kernel(candidate_distance_km)
    desirability = (
        float(attraction)
        * access
        / (
            1.0
            + MCS_COMPETITION_IMPORTANCE * float(mcs_competition)
            + FCS_COMPETITION_IMPORTANCE * float(fcs_competition)
        )
    )

    return {
        'urgency_demand': float(np.clip(urgency_demand, 0.0, 1.0)),
        'distance_ratio': float(np.clip(
            candidate_distance_km / max(float(COMM_RANGE), EPSILON),
            0.0,
            1.0,
        )),
        'attraction': float(np.clip(attraction, 0.0, 1.0)),
        'immediate_iev_attraction': float(np.clip(
            immediate_iev_attraction, 0.0, 1.0
        )),
        'mcs_competition': float(np.clip(
            mcs_competition, 0.0, 1.0
        )),
        'fcs_competition': float(np.clip(
            fcs_competition, 0.0, 1.0
        )),
        'desirability': float(np.clip(desirability, 0.0, 1.0)),
    }


def compute_low_stay_metrics(mcs: MCS) -> Dict[str, float]:
    """计算 Low Actor“保持MCS当前位置”固定候选的局部特征。

    当前位置没有对应的 quasi 实体，因此自身紧迫需求记为 0；其余吸引力
    与竞争力直接聚合 MCS 当前通信范围内的局部邻居。这样 Actor 可以在
    不暴露全局信息或具体 IEV 身份的前提下，将原地等待与移动候选比较。
    """
    attraction_raw = 0.0
    immediate_iev_raw = 0.0
    seen_ev_ids = set()
    for quasi in getattr(mcs, 'near_quasi', []):
        if int(quasi.id) in seen_ev_ids or not quasi.is_quasi:
            continue
        seen_ev_ids.add(int(quasi.id))
        attraction_raw += (
            QUASI_ATTRACTION_WEIGHT * ev_service_opportunity(quasi)
            * distance_kernel(distance_km(mcs, quasi))
        )
    for iev in getattr(mcs, 'near_iev', []):
        if int(iev.id) in seen_ev_ids or not iev.is_iev:
            continue
        seen_ev_ids.add(int(iev.id))
        value = (
            IEV_ATTRACTION_WEIGHT * ev_service_opportunity(iev)
            * distance_kernel(distance_km(mcs, iev))
        )
        attraction_raw += value
        immediate_iev_raw += value

    mcs_competition_raw = 0.0
    seen_mcs_ids = {int(mcs.id)}
    for other in getattr(mcs, 'near_idle_mcs', []):
        if (
            int(other.id) in seen_mcs_ids
            or other.is_broken
            or getattr(other, 'is_energy_stranded', False)
        ):
            continue
        seen_mcs_ids.add(int(other.id))
        mcs_competition_raw += (
            _mcs_capacity_ratio(other)
            * distance_kernel(distance_km(mcs, other))
        )
    for other in getattr(mcs, 'near_task_mcs', []):
        if (
            int(other.id) in seen_mcs_ids
            or other.is_broken
            or getattr(other, 'is_energy_stranded', False)
        ):
            continue
        seen_mcs_ids.add(int(other.id))
        mcs_competition_raw += (
            TASK_MCS_COMPETITION_WEIGHT
            * _mcs_capacity_ratio(other)
            * distance_kernel(distance_km(mcs, other))
        )

    fcs_competition_raw = 0.0
    seen_fcs_ids = set()
    for fcs in getattr(mcs, 'near_available_fcs', []):
        if int(fcs.id) in seen_fcs_ids:
            continue
        seen_fcs_ids.add(int(fcs.id))
        availability = float(fcs.available_slots)
        fcs_competition_raw += availability * distance_kernel(
            distance_km(mcs, fcs)
        )
    for fcs in getattr(mcs, 'near_busy_fcs', []):
        if int(fcs.id) in seen_fcs_ids:
            continue
        seen_fcs_ids.add(int(fcs.id))
        fcs_competition_raw += (
            BUSY_FCS_COMPETITION_WEIGHT
            * distance_kernel(distance_km(mcs, fcs))
        )

    attraction = saturating_count(
        attraction_raw, ATTRACTION_COUNT_REFERENCE
    )
    immediate_iev_attraction = saturating_count(
        immediate_iev_raw, IMMEDIATE_IEV_COUNT_REFERENCE
    )
    mcs_competition = saturating_count(
        mcs_competition_raw, MCS_COMPETITION_REFERENCE
    )
    fcs_competition = saturating_count(
        fcs_competition_raw, FCS_COMPETITION_REFERENCE
    )
    desirability = (
        float(attraction)
        / (
            1.0
            + MCS_COMPETITION_IMPORTANCE * float(mcs_competition)
            + FCS_COMPETITION_IMPORTANCE * float(fcs_competition)
        )
    )
    return {
        'urgency_demand': 0.0,
        'distance_ratio': 0.0,
        'attraction': float(np.clip(attraction, 0.0, 1.0)),
        'immediate_iev_attraction': float(np.clip(
            immediate_iev_attraction, 0.0, 1.0
        )),
        'mcs_competition': float(np.clip(
            mcs_competition, 0.0, 1.0
        )),
        'fcs_competition': float(np.clip(
            fcs_competition, 0.0, 1.0
        )),
        'desirability': float(np.clip(desirability, 0.0, 1.0)),
    }
