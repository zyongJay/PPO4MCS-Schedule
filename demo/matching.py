"""
demo/matching.py — 全局最优匹配模块 (Target-based Replanning)

包含两个全局最优匹配器:
  1. ImmediateMatcher  — IEV → Idle MCS / Available FCS 即时匹配
  2. RechargeMatcher   — Idle MCS (RECHARGE) → Available FCS slots 补电匹配

两者均使用 Hungarian 算法 (Kuhn-Munkres) 求解全局最小代价匹配,
消除顺序依赖, 避免贪心导致的局部最优。

Target-based 变更: 不创建 Mission/Reservation。匹配成功时直接设置实体属性 (ev.start_charging / mcs.set_target / fcs.occupy_for_ev)。
"""

from typing import Dict, List, Tuple

import numpy as np

from config import *
from core import (EV, MCS, FCS, euclidean_distance)


# 有效替代供给并不等于一定能分配给当前 IEV。早期实现使用 2.0 的
# Poisson hazard 把平均反事实失败概率压到约 5%，使成功信用过度稀疏。
# 0.45 保留供给越多失败概率越低的单调性，同时对未来 slot/MCS 的分配
# 不确定性作保守校准。
COUNTERFACTUAL_SUPPLY_HAZARD_SCALE = 0.45
COUNTERFACTUAL_RAW_HAZARD_REFERENCE = 2.0


# ============================================================
# Hungarian 算法 (Kuhn-Munkres, O(n²m))
# ============================================================

def hungarian_min_cost(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    求解矩形最小代价指派问题 (Hungarian 算法)。

    输入: cost (n_rows × n_cols), 其中 n_rows <= n_cols
         不可行的边用大值 (INF) 填充。

    返回: [(row_idx, col_idx), ...] 其中 cost 有效 (非 INF) 的指派。
         未匹配的 row 不出现在结果中。
    """
    INF = 1e9
    n_rows, n_cols = cost.shape

    # Hungarian 要求 n_rows <= n_cols; 否则交换
    if n_rows > n_cols:
        transposed = True
        cost = cost.T
        n_rows, n_cols = n_cols, n_rows
    else:
        transposed = False

    # 使用大值替代 INF 参与计算 (但要明显大于正常 cost 以避开选择)
    BIG = INF * 0.8
    c = np.minimum(cost.copy(), BIG)

    n, m = n_rows, n_cols
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # 增广路径
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # 提取结果: p[j] = i, 即 col=j-1 被指派给 row=i-1
    assignments = []
    HALF_INF = INF * 0.5
    for j in range(1, m + 1):
        if p[j] != 0:
            row = p[j] - 1
            col = j - 1
            if cost[row, col] < HALF_INF:  # 仅保留有效匹配
                if transposed:
                    assignments.append((col, row))
                else:
                    assignments.append((row, col))

    return assignments


# ============================================================
# 全局最优匹配器
# ============================================================

def global_match(cost: np.ndarray) -> List[Tuple[int, int]]:
    """执行 Hungarian 全局匹配。"""
    n_iev, n_res = cost.shape
    if n_iev == 0 or n_res == 0:
        return []
    if n_iev <= n_res:
        return hungarian_min_cost(cost)
    else:
        assignments = hungarian_min_cost(cost.T)
        return [(col, row) for row, col in assignments]


def has_immediate_counterfactual_replacement(
    cost: np.ndarray,
    iev_index: int,
    removed_resource_index: int,
) -> bool:
    """移除当前 MCS 后重跑全局匹配，判断该 IEV 是否仍会即时成功。"""
    if cost.ndim != 2 or cost.shape[1] <= 1:
        return False
    reduced_cost = np.delete(cost, int(removed_resource_index), axis=1)
    return any(
        int(row_index) == int(iev_index)
        for row_index, _column_index in global_match(reduced_cost)
    )


def check_hard_constraints(
    ev: EV,
    r_type: str,
    r_obj,
    all_fcss: List[FCS] | None = None,
):
    """硬约束检查。
      1. 距离 <= match_range
      2. MCS 剩余电量能覆盖 EV 服务电量与 MCS 移动耗电
      3. EV 剩余电量足够移动到充电点 (中点)
      4. MCS 完成充电后的剩余电量严格大于前往最近
         物理 FCS 的能耗
    """
    dist_m = euclidean_distance(ev.pos[0], ev.pos[1], r_obj.pos[0], r_obj.pos[1])
    if dist_m / 1000.0 > COMM_RANGE:
        return False, np.inf

    charge_pos = [(ev.pos[0] + r_obj.pos[0]) / 2.0, (ev.pos[1] + r_obj.pos[1]) / 2.0] if r_type == "MCS" else list(
        r_obj.pos)
    now_track_pos = ev.track[ev.track_index]
    next_track_pos = ev.track[ev.track_index + 1] if ev.track_index < len(ev.track) - 1 else ev.track[-1]
    detour_dist_km = (euclidean_distance(ev.pos[0], ev.pos[1], charge_pos[0], charge_pos[1])
                      + euclidean_distance(charge_pos[0], charge_pos[1], next_track_pos[0], next_track_pos[1])
                      # - euclidean_distance(ev.pos[0], ev.pos[1], next_track_pos[0], next_track_pos[1])
                      - euclidean_distance(now_track_pos[0], now_track_pos[1], next_track_pos[0], next_track_pos[1])
                      ) / 1000.0
    if r_type == 'MCS':
        # EV 到中点距离 = dist / 2
        energy_to_charge_point = (dist_m / 2.0 / 1000.0) * POWER_UNIT
        planned_charge_power = (
            float(ev.need_power) + detour_dist_km * POWER_UNIT
        )
        if r_obj.remain < planned_charge_power + energy_to_charge_point:
            return False, np.inf

        # 即时匹配必须同时保证服务后返回物理 FCS 的绝对
        # 安全性。这里故意使用严格 >：恰好耗尽电量到达 FCS
        # 不属于可行匹配。
        if not all_fcss:
            return False, np.inf
        nearest_fcs_energy = min(
            euclidean_distance(
                charge_pos[0], charge_pos[1], fcs.pos[0], fcs.pos[1]
            ) / 1000.0 * POWER_UNIT
            for fcs in all_fcss
        )
        post_service_remain = (
            float(r_obj.remain)
            - energy_to_charge_point
            - planned_charge_power
        )
        if not post_service_remain > nearest_fcs_energy:
            return False, np.inf
    else:
        # EV 到 FCS 距离 = dist
        energy_to_charge_point = (dist_m / 1000.0) * POWER_UNIT

    # EV 剩余电量不足以移动到充电点
    if ev.remain < energy_to_charge_point:
        return False, np.inf

    return True, detour_dist_km


def _travel_time_min(first_pos, second_pos) -> float:
    distance_m = euclidean_distance(
        first_pos[0], first_pos[1], second_pos[0], second_pos[1]
    )
    return float(distance_m / max(float(MOVE_SPEED), 1e-8) / 60.0)


def _ev_failure_horizon_min(ev: EV) -> float:
    """估计 IEV 在没有新匹配时仍可获救的剩余分钟数。"""
    wait_steps = max(
        int(MAX_WAIT_TIME_STEPS) + 1 - int(ev.wait_time_steps), 1
    )
    wait_horizon = float(wait_steps * STEP_DURATION_MIN)
    usable_energy = max(float(ev.remain) - float(EV_LOWEST_POWER), 0.0)
    energy_distance_km = usable_energy / max(float(POWER_UNIT), 1e-8)
    energy_horizon = (
        energy_distance_km * 1000.0
        / max(float(MOVE_SPEED), 1e-8)
        / 60.0
    )
    # 仿真按离散 step 检查失败，至少保留一个 step 的反应时间。
    energy_horizon = max(float(energy_horizon), float(STEP_DURATION_MIN))
    return float(max(min(wait_horizon, energy_horizon), STEP_DURATION_MIN))


def _timely_success_probability(
    access_time_min: float,
    failure_horizon_min: float,
) -> float:
    """将到达/资源释放时间相对失败期限的裕度平滑映射到成功概率。"""
    horizon = max(float(failure_horizon_min), float(STEP_DURATION_MIN))
    scale = max(0.25 * horizon, 0.5 * float(STEP_DURATION_MIN), 1e-8)
    z = np.clip((horizon - float(access_time_min)) / scale, -12.0, 12.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def _fcs_slot_release_time_min(fcs: FCS, slot_index: int) -> float:
    if slot_index < 0 or slot_index >= int(fcs.num_slots):
        return float('inf')
    if slot_index >= len(fcs.slot_target) or fcs.slot_target[slot_index] is None:
        return 0.0
    target = fcs.slot_target[slot_index]
    travel = 0.0
    if not bool(getattr(target, 'is_arrive', False)):
        travel = _travel_time_min(target.pos, fcs.pos)
    charge = max(float(fcs.slot_charge_remain_min[slot_index]), 0.0)
    return float(travel + charge)


def _mcs_future_availability(mcs: MCS) -> Tuple[float, List[float], float]:
    """返回 MCS 可再次服务的预计时间、位置和电量。"""
    if (
        mcs.is_broken
        or getattr(mcs, 'is_energy_stranded', False)
    ):
        return float('inf'), list(mcs.pos), 0.0
    if mcs.is_idle and not mcs.is_recharging:
        return 0.0, list(mcs.pos), max(float(mcs.remain), 0.0)
    target_pos = getattr(mcs, 'current_target_pos', None)
    if target_pos is None:
        return float('inf'), list(mcs.pos), max(float(mcs.remain), 0.0)
    travel = _travel_time_min(mcs.pos, target_pos)
    travel_distance_km = euclidean_distance(
        mcs.pos[0], mcs.pos[1], target_pos[0], target_pos[1]
    ) / 1000.0
    future_remain = max(
        float(mcs.remain) - travel_distance_km * float(POWER_UNIT), 0.0
    )
    pending_energy = max(float(getattr(mcs, 'charge_power_kwh', 0.0)), 0.0)
    if mcs.is_recharging or getattr(mcs, 'current_target_type', '') == 'FCS':
        future_remain += pending_energy
    else:
        future_remain = max(future_remain - pending_energy, 0.0)
    availability = travel + max(
        float(getattr(mcs, 'charge_time_remain_min', 0.0)), 0.0
    )
    return float(availability), list(target_pos), float(future_remain)


def estimate_counterfactual_failure_probability(
    ev: EV,
    selected_mcs: MCS,
    all_mcss: List[MCS],
    all_fcss: List[FCS],
    *,
    active_iev_count: int = 1,
    exclude_immediate_resources: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """估计“当前 MCS 不接单”时 IEV 最终失败的概率。

    替代成功概率同时考虑 IEV 剩余时间/能量、FCS 当前队列与未来释放、
    其他 MCS 的到达时间、电量和已有任务。多个替代资源按独立救援机会
    聚合，并用当前 IEV 供需压力做保守校准。
    """
    horizon = _ev_failure_horizon_min(ev)
    alternative_probabilities: List[float] = []
    fcs_probabilities: List[float] = []
    mcs_probabilities: List[float] = []

    for fcs in all_fcss:
        distance_km = euclidean_distance(
            ev.pos[0], ev.pos[1], fcs.pos[0], fcs.pos[1]
        ) / 1000.0
        required_energy = distance_km * float(POWER_UNIT)
        if float(ev.remain) + 1e-8 < required_energy:
            continue
        travel_time = _travel_time_min(ev.pos, fcs.pos)
        for slot_index in range(int(fcs.num_slots)):
            if (
                exclude_immediate_resources
                and (
                    slot_index >= len(fcs.slot_target)
                    or fcs.slot_target[slot_index] is None
                )
            ):
                continue
            release_time = _fcs_slot_release_time_min(fcs, slot_index)
            access_time = max(travel_time, release_time)
            probability = _timely_success_probability(access_time, horizon)
            fcs_probabilities.append(probability)

    for mcs in all_mcss:
        if int(mcs.id) == int(selected_mcs.id):
            continue
        if (
            exclude_immediate_resources
            and mcs.is_idle
            and not mcs.is_recharging
        ):
            continue
        availability, future_pos, future_remain = _mcs_future_availability(mcs)
        if not np.isfinite(availability):
            continue
        distance_m = euclidean_distance(
            ev.pos[0], ev.pos[1], future_pos[0], future_pos[1]
        )
        distance_km = distance_m / 1000.0
        if distance_km > float(COMM_RANGE):
            continue
        rendezvous_energy = distance_km / 2.0 * float(POWER_UNIT)
        if float(ev.remain) + 1e-8 < rendezvous_energy:
            continue
        charge_pos = [
            (float(ev.pos[0]) + float(future_pos[0])) / 2.0,
            (float(ev.pos[1]) + float(future_pos[1])) / 2.0,
        ]
        if not all_fcss:
            continue
        nearest_fcs_energy = min(
            euclidean_distance(
                charge_pos[0], charge_pos[1], fcs.pos[0], fcs.pos[1]
            ) / 1000.0 * float(POWER_UNIT)
            for fcs in all_fcss
        )
        required_service_energy = (
            max(float(ev.need_power), 0.0) + rendezvous_energy
        )
        if not (
            future_remain - required_service_energy > nearest_fcs_energy
        ):
            continue
        rendezvous_time = (
            distance_m / 2.0
            / max(float(MOVE_SPEED), 1e-8)
            / 60.0
        )
        access_time = availability + rendezvous_time
        probability = _timely_success_probability(access_time, horizon)
        # 已有任务越长，未来状态估计越不确定，适当降低其替代可靠性。
        reliability = float(np.exp(-availability / max(horizon, 1e-8) * 0.35))
        mcs_probabilities.append(probability * reliability)

    alternative_probabilities.extend(
        np.clip(np.asarray(fcs_probabilities), 0.0, 1.0)
    )
    alternative_probabilities.extend(
        np.clip(np.asarray(mcs_probabilities), 0.0, 1.0)
    )
    if alternative_probabilities:
        # 多个候选并非独立事件：它们还会被同一批 IEV 竞争。把及时可用
        # 概率之和视为有效供给强度，并按当前 IEV 数量均摊；Poisson 零到达
        # 概率给出“最终仍无人能救援”的保守估计。这样不会因同一 FCS 的
        # 多个相关 slot 简单相乘而把失败概率压到接近 0。
        effective_supply = float(sum(alternative_probabilities))
        demand = max(float(active_iev_count), 1.0)
        supply_per_iev = effective_supply / demand
        raw_failure_probability = float(np.exp(
            -COUNTERFACTUAL_RAW_HAZARD_REFERENCE * supply_per_iev
        ))
        failure_probability = float(np.exp(
            -COUNTERFACTUAL_SUPPLY_HAZARD_SCALE * supply_per_iev
        ))
    else:
        raw_failure_probability = 1.0
        failure_probability = 1.0
    failure_probability = float(np.clip(failure_probability, 0.0, 1.0))
    diagnostics = {
        'failure_horizon_min': float(horizon),
        'fcs_alternative_count': float(len(fcs_probabilities)),
        'mcs_alternative_count': float(len(mcs_probabilities)),
        'best_fcs_success_probability': float(
            max(fcs_probabilities, default=0.0)
        ),
        'best_mcs_success_probability': float(
            max(mcs_probabilities, default=0.0)
        ),
        'effective_alternative_supply': float(sum(alternative_probabilities)),
        'active_iev_count': float(max(int(active_iev_count), 1)),
        'raw_counterfactual_failure_probability': float(
            raw_failure_probability
        ),
        'counterfactual_supply_hazard_scale': float(
            COUNTERFACTUAL_SUPPLY_HAZARD_SCALE
        ),
        'excluded_immediate_resources': float(
            bool(exclude_immediate_resources)
        ),
        'counterfactual_failure_probability': failure_probability,
    }
    return failure_probability, diagnostics


class ImmediateMatcher:
    """
    全局最优即时匹配器 (Target-based).

    算法流程:
      1. 收集所有 IEV → 集合 A (行)
      2. 收集所有 Idle MCS + Available FCS slots → 集合 B (列)
      3. 构建代价矩阵 cost[A][B]:
          硬约束通过 → cost = 加权评分 (距离 + 可靠性)
          硬约束不通过 → cost = INF
      4. Hungarian 算法求解全局最小代价匹配
      5. 执行匹配: EV→充电, MCS→set_target("EV", ...) state→TASK, FCS→占用slot
    """

    def __init__(self):
        self.w_dist = 1.0
        self.w_reliability = 0.5

    # ──────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────

    def match_all(self, iev_list: List[EV], all_mcss: List[MCS], all_fcss: List[FCS]):
        """
        对所有 IEV 执行全局最优匹配。
        Returns:[{ev_id, success, provider_type, provider_id, ...}, ...]
        results 中 success=“True” 的是匹配成功的，执行充电任务；success=“False”的是匹配失败的，加入Agent队列
        """
        # ── Step 1: 构造 IEV 集合 ──
        iev_set = [ev for ev in iev_list if ev.is_iev]
        if not iev_set:
            return []
        # ── Step 2: 构造资源集合 ──
        resources = []  # List[(type, provider_obj)]

        for mcs in all_mcss:
            if (
                mcs.is_idle
                and not mcs.is_recharging
                and not mcs.is_broken
                and not getattr(mcs, 'is_energy_stranded', False)
                and not getattr(mcs, 'active_serve_has_matched', False)
            ):
                resources.append(('MCS', mcs))

        for fcs in all_fcss:
            # Each free slot must occupy one Hungarian resource column.
            for _ in range(fcs.available_slots):
                resources.append(('FCS', fcs))

        if not resources:
            return [{'ev_id': ev.id, 'success': False} for ev in iev_set]

        # ── Step 3: 构建代价矩阵 ──
        cost, validity, detour = self.build_cost_matrix(
            iev_set, resources, all_fcss
        )

        # ── Step 4: 全局匹配 ──
        assignments = global_match(cost)
        immediate_replacement_by_pair: Dict[Tuple[int, int], bool] = {}
        for iev_idx, res_idx in assignments:
            if not validity[iev_idx, res_idx]:
                continue
            resource_type, _resource = resources[res_idx]
            if resource_type != 'MCS':
                continue
            immediate_replacement_by_pair[(iev_idx, res_idx)] = (
                has_immediate_counterfactual_replacement(
                    cost, iev_idx, res_idx
                )
            )

        # ── Step 5: 执行有效匹配 ──
        results = []
        matched_iev_indices = set()
        for iev_idx, res_idx in assignments:
            # 筛选合法匹配
            if not validity[iev_idx, res_idx]:
                continue

            # 充电双方对象
            ev = iev_set[iev_idx]
            r_type, r_obj = resources[res_idx]
            # 使用与正式匹配完全相同的 hard-constraint validity 矩阵统计
            # 可替代资源，作为系统边际价值代理，避免仅凭距离臆测。
            feasible_mcs_count = sum(
                bool(validity[iev_idx, index]) and resource_type == 'MCS'
                for index, (resource_type, _resource) in enumerate(resources)
            )
            feasible_fcs_slot_count = sum(
                bool(validity[iev_idx, index]) and resource_type == 'FCS'
                for index, (resource_type, _resource) in enumerate(resources)
            )
            charge_pos = [(ev.pos[0] + r_obj.pos[0]) / 2.0,
                          (ev.pos[1] + r_obj.pos[1]) / 2.0] if r_type == "MCS" else list(r_obj.pos)
            detour_dist_km = float(detour[iev_idx, res_idx])
            charge_power = ev.need_power + POWER_UNIT * detour_dist_km
            charge_time_min = charge_power / CHARGE_SPEED_PER_MIN
            counterfactual_failure_probability = 0.0
            rescue_diagnostics: Dict[str, float] = {}
            if r_type == "MCS":
                immediate_replacement = bool(
                    immediate_replacement_by_pair.get(
                        (iev_idx, res_idx), False
                    )
                )
                if immediate_replacement:
                    counterfactual_failure_probability = 0.0
                    rescue_diagnostics = {
                        'immediate_counterfactual_replacement': 1.0,
                        'counterfactual_failure_probability': 0.0,
                    }
                else:
                    (
                        counterfactual_failure_probability,
                        rescue_diagnostics,
                    ) = estimate_counterfactual_failure_probability(
                        ev,
                        r_obj,
                        all_mcss,
                        all_fcss,
                        active_iev_count=len(iev_set),
                        exclude_immediate_resources=True,
                    )
                    rescue_diagnostics[
                        'immediate_counterfactual_replacement'
                    ] = 0.0

            # ── 充电绑定 ──
            ev.set_target(
                obj=r_obj,
                provider_id=r_obj.id,
                provider_type=r_type,
                charge_pos=charge_pos,
                charge_power_kwh=charge_power,
                charge_time_min=charge_time_min,
                detour_dist_km=detour_dist_km
            )

            if r_type == "MCS":
                slot_idx = r_obj.set_target(
                    obj=ev,
                    target_type="IEV",
                    target_id=ev.id,
                    target_pos=charge_pos,
                    charge_power=charge_power,
                    charge_time=charge_time_min,
                )
            else:  # FCS
                slot_idx = r_obj.set_target(
                    target=ev,
                    target_type="IEV",
                    target_id=ev.id,
                    charge_power_kwh=charge_power,
                    charge_time_min=charge_time_min,
                )

            # 匹配结果
            results.append({
                'ev_id': ev.id, 'success': True,
                'provider_type': r_type,
                'provider_id': r_obj.id,
                'slot_idx': slot_idx,
                'charge_power': charge_power,
                'charge_time': charge_time_min,
                'cost_score': float(cost[iev_idx, res_idx]),
                'feasible_mcs_count': int(feasible_mcs_count),
                'feasible_fcs_slot_count': int(feasible_fcs_slot_count),
                'counterfactual_failure_probability': float(
                    counterfactual_failure_probability
                ),
                'rescue_diagnostics': dict(rescue_diagnostics),
            })
            matched_iev_indices.add(iev_idx)

        # ── 未匹配的 IEV ──
        for i, ev in enumerate(iev_set):
            if i not in matched_iev_indices:
                results.append({
                    'ev_id': ev.id,
                    'success': False,
                    'feasible_mcs_count': int(sum(
                        bool(validity[i, index]) and resource_type == 'MCS'
                        for index, (resource_type, _resource)
                        in enumerate(resources)
                    )),
                    'feasible_fcs_slot_count': int(sum(
                        bool(validity[i, index]) and resource_type == 'FCS'
                        for index, (resource_type, _resource)
                        in enumerate(resources)
                    )),
                })

        return results

    # ──────────────────────────────────────────
    # 代价矩阵构建
    # ──────────────────────────────────────────

    def build_cost_matrix(
        self,
        iev_set: List[EV],
        resources: List[Tuple],
        all_fcss: List[FCS] | None = None,
    ):
        """构建代价矩阵和有效性矩阵。"""
        INF = 1e9
        n_iev = len(iev_set)
        n_res = len(resources)

        cost = np.full((n_iev, n_res), INF, dtype=np.float64)  # 代价矩阵
        validity = np.zeros((n_iev, n_res), dtype=bool)  # 有效性矩阵
        detour = np.full((n_iev, n_res), INF, dtype=np.float64)  # 绕行距离矩阵

        for i, ev in enumerate(iev_set):
            for j, (r_type, r_obj) in enumerate(resources):
                is_valid, detour[i, j] = check_hard_constraints(
                    ev, r_type, r_obj, all_fcss
                )
                if not is_valid:
                    continue
                validity[i, j] = True
                cost[i, j] = self.compute_cost(ev, r_type, r_obj)

        return cost, validity, detour

    def compute_cost(self, ev: EV, r_type: str, r_obj) -> float:
        """计算 (IEV, 资源) 对的综合评分代价。"""
        dist_km = euclidean_distance(ev.pos[0], ev.pos[1], r_obj.pos[0], r_obj.pos[1]) / 1000.0
        dist_norm = dist_km / max(COMM_RANGE, 1.0)

        if r_type == 'MCS':
            dist_norm /= 2      # 充电位置是二者中间
            reliability = 1.0 - r_obj.remain * 1.0 / MCS_BATTERY_CAPACITY
        else:
            reliability = 1.0 - r_obj.available_slots * 1.0 / max(1, r_obj.capacity)

        cost = self.w_dist * dist_norm + self.w_reliability * reliability
        return cost


# ============================================================
# MCS-to-FCS 补电全局最优匹配器
# ============================================================
class RechargeMatcher:
    """
    MCS-to-FCS 补电全局最优匹配器 (Target-based).

    将所有需要补电的 Idle MCS 与所有可用 FCS 充电位进行全局最优匹配，
    使用 Hungarian 算法最小化总距离成本。

    Target-based 变更: 返回匹配结果，由环境调用 mcs.set_target("FCS", ...) 完成设置。
    """

    def __init__(self):
        self.w_dist = 1.0
        self.w_reliability = 0.5

    def match_all(self, recharge_mcss: List[MCS], all_fcss: List[FCS]) -> List[dict]:
        """
        对所有需补电的 MCS 执行全局最优匹配。

        Returns:
            [{mcs_id, success, fcs_id, slot_idx, charge_power, charge_time, cost}, ...]
        """
        n_mcs = len(recharge_mcss)
        if n_mcs == 0:
            return []

        # ── Step 1: 构造 FCS 资源集合 ──
        resources = []
        for fcs in all_fcss:
            # Duplicate the FCS column once per currently available slot.
            for _ in range(fcs.available_slots):
                resources.append(fcs)

        if not resources:
            return [{'mcs_id': mcs.id, 'success': False} for mcs in recharge_mcss]

        # ── Step 2: 预计算每个 MCS 的充电参数 ──
        mcs_charge_info = []
        for mcs in recharge_mcss:
            needed = MCS_BATTERY_CAPACITY - mcs.remain
            mcs_charge_info.append(needed)

        # ── Step 3: 构建距离代价矩阵 ──
        cost, validity = self.build_recharge_cost_matrix(recharge_mcss, resources)

        # ── Step 4: Hungarian 全局匹配 ──
        assignments = global_match(cost)

        # ── Step 5: 补电绑定 ──
        results = []
        matched_mcs = set()
        for mcs_idx, res_idx in assignments:
            # Recheck feasibility before binding; Hungarian may return an INF assignment.
            if not validity[mcs_idx, res_idx]:
                continue

            mcs = recharge_mcss[mcs_idx]
            fcs = resources[res_idx]
            charge_power = mcs_charge_info[mcs_idx]
            charge_power += euclidean_distance(mcs.pos[0], mcs.pos[1], fcs.pos[0], fcs.pos[1]) / 1000.0 * POWER_UNIT
            charge_power = min(charge_power, MAX_RECHARGE_PER_SESSION_KWH)
            charge_time_min = charge_power / RECHARGE_SPEED_PER_MIN

            mcs.set_target(
                obj=fcs,
                target_type='FCS',
                target_id=fcs.id,
                target_pos=list(fcs.pos),
                charge_power=charge_power,
                charge_time=charge_time_min,
            )

            slot_idx = fcs.set_target(
                target=mcs,
                target_type="MCS",
                target_id=mcs.id,
                charge_power_kwh=charge_power,
                charge_time_min=charge_time_min
            )

            # 匹配结果
            results.append({
                'mcs_id': mcs.id, 'success': True,
                'fcs_id': fcs.id, 'slot_idx': slot_idx,
                'charge_power': charge_power, 'charge_time': charge_time_min,
                'cost_score': float(cost[mcs_idx, res_idx]),
            })
            matched_mcs.add(mcs_idx)

        # ── 未匹配的 MCS ──
        for i, mcs in enumerate(recharge_mcss):
            if i not in matched_mcs:
                results.append({'mcs_id': mcs.id, 'success': False})

        return results

    def build_recharge_cost_matrix(self, mcs_list: List[MCS], resources: List[FCS]) -> Tuple[np.ndarray, np.ndarray]:
        """构建距离代价矩阵。"""
        n_mcs = len(mcs_list)
        n_res = len(resources)
        cost = np.full((n_mcs, n_res), 1e9, dtype=np.float64)
        validity = np.zeros((n_mcs, n_res), dtype=bool)

        # Recharge 是全图规划，距离归一化使用本轮候选中的最大距离；
        # COMM_RANGE 不再作为硬约束。
        distance_scale_km = max((
            euclidean_distance(
                mcs.pos[0], mcs.pos[1], fcs.pos[0], fcs.pos[1]
            ) / 1000.0
            for mcs in mcs_list
            for fcs in resources
        ), default=1.0)

        for i, mcs in enumerate(mcs_list):
            if (
                mcs.is_broken
                or getattr(mcs, 'is_energy_stranded', False)
                or mcs.is_recharging
                or not mcs.is_idle
            ):
                continue
            for j, fcs in enumerate(resources):
                dist_km = euclidean_distance(mcs.pos[0], mcs.pos[1], fcs.pos[0], fcs.pos[1]) / 1000.0

                # 硬性约束
                energy_to_fcs = dist_km * POWER_UNIT
                if not float(mcs.remain) > energy_to_fcs:
                    continue

                validity[i, j] = True

                dist_norm = dist_km / max(distance_scale_km, 1e-8)
                reliability = 1.0 - fcs.available_slots * 1.0 / max(1, fcs.capacity)
                cost[i, j] = self.w_dist * dist_norm + self.w_reliability * reliability

        return cost, validity
