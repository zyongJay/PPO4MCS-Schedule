"""
demo/matching.py — 全局最优匹配模块 (Target-based Replanning)

包含两个全局最优匹配器:
  1. ImmediateMatcher  — IEV → Idle MCS / Available FCS 即时匹配
  2. RechargeMatcher   — Idle MCS (RECHARGE) → Available FCS slots 补电匹配

两者均使用 Hungarian 算法 (Kuhn-Munkres) 求解全局最小代价匹配,
消除顺序依赖, 避免贪心导致的局部最优。

Target-based 变更: 不创建 Mission/Reservation。匹配成功时直接设置实体属性 (ev.start_charging / mcs.set_target / fcs.occupy_for_ev)。
"""

from typing import List, Tuple

import numpy as np

from config import *
from core import (EV, MCS, FCS, euclidean_distance)


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


def check_hard_constraints(ev: EV, r_type: str, r_obj):
    """硬约束检查。
      1. 距离 <= match_range
      2. MCS 剩余电量 >= EV.need_power + detour_power + MCS移动耗电
      3. EV 剩余电量足够移动到充电点 (中点)
    """
    dist_m = euclidean_distance(ev.pos[0], ev.pos[1], r_obj.pos[0], r_obj.pos[1])
    if dist_m / 1000.0 > COMM_RANGE:
        return False, np.inf

    charge_pos = [(ev.pos[0] + r_obj.pos[0]) / 2.0, (ev.pos[1] + r_obj.pos[1]) / 2.0] if r_type == "MCS" else list(
        r_obj.pos)
    next_track_pos = ev.track[ev.track_index + 1] if ev.track_index < len(ev.track) - 1 else ev.track[-1]
    detour_dist_km = (euclidean_distance(ev.pos[0], ev.pos[1], charge_pos[0], charge_pos[1])
                      + euclidean_distance(charge_pos[0], charge_pos[1], next_track_pos[0], next_track_pos[1])
                      - euclidean_distance(ev.pos[0], ev.pos[1], next_track_pos[0], next_track_pos[1])
                      ) / 1000.0
    if r_type == 'MCS':
        # EV 到中点距离 = dist / 2
        energy_to_charge_point = (dist_m / 2.0 / 1000.0) * POWER_UNIT
        if r_obj.remain < ev.need_power + detour_dist_km * POWER_UNIT + energy_to_charge_point:
            return False, np.inf
    else:
        # EV 到 FCS 距离 = dist
        energy_to_charge_point = (dist_m / 1000.0) * POWER_UNIT

    # EV 剩余电量不足以移动到充电点
    if ev.remain < energy_to_charge_point:
        return False, np.inf

    return True, detour_dist_km


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
            if mcs.is_idle and not mcs.is_recharging and not mcs.is_broken:
                resources.append(('MCS', mcs))

        for fcs in all_fcss:
            # Each free slot must occupy one Hungarian resource column.
            for _ in range(fcs.available_slots):
                resources.append(('FCS', fcs))

        if not resources:
            return [{'ev_id': ev.id, 'success': False} for ev in iev_set]

        # ── Step 3: 构建代价矩阵 ──
        cost, validity, detour = self.build_cost_matrix(iev_set, resources)

        # ── Step 4: 全局匹配 ──
        assignments = global_match(cost)

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
            charge_pos = [(ev.pos[0] + r_obj.pos[0]) / 2.0,
                          (ev.pos[1] + r_obj.pos[1]) / 2.0] if r_type == "MCS" else list(r_obj.pos)
            detour_dist_km = float(detour[iev_idx, res_idx])
            charge_power = ev.need_power + POWER_UNIT * detour_dist_km
            charge_time_min = charge_power / CHARGE_SPEED_PER_MIN

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
            })
            matched_iev_indices.add(iev_idx)

        # ── 未匹配的 IEV ──
        for i, ev in enumerate(iev_set):
            if i not in matched_iev_indices:
                results.append({'ev_id': ev.id, 'success': False})

        return results

    # ──────────────────────────────────────────
    # 代价矩阵构建
    # ──────────────────────────────────────────

    def build_cost_matrix(self, iev_set: List[EV], resources: List[Tuple]):
        """构建代价矩阵和有效性矩阵。"""
        INF = 1e9
        n_iev = len(iev_set)
        n_res = len(resources)

        cost = np.full((n_iev, n_res), INF, dtype=np.float64)  # 代价矩阵
        validity = np.zeros((n_iev, n_res), dtype=bool)  # 有效性矩阵
        detour = np.full((n_iev, n_res), INF, dtype=np.float64)  # 绕行距离矩阵

        for i, ev in enumerate(iev_set):
            for j, (r_type, r_obj) in enumerate(resources):
                is_valid, detour[i, j] = check_hard_constraints(ev, r_type, r_obj)
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
            charge_time_min = charge_power / CHARGE_SPEED_PER_MIN

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

        for i, mcs in enumerate(mcs_list):
            for j, fcs in enumerate(resources):
                dist_km = euclidean_distance(mcs.pos[0], mcs.pos[1], fcs.pos[0], fcs.pos[1]) / 1000.0

                # 硬性约束
                if dist_km > COMM_RANGE:
                    continue
                energy_to_fcs = dist_km * POWER_UNIT
                if mcs.remain < energy_to_fcs:
                    continue

                validity[i, j] = True

                dist_norm = dist_km / max(COMM_RANGE, 1.0)
                reliability = 1.0 - fcs.available_slots * 1.0 / max(1, fcs.capacity)
                cost[i, j] = self.w_dist * dist_norm + self.w_reliability * reliability

        return cost, validity
