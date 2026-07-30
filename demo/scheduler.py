"""
demo/scheduler.py — 调度器 (Target-based Replanning)

仿真流程中的调度决策模块，为 agent 列表中的智能体生成 action。

1. WaitingScheduler (IEV agent 调度):
   - 为未匹配的 IEV 直接生成 action
   - 策略: 选择最近的任务中 MCS 或繁忙 FCS 作为等待目标
   - 无可选目标时沿自身轨迹移动

2. IdleScheduler (idle MCS agent 调度, 两级决策):
   - High-Actor:  决定模式 "Recharge"(补电) 或 "Serve"(预部署跟踪 quasi)
   - Low-Actor:   在 "Serve" 模式下选择具体的跟踪目标

Target-based 变更: 调度器只生成 action dict, 不直接修改实体充电绑定。
充电绑定由 ImmediateMatcher / RechargeMatcher 在匹配阶段完成。
"""

from typing import Any, Dict, List

from config import *
from core import (
    EV, MCS, FCS, euclidean_distance,
)


# ============================================================
# WaitingScheduler — IEV agent 调度
# ============================================================

class WaitingScheduler:
    """IEV 等待调度器 (Target-based Replanning)。

    为 ImmediateMatcher 未匹配的 IEV 生成移动 action。
    策略: 向最近的 busy MCS 或 busy FCS 移动 (缩短等待距离),
          无可选目标时沿轨迹继续行驶。
    """

    def select_action(self, iev: EV, all_mcss: List[MCS], all_fcss: List[FCS]) -> Dict[str, Any]:
        """为 IEV 智能体选择等待目标, 返回 action dict。

        Args:
            iev: 当前 IEV 智能体
            all_mcss: 全部 MCS 列表
            all_fcss: 全部 FCS 列表

        Returns:
            action dict: {"target_pos": [x, y]}
        """
        # ── 收集候选等待目标 ──
        candidates: List[Dict] = []  # [{'pos': (x,y), 'dist': d}, ...]

        # 繁忙 MCS (TASK 状态, 可能在附近释放)
        for mcs in iev.near_task_mcs:
            if mcs.is_broken:
                continue
            dist = euclidean_distance(iev.pos[0], iev.pos[1], mcs.pos[0], mcs.pos[1])
            candidates.append({'pos': (mcs.pos[0], mcs.pos[1]), 'dist': dist})

        # 繁忙 FCS (全部占用, 等待空位)
        for fcs in iev.near_busy_fcs:
            dist = euclidean_distance(iev.pos[0], iev.pos[1], fcs.pos[0], fcs.pos[1])
            candidates.append({'pos': (fcs.pos[0], fcs.pos[1]), 'dist': dist})

        # ── 选择最近候选 ──
        if candidates:
            candidates.sort(key=lambda c: c['dist'])
            best = candidates[0]
            return {'target_pos': [best['pos'][0], best['pos'][1]]}

        # ── 无候选时, 沿轨迹移动 ──
        return self._follow_track(iev)

    def _follow_track(self, iev: EV) -> Dict[str, Any]:
        """沿轨迹移动一步。"""
        if iev.track is not None and iev.track_index + 1 < len(iev.track):
            next_wp = iev.track[iev.track_index + 1]
            return {'target_pos': [float(next_wp[0]), float(next_wp[1])]}
        else:
            # 轨迹终点或无轨迹, 原地等待
            return {'target_pos': [iev.pos[0], iev.pos[1]]}


# ============================================================
# IdleScheduler — idle MCS agent 调度 (两级决策)
# ============================================================

class IdleScheduler:
    """Idle MCS 调度器 (Target-based Replanning, 两级决策)。

    High-Actor:  根据 MCS 电量判断模式 —— "Recharge"(去FCS补电) 或 "Serve"(跟踪quasi预部署)
    Low-Actor:   在 "Serve" 模式下, 选择最近的 quasi EV 作为跟踪目标

    注意: "Recharge" 模式下的 FCS 绑定由外部 (run_test.py) 调用 RechargeMatcher 完成,
          本调度器只负责模式决策和目标选择。
    """

    def high_level_decide(self, mcs: MCS) -> str:
        """High-Actor: 决定 MCS 当前步的运行模式。

        Args:
            mcs: 当前 idle MCS 智能体

        Returns:
            "Recharge" — 电量低于阈值, 需要去 FCS 补电
            "Serve"    — 电量充足, 可跟踪 quasi EV 预部署
        """
        if mcs.remain < MCS_RECHARGE_THRESHOLD:
            return "Recharge"
        return "Serve"

    def low_level_actor(self, mcs: MCS, all_evs: List[EV]) -> Dict[str, Any]:
        """Low-Actor: 在 "Serve" 模式下选择跟踪目标。

        策略:
          1. 优先选择 near_quasi 中距离最近的 quasi EV
          2. 若无 quasi, 选择 near_iev 中距离最近的 IEV
          3. 若均无, 原地待命

        Args:
            mcs: 当前 idle MCS 智能体
            all_evs: 全部 EV 列表 (备用, 当 near_* 为空时扩大搜索)

        Returns:
            action dict: {"mode": "Serve", "target_pos": [x, y]}
        """
        best_pos = None
        best_dist = float('inf')

        # ── 优先 quasi EV ──
        for ev in mcs.near_quasi:
            if ev.fail_charge or ev.is_normal:
                continue
            dist = euclidean_distance(mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1])
            if dist < best_dist:
                best_dist = dist
                best_pos = (ev.pos[0], ev.pos[1])

        # ── 次选 IEV ──
        if best_pos is None:
            for ev in mcs.near_iev:
                if ev.fail_charge or ev.is_normal:
                    continue
                dist = euclidean_distance(mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (ev.pos[0], ev.pos[1])

        # ── 无候选: 原地待命 ──
        if best_pos is None:
            best_pos = (mcs.pos[0], mcs.pos[1])

        return {
            'mode': 'Serve',
            'target_pos': [best_pos[0], best_pos[1]],
        }
