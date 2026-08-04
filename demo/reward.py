"""
demo/reward.py — 奖励计算器 (Target-based Replanning)

为 MCS / IEV 智能体提供 step 级奖励信号。
参照 env/world.py::mix_get_reward_n() 的奖励设计思路:

  MCS 奖励:
    - 局部: 对 near_quasi 的吸引力 + 对 near_iev 的潜在服务价值
    - 信用分配: 成功被匹配到充电任务获得正奖励
    - 惩罚: 移动开销、空闲惩罚、低电量惩罚、附近失败 IEV 惩罚

  IEV 奖励:
    - 局部: 对 near_task_mcs 的潜在充电机会吸引力
    - 信用分配: 充电成功获得正奖励, 充电失败获得负奖励
    - 惩罚: 等待惩罚 (与等待时间成正比)

  全局奖励: 充电功率比 + 成功率信号, 与局部奖励按 0.3/0.7 权重混合
"""

from typing import Dict, List

import numpy as np

from config import *
from core import EV, MCS, euclidean_distance


class RewardBuilder:
    """奖励构建器 (简化 APF + 信用分配)。

    参照 env/world.py::mix_get_reward_n() 的设计:
      - 局部 APF 奖励 (吸引/排斥/竞争)
      - Credit Assignment (匹配成功奖励)
      - 全局信号: 充电功率比 + 成功率
      - 混合: reward = 0.3 * r_global + 0.7 * tanh(r_local * scale)
    """

    def __init__(self, local_scale: float = 0.3, global_weight: float = 0.3):
        self.local_scale = local_scale
        self.global_weight = global_weight
        self.local_weight = 1.0 - global_weight

        # ── 步级统计 (由外部在每步开始前重置) ──
        self.step_success_count = 0
        self.step_fail_count = 0

    # ============================================================
    # MCS 奖励
    # ============================================================

    def compute_mcs_reward(self, mcs: MCS, event: Dict, MCSState=None) -> Dict[str, float]:
        """Return the MCS step reward together with auditable components.

        Rewards are based on what actually happened during the step, not on a
        fixed reward attached to the requested action.  This is also suitable
        for accumulating option returns while an MCS is serving or recharging.
        """
        max_move_energy = max(
            (MAX_MOVE_PER_STEP / 1000.0) * POWER_UNIT,
            1e-8,
        )
        service_kwh = max(float(event.get('service_kwh', 0.0)), 0.0)
        recharged_kwh = max(float(event.get('recharged_kwh', 0.0)), 0.0)
        movement_energy = max(float(event.get('movement_energy_kwh', 0.0)), 0.0)

        previous_remain = float(event.get('previous_remain_kwh', mcs.remain))
        current_remain = float(event.get('current_remain_kwh', mcs.remain))
        threshold = max(float(MCS_RECHARGE_THRESHOLD), 1e-8)
        previous_risk = float(np.clip((threshold - previous_remain) / threshold, 0.0, 1.0))
        current_risk = float(np.clip((threshold - current_remain) / threshold, 0.0, 1.0))

        components = {
            'service': service_kwh / max(MAX_CHARGE_PER_SESSION_KWH, 1e-8),
            'recharge': recharged_kwh / max(MAX_RECHARGE_PER_SESSION_KWH, 1e-8),
            'movement': -0.10 * movement_energy / max_move_energy,
            'wait': -0.02 if event.get('waited', False) else 0.0,
            'recharge_match_failure': (
                -0.05
                if event.get('requested_mode') == 'Recharge'
                and not event.get('recharge_matched', False)
                else 0.0
            ),
            'broken': -2.0 if event.get('newly_broken', False) else 0.0,
            'battery_potential': 0.20 * (previous_risk - current_risk),
        }
        components['total'] = float(sum(components.values()))
        return {name: float(value) for name, value in components.items()}

    # ============================================================
    # IEV 奖励
    # ============================================================

    def compute_iev_reward(self, ev: EV, info: Dict) -> float:
        """计算单个 IEV 智能体的步级奖励。

        参照 env/world.py IEV 局部奖励:
          1. task MCS 吸引力 (剩余电量足够服务)
          2. 竞争惩罚 (附近其他 IEV)
          3. Credit Assignment: 充电成功/失败
        """
        r_local = 0.0
        max_charge = MAX_CHARGE_PER_SESSION_KWH

        # ── 1. task MCS 吸引力 ──
        for mcs in ev.near_task_mcs:
            if mcs.is_broken:
                continue
            # MCS 到达充电位置后的剩余电量
            dist_m_to_charge = euclidean_distance(
                mcs.pos[0], mcs.pos[1],
                mcs.current_target_pos[0] if mcs.current_target_pos else mcs.pos[0],
                mcs.current_target_pos[1] if mcs.current_target_pos else mcs.pos[1],
            ) / 1000.0
            remain_after = mcs.remain - mcs.charge_power_kwh - dist_m_to_charge * POWER_UNIT

            can_serve = (remain_after > min(ev.need_power, max_charge) and
                         ev.wait_time_steps + mcs.charge_time_remain_min <= MAX_WAIT_TIME_STEPS * 5)

            if can_serve:
                charge_pos = mcs.current_target_pos if mcs.current_target_pos else mcs.pos
                dist = euclidean_distance(
                    ev.pos[0], ev.pos[1], charge_pos[0], charge_pos[1]) / 1000.0
                if dist < 1e-6:
                    dist = 1e-6
                r_local += (remain_after / MCS_BATTERY_CAPACITY) / (dist + 1.0)

            # ── 竞争惩罚: 附近其他 IEV ──
            for other in getattr(mcs, 'near_iev', []):
                if other is not ev:
                    dist_o = euclidean_distance(
                        other.pos[0], other.pos[1], mcs.pos[0], mcs.pos[1]) / 1000.0
                    if dist_o < 1e-6:
                        dist_o = 1e-6
                    r_local -= 0.5 * (other.need_power / max(1.0, max_charge)) / (dist_o + 1.0)

        # ── 2. 等待惩罚 ──
        if ev.is_iev and not ev.is_charged:
            wait_ratio = ev.wait_time_steps / max(1, MAX_WAIT_TIME_STEPS)
            r_local -= 0.1 * (1.0 + wait_ratio)

        # ── 3. Credit Assignment ──
        if ev.is_charged and ev.charge_pos is not None:
            # 刚被匹配 / 充电进行中
            charge_ratio = min(ev.need_power, max_charge) / max(1.0, max_charge)
            r_local += 1.0 * charge_ratio

        if ev.fail_charge:
            r_local -= 0.5

        # ── 4. 全局奖励 ──
        r_global = 0.0
        if ev.is_charged and ev.charge_pos is not None:
            total = self.step_success_count + self.step_fail_count + 1
            r_global += self.step_success_count / total
            r_global += min(ev.need_power, max_charge) / max(1.0, max_charge)
        if ev.fail_charge:
            r_global -= min(ev.need_power, max_charge) / max(1.0, max_charge)

        # ── 混合 ──
        reward = self.global_weight * r_global + self.local_weight * np.tanh(r_local * self.local_scale)
        return float(reward)

    # ============================================================
    # 步级统计更新
    # ============================================================

    def update_step_stats(self, EVs: List[EV]):
        """在每个 step 的 mix_get_reward_n 之前调用, 统计本步充电成功/失败数。"""
        self.step_success_count = sum(1 for e in EVs if e.is_charged and e.charge_pos is not None)
        self.step_fail_count = sum(1 for e in EVs if e.fail_charge)
