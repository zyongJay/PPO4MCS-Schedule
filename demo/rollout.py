"""High/Low 两条互不串扰的 option rollout 与回报计算通道。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch


@dataclass
class OpenHighOption:
    option_id: int
    mcs_id: int
    high_state: np.ndarray
    high_mask: np.ndarray
    high_action: int
    high_log_prob: float
    critic_state: np.ndarray
    value: float
    recharge_matched: bool
    start_remain_kwh: float = 0.0
    start_serve_potential: float = 0.0
    task_started: bool = False
    consecutive_no_candidate_replans: int = 0
    discounted_reward: float = 0.0
    reward_discount: float = 1.0
    duration_steps: int = 0
    boundary_reward: float = 0.0
    termination_reward: float = 0.0
    recharge_option_reward: float = 0.0
    recharge_risk_delta: float = 0.0
    option_event_reward: float = 0.0
    delayed_option_event_reward: float = 0.0


@dataclass
class HighOptionTransition:
    option_id: int
    mcs_id: int
    high_state: np.ndarray
    high_mask: np.ndarray
    high_action: int
    high_log_prob: float
    critic_state: np.ndarray
    value: float
    option_reward: float
    duration_steps: int
    next_critic_state: np.ndarray
    next_value: float
    terminal: bool
    truncated: bool
    recharge_matched: bool
    termination_reason: str
    start_remain_kwh: float = 0.0
    start_serve_potential: float = 0.0
    end_remain_kwh: float = 0.0
    boundary_reward: float = 0.0
    termination_reward: float = 0.0
    recharge_option_reward: float = 0.0
    recharge_risk_delta: float = 0.0
    option_event_reward: float = 0.0
    delayed_option_event_reward: float = 0.0
    advantage: float = 0.0
    return_target: float = 0.0


class HighOptionRolloutBuffer:
    """保存全部 High option，并计算 duration-aware SMDP-GAE。"""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.open_options: Dict[int, OpenHighOption] = {}
        self.transitions: List[HighOptionTransition] = []
        self._transition_by_option_id: Dict[int, HighOptionTransition] = {}
        self.delayed_event_count = 0
        self._returns_ready = False

    def start_option(
        self,
        option_id: int,
        mcs_id: int,
        observation: Dict,
        action: Dict,
        critic_state: np.ndarray,
        value: float,
        recharge_matched: bool,
        start_remain_kwh: float | None = None,
        start_serve_potential: float = 0.0,
    ) -> None:
        if mcs_id in self.open_options:
            raise RuntimeError(f'MCS {mcs_id} 已有未关闭的 High option')
        self._returns_ready = False
        self.open_options[mcs_id] = OpenHighOption(
            option_id=int(option_id),
            mcs_id=int(mcs_id),
            high_state=np.asarray(observation['high_state'], dtype=np.float32).copy(),
            high_mask=np.asarray(observation['high_action_mask'], dtype=bool).copy(),
            high_action=int(action['high_action']),
            high_log_prob=float(action['high_log_prob']),
            critic_state=np.asarray(critic_state, dtype=np.float32).copy(),
            value=float(value),
            recharge_matched=bool(recharge_matched),
            start_remain_kwh=float(
                0.0 if start_remain_kwh is None else start_remain_kwh
            ),
            start_serve_potential=float(np.clip(
                start_serve_potential, 0.0, 1.0
            )),
        )

    def mode(self, mcs_id: int) -> str:
        option = self.open_options[int(mcs_id)]
        return 'Serve' if option.high_action == 0 else 'Recharge'

    def option_id(self, mcs_id: int) -> int:
        return int(self.open_options[int(mcs_id)].option_id)

    def record_low_replan(
        self, mcs_id: int, has_quasi_candidate: bool
    ) -> None:
        option = self.open_options[int(mcs_id)]
        if option.high_action != 0:
            raise RuntimeError('Recharge option 不能记录 Low 重规划')
        if has_quasi_candidate:
            option.consecutive_no_candidate_replans = 0
        else:
            option.consecutive_no_candidate_replans += 1

    def mark_task_started(self, mcs_ids: Iterable[int]) -> None:
        for mcs_id in mcs_ids:
            option = self.open_options.get(int(mcs_id))
            if option is not None and option.high_action == 0:
                option.task_started = True

    def mark_recharge_matched(self, mcs_ids: Iterable[int]) -> None:
        for mcs_id in mcs_ids:
            option = self.open_options.get(int(mcs_id))
            if option is not None and option.high_action == 1:
                option.recharge_matched = True

    def add_step_rewards(self, reward_by_mcs_id: Dict[int, float]) -> None:
        self._returns_ready = False
        for mcs_id, option in self.open_options.items():
            reward = float(reward_by_mcs_id.get(mcs_id, 0.0))
            option.discounted_reward += option.reward_discount * reward
            option.reward_discount *= self.gamma
            option.duration_steps += 1

    def add_option_event_rewards(
        self, reward_by_option_id: Dict[int, float]
    ) -> None:
        """按稳定 option_id 写入成功事件，支持关闭后的延迟回写。"""
        self._returns_ready = False
        open_by_option_id = {
            option.option_id: option for option in self.open_options.values()
        }
        for raw_option_id, raw_reward in reward_by_option_id.items():
            option_id = int(raw_option_id)
            reward = float(raw_reward)
            option = open_by_option_id.get(option_id)
            if option is not None:
                option.discounted_reward += option.reward_discount * reward
                option.option_event_reward += reward
                continue
            transition = self._transition_by_option_id.get(option_id)
            if transition is None:
                raise RuntimeError(
                    f'High reward 无对应 option_id={option_id}'
                )
            transition.option_reward += (
                self.gamma ** transition.duration_steps
            ) * reward
            transition.option_event_reward += reward
            transition.delayed_option_event_reward += reward
            self.delayed_event_count += 1

    def add_boundary_rewards(
        self,
        reward_by_mcs_id: Dict[int, float],
        termination_reward_by_mcs_id: Dict[int, float] | None = None,
        recharge_reward_by_mcs_id: Dict[int, float] | None = None,
        recharge_risk_delta_by_mcs_id: Dict[int, float] | None = None,
    ) -> None:
        """在 Option 边界按 ``gamma ** duration`` 追加一次性奖励。"""
        self._returns_ready = False
        termination_rewards = termination_reward_by_mcs_id or {}
        recharge_rewards = recharge_reward_by_mcs_id or {}
        recharge_risk_deltas = recharge_risk_delta_by_mcs_id or {}
        for raw_mcs_id, raw_reward in reward_by_mcs_id.items():
            mcs_id = int(raw_mcs_id)
            option = self.open_options.get(mcs_id)
            if option is None:
                raise RuntimeError(f'MCS {mcs_id} 没有可写入边界奖励的 High option')
            reward = float(raw_reward)
            option.discounted_reward += option.reward_discount * reward
            option.boundary_reward += reward
            option.termination_reward += float(
                termination_rewards.get(mcs_id, 0.0)
            )
            option.recharge_option_reward += float(
                recharge_rewards.get(mcs_id, 0.0)
            )
            option.recharge_risk_delta += float(
                recharge_risk_deltas.get(mcs_id, 0.0)
            )

    def close_options(
        self,
        decision_ready_ids: Iterable[int],
        next_critic_states: Dict[int, np.ndarray],
        next_values: Dict[int, float],
        terminal_ids: Iterable[int] = (),
        termination_reasons: Dict[int, str] | None = None,
        end_remain_by_mcs_id: Dict[int, float] | None = None,
        close_all: bool = False,
        truncated: bool = False,
    ) -> int:
        self._returns_ready = False
        ready = {int(value) for value in decision_ready_ids}
        terminal = {int(value) for value in terminal_ids}
        reasons = {
            int(key): str(value)
            for key, value in (termination_reasons or {}).items()
        }
        end_remains = {
            int(key): float(value)
            for key, value in (end_remain_by_mcs_id or {}).items()
        }
        close_ids = [
            mcs_id for mcs_id in self.open_options
            if close_all or mcs_id in ready or mcs_id in terminal
        ]
        for mcs_id in close_ids:
            option = self.open_options.pop(mcs_id)
            is_terminal = mcs_id in terminal or (close_all and not truncated)
            if is_terminal:
                bootstrap_value = 0.0
                bootstrap_state = np.zeros_like(option.critic_state)
            elif mcs_id in next_critic_states and mcs_id in next_values:
                bootstrap_value = float(next_values[mcs_id])
                bootstrap_state = next_critic_states[mcs_id]
            else:
                if mcs_id not in next_critic_states or mcs_id not in next_values:
                    raise RuntimeError(f'MCS {mcs_id} 缺少 High bootstrap')
                bootstrap_value = float(next_values[mcs_id])
                bootstrap_state = next_critic_states[mcs_id]
            transition = HighOptionTransition(
                option_id=option.option_id,
                mcs_id=option.mcs_id,
                high_state=option.high_state,
                high_mask=option.high_mask,
                high_action=option.high_action,
                high_log_prob=option.high_log_prob,
                critic_state=option.critic_state,
                value=option.value,
                option_reward=option.discounted_reward,
                duration_steps=max(option.duration_steps, 1),
                next_critic_state=np.asarray(bootstrap_state, dtype=np.float32).copy(),
                next_value=bootstrap_value,
                terminal=is_terminal,
                truncated=bool(close_all and truncated),
                recharge_matched=option.recharge_matched,
                termination_reason=reasons.get(
                    mcs_id,
                    'truncated' if close_all and truncated else (
                        'terminal' if is_terminal else 'option_boundary'
                    ),
                ),
                start_remain_kwh=option.start_remain_kwh,
                start_serve_potential=option.start_serve_potential,
                end_remain_kwh=end_remains.get(
                    mcs_id, option.start_remain_kwh
                ),
                boundary_reward=option.boundary_reward,
                termination_reward=option.termination_reward,
                recharge_option_reward=option.recharge_option_reward,
                recharge_risk_delta=option.recharge_risk_delta,
                option_event_reward=option.option_event_reward,
                delayed_option_event_reward=(
                    option.delayed_option_event_reward
                ),
            )
            self.transitions.append(transition)
            self._transition_by_option_id[option.option_id] = transition
        return len(close_ids)

    def compute_returns_and_advantages(self) -> None:
        running_advantage: Dict[int, float] = {}
        for transition in reversed(self.transitions):
            continuation = 0.0 if transition.terminal else 1.0
            option_discount = self.gamma ** transition.duration_steps
            delta = (
                transition.option_reward
                + continuation * option_discount * transition.next_value
                - transition.value
            )
            next_advantage = running_advantage.get(transition.mcs_id, 0.0)
            transition.advantage = float(
                delta
                + continuation * option_discount * self.gae_lambda
                * next_advantage
            )
            transition.return_target = transition.advantage + transition.value
            running_advantage[transition.mcs_id] = transition.advantage
        self._returns_ready = True

    def extend_completed(self, other: 'HighOptionRolloutBuffer') -> None:
        if other.open_options:
            raise RuntimeError('不能合并仍有 open High option 的 rollout')
        if not other._returns_ready:
            other.compute_returns_and_advantages()
        self.transitions.extend(other.transitions)
        self._transition_by_option_id.update(
            other._transition_by_option_id
        )
        self.delayed_event_count += other.delayed_event_count
        self._returns_ready = True

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if not self.transitions:
            raise RuntimeError('High rollout 为空')
        if not self._returns_ready:
            self.compute_returns_and_advantages()
        arrays = {
            'high_states': np.stack([x.high_state for x in self.transitions]),
            'high_masks': np.stack([x.high_mask for x in self.transitions]),
            'high_actions': np.asarray([x.high_action for x in self.transitions]),
            'old_high_log_probs': np.asarray([x.high_log_prob for x in self.transitions]),
            'critic_states': np.stack([x.critic_state for x in self.transitions]),
            'advantages': np.asarray([x.advantage for x in self.transitions]),
            'returns': np.asarray([x.return_target for x in self.transitions]),
            'durations': np.asarray([x.duration_steps for x in self.transitions]),
            'recharge_matched': np.asarray([x.recharge_matched for x in self.transitions]),
        }
        integer = {'high_actions', 'durations'}
        boolean = {'high_masks', 'recharge_matched'}
        return {
            key: torch.as_tensor(
                value,
                dtype=(torch.long if key in integer else torch.bool if key in boolean else torch.float32),
                device=device,
            )
            for key, value in arrays.items()
        }

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class OpenLowServeOption:
    decision_id: int
    serve_option_id: int
    mcs_id: int
    low_self_state: np.ndarray
    low_candidates: np.ndarray
    low_mask: np.ndarray
    low_action: int
    selected_candidate_id: int
    stay_selected: bool
    forced_stay: bool
    raw_quasi_count: int
    safe_quasi_count: int
    topk_truncated_count: int
    selected_urgency_rank: int
    selected_urgency: float
    selected_remain_kwh: float
    selected_need_power_kwh: float
    selected_distance_ratio: float
    selected_attraction: float
    selected_immediate_iev_attraction: float
    selected_mcs_competition: float
    selected_fcs_competition: float
    selected_desirability: float
    available_avg_attraction: float
    available_avg_immediate_iev_attraction: float
    available_avg_mcs_competition: float
    available_avg_fcs_competition: float
    available_avg_desirability: float
    low_log_prob: float
    global_state: np.ndarray
    value: float
    discounted_reward: float = 0.0
    reward_discount: float = 1.0
    duration_steps: int = 0


@dataclass
class LowServeTransition:
    decision_id: int
    serve_option_id: int
    mcs_id: int
    low_self_state: np.ndarray
    low_candidates: np.ndarray
    low_mask: np.ndarray
    low_action: int
    selected_candidate_id: int
    stay_selected: bool
    forced_stay: bool
    raw_quasi_count: int
    safe_quasi_count: int
    topk_truncated_count: int
    selected_urgency_rank: int
    selected_urgency: float
    selected_remain_kwh: float
    selected_need_power_kwh: float
    selected_distance_ratio: float
    selected_attraction: float
    selected_immediate_iev_attraction: float
    selected_mcs_competition: float
    selected_fcs_competition: float
    selected_desirability: float
    available_avg_attraction: float
    available_avg_immediate_iev_attraction: float
    available_avg_mcs_competition: float
    available_avg_fcs_competition: float
    available_avg_desirability: float
    low_log_prob: float
    global_state: np.ndarray
    value: float
    option_reward: float
    duration_steps: int
    next_value: float
    terminal: bool
    truncated: bool
    termination_reason: str = 'low_replan'
    bootstrap_resolved: bool = False
    advantage: float = 0.0
    return_target: float = 0.0
    delayed_reward: float = 0.0


class LowServeRolloutBuffer:
    """保存 Low Serve option，并按实际 Low 决策间隔计算 SMDP-GAE。

    一个 Low option 在 MCS 再次可决策时关闭，并只向同一
    High Serve option 内的下一次 Low 决策 bootstrap。High Serve
    正常/异常终止时同步切断 Low GAE 链，不再跨 Recharge 传播。
    时间上限截断不属于业务终止，最后 transition 使用截断
    状态的真实 Low value bootstrap。
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.open_options: Dict[int, OpenLowServeOption] = {}
        self.open_decision_by_mcs: Dict[int, int] = {}
        self.pending_transition_by_mcs: Dict[int, LowServeTransition] = {}
        self.transitions: List[LowServeTransition] = []
        self._transition_by_decision: Dict[int, LowServeTransition] = {}
        self._returns_ready = False
        self.delayed_event_count = 0

    def start_option(
        self,
        decision_id: int,
        serve_option_id: int,
        mcs_id: int,
        observation: Dict,
        action: Dict,
        global_state: np.ndarray,
        value: float,
    ) -> None:
        decision_id = int(decision_id)
        serve_option_id = int(serve_option_id)
        mcs_id = int(mcs_id)
        if decision_id in self.open_options or decision_id in self._transition_by_decision:
            raise RuntimeError(f'Low decision_id={decision_id} 重复')
        if mcs_id in self.open_decision_by_mcs:
            raise RuntimeError(f'MCS {mcs_id} 已有未关闭的 Low option')
        if int(action['low_action']) < 0:
            raise ValueError('只有 Serve 动作可以创建 Low option')
        pending = self.pending_transition_by_mcs.pop(mcs_id, None)
        if pending is not None:
            if pending.serve_option_id != serve_option_id:
                raise RuntimeError(
                    'Low bootstrap 跨越了 High Serve option 边界'
                )
            pending.next_value = float(value)
            pending.bootstrap_resolved = True
        low_action = int(action['low_action'])
        selected_candidate_id = int(
            observation['candidate_ids'][low_action]
        )
        # -2 是 ObservationBuilder 约定的MCS当前位置固定候选；这里按ID
        # 判断以兼容尚未携带candidate_is_stay字段的旧合成测试。
        stay_selected = selected_candidate_id == -2
        low_candidates = np.asarray(
            observation['low_candidates'], dtype=np.float32
        )
        low_mask = np.asarray(
            observation['low_candidate_mask'], dtype=bool
        )
        candidate_ids = np.asarray(
            observation['candidate_ids'], dtype=np.int64
        )
        quasi_indices = np.flatnonzero(
            low_mask & (candidate_ids >= 0)
        )
        candidate_urgencies = np.asarray(
            observation.get(
                'candidate_urgencies', np.zeros_like(candidate_ids, dtype=float)
            ),
            dtype=np.float32,
        )
        candidate_desirabilities = np.asarray(
            observation.get(
                'candidate_desirabilities',
                np.zeros_like(candidate_ids, dtype=float),
            ),
            dtype=np.float32,
        )
        candidate_remain = np.asarray(
            observation.get(
                'candidate_remain_kwh',
                np.full_like(candidate_ids, -1, dtype=float),
            ),
            dtype=np.float32,
        )
        candidate_need_power = np.asarray(
            observation.get(
                'candidate_need_power_kwh',
                np.full_like(candidate_ids, -1, dtype=float),
            ),
            dtype=np.float32,
        )

        def candidate_mean(column: np.ndarray) -> float:
            if quasi_indices.size == 0:
                return 0.0
            return float(np.mean(column[quasi_indices]))

        self._returns_ready = False
        self.open_options[decision_id] = OpenLowServeOption(
            decision_id=decision_id,
            serve_option_id=serve_option_id,
            mcs_id=mcs_id,
            low_self_state=np.asarray(observation['low_self_state'], dtype=np.float32).copy(),
            low_candidates=low_candidates.copy(),
            low_mask=low_mask.copy(),
            low_action=low_action,
            selected_candidate_id=selected_candidate_id,
            stay_selected=stay_selected,
            forced_stay=bool(
                stay_selected
                and int(observation.get('quasi_candidate_count', 0)) == 0
            ),
            raw_quasi_count=int(observation.get('raw_quasi_count', 0)),
            safe_quasi_count=int(observation.get('safe_quasi_count', 0)),
            topk_truncated_count=int(
                observation.get('topk_truncated_count', 0)
            ),
            selected_urgency_rank=(0 if stay_selected else low_action),
            selected_urgency=float(candidate_urgencies[low_action]),
            selected_remain_kwh=float(candidate_remain[low_action]),
            selected_need_power_kwh=float(
                candidate_need_power[low_action]
            ),
            selected_distance_ratio=float(low_candidates[low_action, 1]),
            selected_attraction=float(low_candidates[low_action, 2]),
            selected_immediate_iev_attraction=float(
                low_candidates[low_action, 3]
            ),
            selected_mcs_competition=float(
                low_candidates[low_action, 4]
            ),
            selected_fcs_competition=float(
                low_candidates[low_action, 5]
            ),
            selected_desirability=float(
                candidate_desirabilities[low_action]
            ),
            available_avg_attraction=candidate_mean(low_candidates[:, 2]),
            available_avg_immediate_iev_attraction=candidate_mean(
                low_candidates[:, 3]
            ),
            available_avg_mcs_competition=candidate_mean(
                low_candidates[:, 4]
            ),
            available_avg_fcs_competition=candidate_mean(
                low_candidates[:, 5]
            ),
            available_avg_desirability=candidate_mean(
                candidate_desirabilities
            ),
            low_log_prob=float(action['low_log_prob']),
            global_state=np.asarray(global_state, dtype=np.float32).copy(),
            value=float(value),
        )
        self.open_decision_by_mcs[mcs_id] = decision_id

    def pending_mcs_ids(self) -> set[int]:
        """返回正在等待下一次真实 Low 决策的 MCS。"""
        return set(self.pending_transition_by_mcs)

    def bootstrap_mcs_ids(self) -> set[int]:
        """返回截断时需要真实 Low value 的 MCS。"""
        return (
            set(self.open_decision_by_mcs)
            | set(self.pending_transition_by_mcs)
        )

    def advance_pending_steps(self, mcs_ids: Iterable[int]) -> None:
        """让未产生新 Low 决策的 transition 跨过一个零 Low-reward 步。"""
        self._returns_ready = False
        for mcs_id in mcs_ids:
            transition = self.pending_transition_by_mcs.get(int(mcs_id))
            if transition is not None:
                transition.duration_steps += 1

    def add_step_rewards(self, reward_by_decision_id: Dict[int, float]) -> None:
        self._returns_ready = False
        # 每个 open Serve option 均经历一个底层环境 step，即使本 step 奖励为 0。
        for option in self.open_options.values():
            reward = float(reward_by_decision_id.get(option.decision_id, 0.0))
            option.discounted_reward += option.reward_discount * reward
            option.reward_discount *= self.gamma
            option.duration_steps += 1
        # 若环境以后把事件推迟到 option 关闭后，仍按唯一 decision_id 补记。
        for decision_id, reward in reward_by_decision_id.items():
            decision_id = int(decision_id)
            if decision_id in self.open_options:
                continue
            transition = self._transition_by_decision.get(decision_id)
            if transition is None:
                raise RuntimeError(f'Low reward 无对应 decision_id={decision_id}')
            delayed = (self.gamma ** transition.duration_steps) * float(reward)
            transition.option_reward += delayed
            transition.delayed_reward += delayed
            self.delayed_event_count += 1

    def close_options(
        self,
        decision_ready_ids: Iterable[int],
        next_values: Dict[int, float] | None = None,
        terminal_ids: Iterable[int] = (),
        close_all: bool = False,
        truncated: bool = False,
    ) -> int:
        self._returns_ready = False
        ready = {int(value) for value in decision_ready_ids}
        bootstrap_values = {
            int(key): float(value)
            for key, value in (next_values or {}).items()
        }
        terminal = {int(value) for value in terminal_ids}
        close_decisions = [
            decision_id for decision_id, option in self.open_options.items()
            if close_all or option.mcs_id in ready or option.mcs_id in terminal
        ]
        for decision_id in close_decisions:
            option = self.open_options.pop(decision_id)
            self.open_decision_by_mcs.pop(option.mcs_id, None)
            is_terminal = bool(
                option.mcs_id in terminal
                or (close_all and not truncated)
            )
            transition = LowServeTransition(
                decision_id=option.decision_id,
                serve_option_id=option.serve_option_id,
                mcs_id=option.mcs_id,
                low_self_state=option.low_self_state,
                low_candidates=option.low_candidates,
                low_mask=option.low_mask,
                low_action=option.low_action,
                selected_candidate_id=option.selected_candidate_id,
                stay_selected=option.stay_selected,
                forced_stay=option.forced_stay,
                raw_quasi_count=option.raw_quasi_count,
                safe_quasi_count=option.safe_quasi_count,
                topk_truncated_count=option.topk_truncated_count,
                selected_urgency_rank=option.selected_urgency_rank,
                selected_urgency=option.selected_urgency,
                selected_remain_kwh=option.selected_remain_kwh,
                selected_need_power_kwh=option.selected_need_power_kwh,
                selected_distance_ratio=option.selected_distance_ratio,
                selected_attraction=option.selected_attraction,
                selected_immediate_iev_attraction=(
                    option.selected_immediate_iev_attraction
                ),
                selected_mcs_competition=option.selected_mcs_competition,
                selected_fcs_competition=option.selected_fcs_competition,
                selected_desirability=option.selected_desirability,
                available_avg_attraction=option.available_avg_attraction,
                available_avg_immediate_iev_attraction=(
                    option.available_avg_immediate_iev_attraction
                ),
                available_avg_mcs_competition=(
                    option.available_avg_mcs_competition
                ),
                available_avg_fcs_competition=(
                    option.available_avg_fcs_competition
                ),
                available_avg_desirability=(
                    option.available_avg_desirability
                ),
                low_log_prob=option.low_log_prob,
                global_state=option.global_state,
                value=option.value,
                option_reward=option.discounted_reward,
                duration_steps=max(option.duration_steps, 1),
                next_value=(
                    bootstrap_values[option.mcs_id]
                    if close_all and truncated
                    and option.mcs_id not in terminal
                    and option.mcs_id in bootstrap_values
                    else 0.0
                ),
                terminal=is_terminal,
                truncated=bool(close_all and truncated),
                termination_reason=(
                    'truncated' if close_all and truncated else (
                        'terminal' if is_terminal else 'low_replan'
                    )
                ),
                bootstrap_resolved=bool(is_terminal or close_all),
            )
            if (
                close_all and truncated
                and option.mcs_id not in terminal
                and option.mcs_id not in bootstrap_values
            ):
                raise RuntimeError(
                    f'MCS {option.mcs_id} 缺少截断 Low bootstrap'
                )
            self.transitions.append(transition)
            self._transition_by_decision[decision_id] = transition
            if not transition.bootstrap_resolved:
                if option.mcs_id in self.pending_transition_by_mcs:
                    raise RuntimeError(
                        f'MCS {option.mcs_id} 已有待bootstrap的Low transition'
                    )
                self.pending_transition_by_mcs[option.mcs_id] = transition

        finalize_pending_ids = [
            mcs_id for mcs_id in self.pending_transition_by_mcs
            if close_all or mcs_id in terminal
        ]
        for mcs_id in finalize_pending_ids:
            transition = self.pending_transition_by_mcs[mcs_id]
            if close_all and truncated and mcs_id not in terminal:
                if mcs_id not in bootstrap_values:
                    raise RuntimeError(
                        f'MCS {mcs_id} 缺少截断 Low bootstrap'
                    )
                transition.next_value = bootstrap_values[mcs_id]
                transition.termination_reason = 'truncated'
            else:
                transition.next_value = 0.0
                transition.termination_reason = 'terminal'
            transition.terminal = bool(
                mcs_id in terminal or (close_all and not truncated)
            )
            transition.truncated = bool(close_all and truncated)
            transition.bootstrap_resolved = True
            self.pending_transition_by_mcs.pop(mcs_id)
        return len(close_decisions)

    def terminate_serve_options(
        self,
        serve_option_ids: Iterable[int],
        reason: str,
    ) -> int:
        """在 High Serve 边界上终止对应 Low GAE 链。"""
        target_ids = {int(value) for value in serve_option_ids}
        if not target_ids:
            return 0
        self._returns_ready = False
        affected = 0
        for transition in self.transitions:
            if (
                transition.serve_option_id in target_ids
                and not transition.bootstrap_resolved
            ):
                transition.next_value = 0.0
                transition.terminal = True
                transition.truncated = False
                transition.termination_reason = str(reason)
                transition.bootstrap_resolved = True
                self.pending_transition_by_mcs.pop(
                    transition.mcs_id, None
                )
                affected += 1
        dangling = [
            item.decision_id for item in self.open_options.values()
            if item.serve_option_id in target_ids
        ]
        if dangling:
            raise RuntimeError(
                f'High Serve 终止时仍有未关闭 Low option: {dangling[:10]}'
            )
        return affected

    def compute_returns_and_advantages(self) -> None:
        if self.open_options:
            raise RuntimeError('存在未关闭的Low option，不能计算GAE')
        if self.pending_transition_by_mcs:
            raise RuntimeError('存在未完成bootstrap的Low transition')
        unresolved = [
            item.decision_id for item in self.transitions
            if not item.bootstrap_resolved
        ]
        if unresolved:
            raise RuntimeError(
                f'Low transition缺少bootstrap: {unresolved[:10]}'
            )

        running_advantage: Dict[tuple[int, int], float] = {}
        for transition in reversed(self.transitions):
            continuation = 0.0 if transition.terminal else 1.0
            option_discount = self.gamma ** transition.duration_steps
            delta = (
                transition.option_reward
                + continuation * option_discount * transition.next_value
                - transition.value
            )
            chain_key = (
                transition.mcs_id, transition.serve_option_id
            )
            next_advantage = running_advantage.get(chain_key, 0.0)
            transition.advantage = float(
                delta
                + continuation * option_discount * self.gae_lambda
                * next_advantage
            )
            transition.return_target = transition.advantage + transition.value
            running_advantage[chain_key] = transition.advantage
        self._returns_ready = True

    def extend_completed(self, other: 'LowServeRolloutBuffer') -> None:
        if other.open_options:
            raise RuntimeError('不能合并仍有 open Low option 的 rollout')
        if other.pending_transition_by_mcs:
            raise RuntimeError('不能合并仍待bootstrap的Low rollout')
        if not other._returns_ready:
            other.compute_returns_and_advantages()
        for transition in other.transitions:
            if transition.decision_id in self._transition_by_decision:
                raise RuntimeError('合并 Low rollout 时 decision_id 冲突')
            self.transitions.append(transition)
            self._transition_by_decision[transition.decision_id] = transition
        self.delayed_event_count += other.delayed_event_count
        self._returns_ready = True

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if not self.transitions:
            raise RuntimeError('Low rollout 为空')
        if not self._returns_ready:
            self.compute_returns_and_advantages()
        arrays = {
            'decision_ids': np.asarray([x.decision_id for x in self.transitions]),
            'low_self_states': np.stack([x.low_self_state for x in self.transitions]),
            'low_candidates': np.stack([x.low_candidates for x in self.transitions]),
            'low_masks': np.stack([x.low_mask for x in self.transitions]),
            'low_actions': np.asarray([x.low_action for x in self.transitions]),
            'old_low_log_probs': np.asarray([x.low_log_prob for x in self.transitions]),
            'global_states': np.stack([x.global_state for x in self.transitions]),
            'advantages': np.asarray([x.advantage for x in self.transitions]),
            'returns': np.asarray([x.return_target for x in self.transitions]),
            'durations': np.asarray([x.duration_steps for x in self.transitions]),
        }
        integer = {'decision_ids', 'low_actions', 'durations'}
        boolean = {'low_masks'}
        return {
            key: torch.as_tensor(
                value,
                dtype=(torch.long if key in integer else torch.bool if key in boolean else torch.float32),
                device=device,
            )
            for key, value in arrays.items()
        }

    def __len__(self) -> int:
        return len(self.transitions)


# 兼容仍引用旧类名的外部代码；新训练流程只使用显式 High/Low 类名。
OptionAwareRolloutBuffer = HighOptionRolloutBuffer
