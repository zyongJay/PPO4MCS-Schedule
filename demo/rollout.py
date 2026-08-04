"""Semi-Markov rollout storage for MCS High/Low options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch


@dataclass
class OpenMCSOption:
    mcs_id: int
    high_state: np.ndarray
    high_mask: np.ndarray
    high_action: int
    high_log_prob: float
    low_self_state: np.ndarray
    low_candidates: np.ndarray
    low_mask: np.ndarray
    low_action: int
    low_log_prob: float
    critic_state: np.ndarray
    value: float
    recharge_matched: bool
    discounted_reward: float = 0.0
    reward_discount: float = 1.0
    duration_steps: int = 0


@dataclass
class MCSOptionTransition:
    mcs_id: int
    high_state: np.ndarray
    high_mask: np.ndarray
    high_action: int
    high_log_prob: float
    low_self_state: np.ndarray
    low_candidates: np.ndarray
    low_mask: np.ndarray
    low_action: int
    low_log_prob: float
    critic_state: np.ndarray
    value: float
    option_reward: float
    duration_steps: int
    next_critic_state: np.ndarray
    next_value: float
    terminal: bool
    recharge_matched: bool
    advantage: float = 0.0
    return_target: float = 0.0


class OptionAwareRolloutBuffer:
    """Store one transition per High decision, not one per world step.

    An option remains open while its MCS is absent from the High decision set.
    Rewards from every underlying world step are discounted into the option
    return.  The transition closes when that MCS can make another High decision,
    becomes terminal, or the episode is truncated.
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.open_options: Dict[int, OpenMCSOption] = {}
        self.transitions: List[MCSOptionTransition] = []
        self._returns_ready = False

    def start_option(
        self,
        mcs_id: int,
        observation: Dict,
        action: Dict,
        critic_state: np.ndarray,
        value: float,
        recharge_matched: bool,
    ) -> None:
        if mcs_id in self.open_options:
            raise RuntimeError(f'MCS {mcs_id} already has an open option')
        self._returns_ready = False
        self.open_options[mcs_id] = OpenMCSOption(
            mcs_id=int(mcs_id),
            high_state=np.asarray(observation['high_state'], dtype=np.float32).copy(),
            high_mask=np.asarray(observation['high_action_mask'], dtype=bool).copy(),
            high_action=int(action['high_action']),
            high_log_prob=float(action['high_log_prob']),
            low_self_state=np.asarray(
                observation['low_self_state'], dtype=np.float32
            ).copy(),
            low_candidates=np.asarray(
                observation['low_candidates'], dtype=np.float32
            ).copy(),
            low_mask=np.asarray(
                observation['low_candidate_mask'], dtype=bool
            ).copy(),
            low_action=int(action['low_action']),
            low_log_prob=float(action['low_log_prob']),
            critic_state=np.asarray(critic_state, dtype=np.float32).copy(),
            value=float(value),
            recharge_matched=bool(recharge_matched),
        )

    def add_step_rewards(self, reward_by_mcs_id: Dict[int, float]) -> None:
        self._returns_ready = False
        for mcs_id, option in self.open_options.items():
            reward = float(reward_by_mcs_id.get(mcs_id, 0.0))
            option.discounted_reward += option.reward_discount * reward
            option.reward_discount *= self.gamma
            option.duration_steps += 1

    def close_options(
        self,
        decision_ready_ids: Iterable[int],
        next_critic_states: Dict[int, np.ndarray],
        next_values: Dict[int, float],
        terminal_ids: Iterable[int] = (),
        close_all: bool = False,
    ) -> int:
        self._returns_ready = False
        ready = set(int(value) for value in decision_ready_ids)
        terminal = set(int(value) for value in terminal_ids)
        close_ids = []
        for mcs_id in self.open_options:
            if close_all or mcs_id in ready or mcs_id in terminal:
                close_ids.append(mcs_id)

        for mcs_id in close_ids:
            option = self.open_options.pop(mcs_id)
            is_terminal = close_all or mcs_id in terminal
            if is_terminal:
                bootstrap_value = 0.0
                bootstrap_state = np.zeros_like(option.critic_state)
            else:
                if mcs_id not in next_critic_states or mcs_id not in next_values:
                    raise RuntimeError(
                        f'missing bootstrap state/value for MCS {mcs_id}'
                    )
                bootstrap_value = float(next_values[mcs_id])
                bootstrap_state = next_critic_states[mcs_id]
            self.transitions.append(MCSOptionTransition(
                mcs_id=option.mcs_id,
                high_state=option.high_state,
                high_mask=option.high_mask,
                high_action=option.high_action,
                high_log_prob=option.high_log_prob,
                low_self_state=option.low_self_state,
                low_candidates=option.low_candidates,
                low_mask=option.low_mask,
                low_action=option.low_action,
                low_log_prob=option.low_log_prob,
                critic_state=option.critic_state,
                value=option.value,
                option_reward=option.discounted_reward,
                duration_steps=max(option.duration_steps, 1),
                next_critic_state=np.asarray(
                    bootstrap_state, dtype=np.float32
                ).copy(),
                next_value=bootstrap_value,
                terminal=is_terminal,
                recharge_matched=option.recharge_matched,
            ))
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
                + continuation
                * option_discount
                * self.gae_lambda
                * next_advantage
            )
            transition.return_target = transition.advantage + transition.value
            running_advantage[transition.mcs_id] = transition.advantage
        self._returns_ready = True

    def extend_completed(self, other: 'OptionAwareRolloutBuffer') -> None:
        """Merge a finished trajectory without linking GAE across scenarios."""
        if other.open_options:
            raise RuntimeError('cannot merge a rollout with open options')
        if not other._returns_ready:
            other.compute_returns_and_advantages()
        self.transitions.extend(other.transitions)
        self._returns_ready = True

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if not self.transitions:
            raise RuntimeError('cannot tensorize an empty rollout')
        if not self._returns_ready:
            self.compute_returns_and_advantages()
        data = {
            'high_states': np.stack([x.high_state for x in self.transitions]),
            'high_masks': np.stack([x.high_mask for x in self.transitions]),
            'high_actions': np.asarray([x.high_action for x in self.transitions]),
            'old_high_log_probs': np.asarray([x.high_log_prob for x in self.transitions]),
            'low_self_states': np.stack([x.low_self_state for x in self.transitions]),
            'low_candidates': np.stack([x.low_candidates for x in self.transitions]),
            'low_masks': np.stack([x.low_mask for x in self.transitions]),
            'low_actions': np.asarray([x.low_action for x in self.transitions]),
            'old_low_log_probs': np.asarray([x.low_log_prob for x in self.transitions]),
            'critic_states': np.stack([x.critic_state for x in self.transitions]),
            'advantages': np.asarray([x.advantage for x in self.transitions]),
            'returns': np.asarray([x.return_target for x in self.transitions]),
            'durations': np.asarray([x.duration_steps for x in self.transitions]),
            'recharge_matched': np.asarray(
                [x.recharge_matched for x in self.transitions]
            ),
        }
        tensors = {}
        integer_keys = {'high_actions', 'low_actions', 'durations'}
        boolean_keys = {'high_masks', 'low_masks', 'recharge_matched'}
        for key, value in data.items():
            dtype = torch.float32
            if key in integer_keys:
                dtype = torch.long
            elif key in boolean_keys:
                dtype = torch.bool
            tensors[key] = torch.as_tensor(value, dtype=dtype, device=device)
        return tensors

    def clear(self) -> None:
        self.open_options.clear()
        self.transitions.clear()
        self._returns_ready = False

    def __len__(self) -> int:
        return len(self.transitions)
