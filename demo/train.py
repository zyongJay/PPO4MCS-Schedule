"""Train the MCS High/Low actors with option-aware PPO.

IEV actions remain deterministic track-following actions.  The trainable policy
contains only the MCS High Actor (Serve/Recharge), the Serve-conditioned Low
Actor (quasi candidate or fixed current-position action), and two level-specific
centralized critics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import world as world_module
from config import (
    EV_LOW_POWER_THRESHOLD,
    HIDDEN_DIM,
    MAX_STEPS_PER_EPISODE,
    MCS_CRITIC_STATE_DIM,
    MCS_FEAT_DIM_self,
    MCS_FEAT_DIM_tgt,
    MCS_HIGH_FEAT_DIM,
    TOP_K_MCS_CANDIDATES,
    TRACK_DATA_PATH,
)
from core import EV, MCS
from environment import MultiAgentEnv
from matching import RechargeMatcher
from network import MCSMAPPOAgent
from observation import MCS_STAY_CANDIDATE_ID
from reward import (
    LOW_EVENT_REWARD_WEIGHT,
    LOW_FCS_ALTERNATIVE_PENALTY,
    LOW_MCS_ALTERNATIVE_PENALTY,
    LOW_SPATIAL_OPPORTUNITY_WEIGHT,
    LOW_STEP_REWARD_WEIGHT,
    SERVE_MOVE_PENALTY,
)
from rollout import HighOptionRolloutBuffer, LowServeRolloutBuffer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / 'training_results'
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


def critic_input(global_state: np.ndarray, high_state: np.ndarray) -> np.ndarray:
    value = np.concatenate((global_state, high_state)).astype(np.float32)
    if value.shape != (MCS_CRITIC_STATE_DIM,):
        raise RuntimeError(f'unexpected critic input shape: {value.shape}')
    return value


def iev_track_action(ev: EV) -> Dict:
    if ev.track and ev.track_index + 1 < len(ev.track):
        point = ev.track[ev.track_index + 1]
        target = [float(point[0]), float(point[1])]
    else:
        target = list(ev.pos)
    return {'target_pos': target}


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )


def explained_variance(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> float:
    """返回 1-var(error)/var(target)；目标近似常量时记为 0。"""
    target_var = torch.var(targets, unbiased=False)
    if float(target_var.item()) < 1e-12:
        return 0.0
    residual_var = torch.var(targets - predictions, unbiased=False)
    return float((1.0 - residual_var / target_var).item())


def ppo_update(
    agent: MCSMAPPOAgent,
    high_buffer: HighOptionRolloutBuffer,
    low_buffer: LowServeRolloutBuffer,
    update_epochs: int,
    high_minibatch_size: int,
    low_minibatch_size: int,
    clip_ratio: float,
    high_entropy_coef: float,
    low_entropy_coef: float,
    max_grad_norm: float,
    train_high: bool = True,
) -> Dict[str, float]:
    high_data = high_buffer.as_tensors(agent.device)
    high_advantages = normalize_advantages(high_data['advantages'])
    high_sample_count = high_advantages.shape[0]
    high_losses: List[float] = []
    low_losses: List[float] = []
    high_critic_losses: List[float] = []
    low_critic_losses: List[float] = []
    high_entropies: List[float] = []
    low_entropies: List[float] = []

    with torch.no_grad():
        high_ev = explained_variance(
            high_data['returns'],
            agent.high_values(high_data['critic_states']),
        )

    low_data = None
    low_advantages = None
    low_sample_count = len(low_buffer)
    low_ev = 0.0
    if low_sample_count:
        low_data = low_buffer.as_tensors(agent.device)
        low_advantages = normalize_advantages(low_data['advantages'])
        with torch.no_grad():
            low_ev = explained_variance(
                low_data['returns'],
                agent.low_values(
                    low_data['global_states'],
                    low_data['low_self_states'],
                    low_data['low_candidates'],
                    low_data['low_masks'],
                ),
            )

    for _ in range(update_epochs):
        if train_high:
            permutation = torch.randperm(
                high_sample_count, device=agent.device
            )
            for start in range(0, high_sample_count, high_minibatch_size):
                indices = permutation[start:start + high_minibatch_size]
                batch_advantages = high_advantages[indices]

                new_high_log_probs, high_entropy = agent.evaluate_high(
                    high_data['high_states'][indices],
                    high_data['high_masks'][indices],
                    high_data['high_actions'][indices],
                )
                high_ratio = torch.exp(
                    new_high_log_probs
                    - high_data['old_high_log_probs'][indices]
                )
                high_surrogate = high_ratio * batch_advantages
                high_clipped = torch.clamp(
                    high_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                ) * batch_advantages
                high_loss = -torch.min(
                    high_surrogate, high_clipped
                ).mean()
                high_objective = (
                    high_loss
                    - high_entropy_coef * high_entropy.mean()
                )
                agent.high_optimizer.zero_grad()
                high_objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.high_actor.parameters(), max_grad_norm
                )
                agent.high_optimizer.step()
                high_losses.append(float(high_loss.item()))
                high_entropies.append(float(high_entropy.mean().item()))

                predicted_values = agent.high_values(
                    high_data['critic_states'][indices]
                )
                high_critic_loss = 0.5 * F.mse_loss(
                    predicted_values, high_data['returns'][indices]
                )
                agent.high_critic_optimizer.zero_grad()
                high_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.high_critic.parameters(), max_grad_norm
                )
                agent.high_critic_optimizer.step()
                high_critic_losses.append(float(high_critic_loss.item()))

        if low_data is None or low_advantages is None:
            continue
        permutation = torch.randperm(low_sample_count, device=agent.device)
        for start in range(0, low_sample_count, low_minibatch_size):
            indices = permutation[start:start + low_minibatch_size]
            batch_advantages = low_advantages[indices]
            new_low_log_probs, low_entropy = agent.evaluate_low(
                low_data['low_self_states'][indices],
                low_data['low_candidates'][indices],
                low_data['low_masks'][indices],
                low_data['low_actions'][indices],
            )
            low_ratio = torch.exp(
                new_low_log_probs
                - low_data['old_low_log_probs'][indices]
            )
            low_surrogate = low_ratio * batch_advantages
            low_clipped = torch.clamp(
                low_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * batch_advantages
            low_loss = -torch.min(low_surrogate, low_clipped).mean()
            low_objective = (
                low_loss - low_entropy_coef * low_entropy.mean()
            )
            agent.low_optimizer.zero_grad()
            low_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.low_actor.parameters(), max_grad_norm
            )
            agent.low_optimizer.step()
            low_losses.append(float(low_loss.item()))
            low_entropies.append(float(low_entropy.mean().item()))

            predictions = agent.low_values(
                low_data['global_states'][indices],
                low_data['low_self_states'][indices],
                low_data['low_candidates'][indices],
                low_data['low_masks'][indices],
            )
            low_critic_loss = 0.5 * F.mse_loss(
                predictions, low_data['returns'][indices]
            )
            agent.low_critic_optimizer.zero_grad()
            low_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.low_critic.parameters(), max_grad_norm
            )
            agent.low_critic_optimizer.step()
            low_critic_losses.append(float(low_critic_loss.item()))

    high_mean = float(np.mean(high_losses)) if high_losses else 0.0
    low_mean = float(np.mean(low_losses)) if low_losses else 0.0
    return {
        'actor_loss': high_mean + low_mean,
        'high_actor_loss': high_mean,
        'low_actor_loss': low_mean,
        'critic_loss': (
            (float(np.mean(high_critic_losses)) if high_critic_losses else 0.0)
            + (float(np.mean(low_critic_losses)) if low_critic_losses else 0.0)
        ),
        'high_critic_loss': float(np.mean(high_critic_losses)) if high_critic_losses else 0.0,
        'low_critic_loss': float(np.mean(low_critic_losses)) if low_critic_losses else 0.0,
        'high_explained_variance': high_ev,
        'low_explained_variance': low_ev,
        'high_sample_count': int(high_sample_count),
        'low_sample_count': int(low_sample_count),
        'high_frozen': int(not train_high),
        'high_entropy': float(np.mean(high_entropies)) if high_entropies else 0.0,
        'low_entropy': float(np.mean(low_entropies)) if low_entropies else 0.0,
    }


def append_high_option_log(
    path: Path,
    episode: int,
    buffer: HighOptionRolloutBuffer,
) -> None:
    fieldnames = [
        'episode', 'mcs_id', 'high_action', 'duration_steps',
        'option_reward', 'advantage', 'return_target', 'value',
        'terminal', 'truncated', 'recharge_matched',
    ]
    write_header = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for item in buffer.transitions:
            writer.writerow({
                'episode': episode,
                'mcs_id': item.mcs_id,
                'high_action': item.high_action,
                'duration_steps': item.duration_steps,
                'option_reward': item.option_reward,
                'advantage': item.advantage,
                'return_target': item.return_target,
                'value': item.value,
                'terminal': item.terminal,
                'truncated': item.truncated,
                'recharge_matched': item.recharge_matched,
            })


def append_low_option_log(
    path: Path,
    episode: int,
    buffer: LowServeRolloutBuffer,
) -> None:
    fieldnames = [
        'episode', 'decision_id', 'mcs_id', 'low_action',
        'selected_candidate_id', 'stay_selected', 'forced_stay',
        'raw_quasi_count', 'safe_quasi_count', 'topk_truncated_count',
        'selected_urgency_rank', 'selected_urgency',
        'selected_remain_kwh', 'selected_need_power_kwh',
        'selected_distance_ratio', 'selected_attraction',
        'selected_mcs_competition', 'selected_fcs_competition',
        'selected_desirability', 'available_avg_attraction',
        'available_avg_mcs_competition',
        'available_avg_fcs_competition',
        'available_avg_desirability',
        'duration_steps', 'option_reward', 'delayed_reward',
        'advantage', 'return_target', 'value', 'terminal', 'truncated',
    ]
    write_header = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for item in buffer.transitions:
            writer.writerow({
                'episode': episode,
                'decision_id': item.decision_id,
                'mcs_id': item.mcs_id,
                'low_action': item.low_action,
                'selected_candidate_id': item.selected_candidate_id,
                'stay_selected': item.stay_selected,
                'forced_stay': item.forced_stay,
                'raw_quasi_count': item.raw_quasi_count,
                'safe_quasi_count': item.safe_quasi_count,
                'topk_truncated_count': item.topk_truncated_count,
                'selected_urgency_rank': item.selected_urgency_rank,
                'selected_urgency': item.selected_urgency,
                'selected_remain_kwh': item.selected_remain_kwh,
                'selected_need_power_kwh': item.selected_need_power_kwh,
                'selected_distance_ratio': item.selected_distance_ratio,
                'selected_attraction': item.selected_attraction,
                'selected_mcs_competition': (
                    item.selected_mcs_competition
                ),
                'selected_fcs_competition': (
                    item.selected_fcs_competition
                ),
                'selected_desirability': item.selected_desirability,
                'available_avg_attraction': item.available_avg_attraction,
                'available_avg_mcs_competition': (
                    item.available_avg_mcs_competition
                ),
                'available_avg_fcs_competition': (
                    item.available_avg_fcs_competition
                ),
                'available_avg_desirability': (
                    item.available_avg_desirability
                ),
                'duration_steps': item.duration_steps,
                'option_reward': item.option_reward,
                'delayed_reward': item.delayed_reward,
                'advantage': item.advantage,
                'return_target': item.return_target,
                'value': item.value,
                'terminal': item.terminal,
                'truncated': item.truncated,
            })


def rolling(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=max(window, 1), min_periods=1).mean()


def plot_convergence(log_table: pd.DataFrame, save_path: Path, window: int) -> None:
    episodes = log_table['episode']
    # High/Low 已使用独立的 reward、buffer、GAE 和 Critic，收敛曲线也必须
    # 分开呈现，避免汇总值掩盖其中一层的震荡或退化。
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)

    axes[0, 0].plot(
        episodes,
        rolling(log_table['high_actor_loss'], window),
        color='#264653',
    )
    axes[0, 0].set_title('High Actor Loss')

    axes[0, 1].plot(
        episodes,
        rolling(log_table['low_actor_loss'], window),
        color='#2A9D8F',
    )
    axes[0, 1].set_title('Low Actor Loss')

    axes[1, 0].plot(
        episodes,
        rolling(log_table['high_critic_loss'], window),
        color='#E9C46A',
    )
    axes[1, 0].set_title('High Critic Loss')

    axes[1, 1].plot(
        episodes,
        rolling(log_table['low_critic_loss'], window),
        color='#F4A261',
    )
    axes[1, 1].set_title('Low Critic Loss')

    axes[2, 0].plot(
        episodes,
        rolling(log_table['avg_high_reward'], window),
        color='#457B9D',
    )
    axes[2, 0].set_title('Average High Reward')

    axes[2, 1].plot(
        episodes,
        rolling(log_table['avg_low_reward'], window),
        color='#8A5AAB',
    )
    axes[2, 1].set_title('Average Low Reward')

    axes[3, 0].plot(
        episodes,
        rolling(log_table['charge_success_rate'], window),
        color='#1D7874',
    )
    axes[3, 0].set_ylim(0.0, 1.0)
    axes[3, 0].set_title('EV Charging Success Rate')

    axes[3, 1].plot(
        episodes,
        rolling(log_table['avg_mcs_profit'], window),
        color='#E76F51',
    )
    axes[3, 1].set_title('Average MCS total_profit')

    for axis in axes.flat:
        axis.set_xlabel('Episode')
        axis.grid(alpha=0.25)
    figure.savefig(save_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Option-aware MCS PPO training')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--episodes-per-update', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden-dim', type=int, default=max(HIDDEN_DIM, 128))
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=5e-4)
    parser.add_argument('--low-actor-lr', type=float, default=3e-4)
    parser.add_argument('--low-critic-lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--low-gamma', type=float, default=0.99)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--low-entropy-coef', type=float, default=0.01)
    parser.add_argument('--update-epochs', type=int, default=5)
    parser.add_argument('--minibatch-size', type=int, default=1024)
    parser.add_argument('--low-minibatch-size', type=int, default=512)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--plot-window', type=int, default=20)
    parser.add_argument('--checkpoint-every', type=int, default=40)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument(
        '--high-checkpoint',
        type=Path,
        default=None,
        help='仅加载 High Actor/High Critic；Low 分支保持从零初始化',
    )
    parser.add_argument(
        '--freeze-high',
        action='store_true',
        help='冻结 High Actor 与 High Critic，只更新 Low Actor/Critic',
    )
    return parser.parse_args()


def collect_episode(
    agent: MCSMAPPOAgent,
    args: argparse.Namespace,
    episode: int,
) -> tuple[Dict, HighOptionRolloutBuffer, LowServeRolloutBuffer]:
    """Collect one complete scenario using batched MCS policy inference."""
    scenario_seed = args.seed + episode - 1
    env = MultiAgentEnv(scenario_seed)
    env.world.verbose = False
    recharge_matcher = RechargeMatcher()
    observations = env.reset()
    high_buffer = HighOptionRolloutBuffer(args.gamma, args.gae_lambda)
    low_buffer = LowServeRolloutBuffer(args.low_gamma)
    episode_high_reward = 0.0
    episode_low_reward = 0.0
    next_low_decision_serial = 1
    reward_component_names = (
        'system_success',
        'system_failure',
        'attributed_mcs_success',
        'controllable_failure',
        'uncontrollable_failure',
        'fcs_success_kpi',
        'attributed_mcs_success_weight',
        'controllable_failure_weight',
        'service',
        'serve_attraction',
        'serve_competition',
        'serve_potential_improvement',
        'recharge',
        'movement',
        'wait',
        'wait_base',
        'wait_opportunity',
        'wait_streak',
        'wait_forced_time',
        'recharge_match_failure',
        'broken',
        'energy_stranded',
        'energy_stranded_event',
        'battery_potential',
        'high_step',
        'high_event',
        'recharge_step',
        'recharge_event',
        'broken_event',
        'high_total',
        'low_total',
        'low_step',
        'low_event',
        'low_step_weighted',
        'low_event_weighted',
        'resource_gap_improvement',
        'low_spatial_opportunity_gain',
        'low_attraction_metric',
        'low_mcs_competition_metric',
        'low_fcs_competition_metric',
        'urgency_coverage',
        'low_attraction_reward',
        'low_immediate_iev_attraction',
        'mcs_cluster_penalty',
        'fcs_redundancy_penalty',
        'high_serve_suitability',
        'low_attributed_mcs_success',
        'low_controllable_failure',
        'low_active_wait_count',
        'low_passive_wait_count',
    )
    reward_component_sums = {
        name: 0.0 for name in reward_component_names
    }
    action_counts = {'Serve': 0, 'Recharge': 0}
    recharge_request_count = 0
    recharge_match_count = 0
    event_audit_totals = {
        'attributed_mcs_success_count': 0,
        'attributed_mcs_success_weight_sum': 0.0,
        'controllable_failure_weight_sum': 0.0,
        'controllable_failure_count': 0,
        'uncontrollable_failure_count': 0,
        'fcs_success_kpi_count': 0,
        'unattributed_mcs_success_count': 0,
        'forced_wait_count': 0,
        'voluntary_wait_count': 0,
        'energy_stranded_count': 0,
        'low_active_wait_count': 0,
        'low_passive_wait_count': 0,
        'low_attributed_mcs_success_weight_sum': 0.0,
        'low_controllable_failure_weight_sum': 0.0,
    }
    executed_steps = 0
    ev_by_id = {ev.id: ev for ev in env.world.EVs}

    for step_index in range(args.max_steps):
        acting_agents = list(env.world.agents)
        observation_by_agent = {
            actor: observations[index]
            for index, actor in enumerate(acting_agents)
            if index < len(observations)
        }
        global_state = env.world.get_global_state()
        acting_mcss = [
            actor for actor in acting_agents if isinstance(actor, MCS)
        ]
        mcs_observations = [
            observation_by_agent[mcs] for mcs in acting_mcss
        ]
        # One High forward for all MCSs, followed by one Low forward for the
        # Serve subset.  On CUDA these are two small batched GPU kernels.
        mcs_actions = agent.select_mcs_actions_batch(mcs_observations)

        sampled: Dict[int, Dict] = {}
        recharge_requests: List[MCS] = []
        value_state_by_id: Dict[int, np.ndarray] = {}
        value_by_id: Dict[int, float] = {}
        low_value_by_id: Dict[int, float] = {}
        if acting_mcss:
            value_states = np.stack([
                critic_input(global_state, observation['high_state'])
                for observation in mcs_observations
            ])
            value_batch = agent.get_high_values_batch(value_states)
            for index, (mcs, observation, action) in enumerate(zip(
                acting_mcss, mcs_observations, mcs_actions
            )):
                sampled[mcs.id] = {
                    'mcs': mcs,
                    'observation': observation,
                    'action': action,
                }
                value_state_by_id[mcs.id] = value_states[index]
                value_by_id[mcs.id] = float(value_batch[index])
                action_counts[action['mode']] += 1
                if action['mode'] == 'Recharge':
                    recharge_requests.append(mcs)
                    recharge_request_count += 1

            serve_batch_indices = [
                index for index, action in enumerate(mcs_actions)
                if action['mode'] == 'Serve'
            ]
            if serve_batch_indices:
                low_value_batch = agent.get_low_values_batch(
                    np.stack([
                        global_state for _index in serve_batch_indices
                    ]),
                    np.stack([
                        mcs_observations[index]['low_self_state']
                        for index in serve_batch_indices
                    ]),
                    np.stack([
                        mcs_observations[index]['low_candidates']
                        for index in serve_batch_indices
                    ]),
                    np.stack([
                        mcs_observations[index]['low_candidate_mask']
                        for index in serve_batch_indices
                    ]),
                )
                for batch_index, observation_index in enumerate(
                    serve_batch_indices
                ):
                    low_value_by_id[
                        acting_mcss[observation_index].id
                    ] = float(low_value_batch[batch_index])

        recharge_results = recharge_matcher.match_all(
            recharge_requests, env.world.FCSs
        )
        recharge_matched_ids = {
            result['mcs_id']
            for result in recharge_results
            if result.get('success')
        }
        recharge_match_count += len(recharge_matched_ids)

        action_n = []
        for actor in acting_agents:
            if isinstance(actor, EV):
                action_n.append(iev_track_action(actor))
                continue

            decision = sampled[actor.id]
            observation = decision['observation']
            action = decision['action']
            matched = actor.id in recharge_matched_ids
            high_buffer.start_option(
                actor.id,
                observation,
                action,
                value_state_by_id[actor.id],
                value_by_id[actor.id],
                matched,
            )

            if action['mode'] == 'Recharge':
                if matched:
                    action_n.append({
                        'mode': 'Recharge',
                        'requested_mode': 'Recharge',
                        'recharge_matched': True,
                        'high_action_mask': observation[
                            'high_action_mask'
                        ].tolist(),
                        'target_pos': list(actor.current_target_pos),
                    })
                else:
                    action_n.append({
                        'mode': 'Wait',
                        'requested_mode': 'Recharge',
                        'recharge_matched': False,
                        'high_action_mask': observation[
                            'high_action_mask'
                        ].tolist(),
                        'target_pos': list(actor.pos),
                    })
            else:
                candidate_index = action['low_action']
                candidate_id = int(
                    observation['candidate_ids'][candidate_index]
                )
                low_stay_selected = candidate_id == MCS_STAY_CANDIDATE_ID
                has_quasi_candidate = bool(
                    observation.get('quasi_candidate_count', 0) > 0
                )
                low_forced_stay = bool(
                    low_stay_selected and not has_quasi_candidate
                )
                low_decision_id = (
                    int(episode) * 1_000_000 + next_low_decision_serial
                )
                next_low_decision_serial += 1
                low_buffer.start_option(
                    low_decision_id,
                    actor.id,
                    observation,
                    action,
                    global_state,
                    low_value_by_id[actor.id],
                )
                target_pos = (
                    list(actor.pos)
                    if low_stay_selected
                    else list(ev_by_id[candidate_id].pos)
                )
                action_n.append({
                    'mode': 'Serve',
                    'requested_mode': 'Serve',
                    'recharge_matched': False,
                    'high_action_mask': observation[
                        'high_action_mask'
                    ].tolist(),
                    'target_pos': target_pos,
                    'low_decision_id': low_decision_id,
                    'serve_option_id': low_decision_id,
                    'low_candidate_id': candidate_id,
                    'low_stay_selected': low_stay_selected,
                    'low_forced_stay': low_forced_stay,
                    'has_quasi_candidate': has_quasi_candidate,
                })

        new_observations, _, _, _ = env.step(action_n)
        executed_steps += 1
        high_reward_by_mcs = {
            mcs.id: env.world.last_mcs_reward_components[mcs.id][
                'high_total'
            ]
            for mcs in env.world.MCSs
        }
        low_reward_by_decision = dict(
            env.world.last_low_reward_by_decision
        )
        episode_high_reward += sum(high_reward_by_mcs.values())
        episode_low_reward += sum(low_reward_by_decision.values())
        for components in env.world.last_mcs_reward_components.values():
            for name in reward_component_names:
                reward_component_sums[name] += float(
                    components.get(name, 0.0)
                )
        for name in event_audit_totals:
            event_audit_totals[name] += env.world.last_system_reward_event.get(
                name, 0
            )
        high_buffer.add_step_rewards(high_reward_by_mcs)
        low_buffer.add_step_rewards(low_reward_by_decision)

        next_agents = list(env.world.agents)
        next_observation_by_agent = {
            actor: new_observations[index]
            for index, actor in enumerate(next_agents)
            if index < len(new_observations)
        }
        next_global_state = env.world.get_global_state()
        ready_mcss = [
            actor for actor in next_agents if isinstance(actor, MCS)
        ]
        ready_ids = {mcs.id for mcs in ready_mcss}
        next_states: Dict[int, np.ndarray] = {}
        next_values: Dict[int, float] = {}
        if ready_mcss:
            ready_state_batch = np.stack([
                critic_input(
                    next_global_state,
                    next_observation_by_agent[mcs]['high_state'],
                )
                for mcs in ready_mcss
            ])
            ready_value_batch = agent.get_high_values_batch(ready_state_batch)
            for index, mcs in enumerate(ready_mcss):
                next_states[mcs.id] = ready_state_batch[index]
                next_values[mcs.id] = float(ready_value_batch[index])

        terminal_mcs_ids = {
            mcs.id for mcs in env.world.MCSs
            if mcs.is_broken or mcs.is_energy_stranded
        }
        final_step = step_index + 1 >= args.max_steps or env.world.get_done()
        high_buffer.close_options(
            decision_ready_ids=ready_ids,
            next_critic_states=next_states,
            next_values=next_values,
            terminal_ids=terminal_mcs_ids,
            close_all=final_step,
            truncated=final_step,
        )
        low_buffer.close_options(
            decision_ready_ids=ready_ids,
            terminal_ids=terminal_mcs_ids,
            close_all=final_step,
            truncated=final_step,
        )
        observations = new_observations
        if final_step:
            break

    high_buffer.compute_returns_and_advantages()
    low_buffer.compute_returns_and_advantages()
    successes = sum(
        ev.is_charged for ev in env.world.EVs
    )
    mcs_successes = sum(
        ev.is_charged and ev.charge_provider_type == 'MCS'
        for ev in env.world.EVs
    )
    fcs_successes = sum(
        ev.is_charged and ev.charge_provider_type == 'FCS'
        for ev in env.world.EVs
    )
    failures = sum(ev.fail_charge for ev in env.world.EVs)
    # 仅记录终止时仍存在的未完成充电需求。其业务归类尚不明确，因此不
    # 擅自改写为 fail，也不在本版本追加终止责任惩罚。
    unresolved_count = sum(
        ev.need_charge and not ev.is_charged and not ev.fail_charge
        for ev in env.world.EVs
    )
    finished = successes + failures
    success_rate = successes / finished if finished else 0.0
    mcs_count = max(len(env.world.MCSs), 1)
    total_mcs_profit = float(sum(
        mcs.total_profit for mcs in env.world.MCSs
    ))
    total_mcs_cost = float(sum(
        mcs.total_cost for mcs in env.world.MCSs
    ))
    total_mcs_energy_consumed = float(sum(
        mcs.total_energy_consumed for mcs in env.world.MCSs
    ))
    total_mcs_charged_kwh = float(sum(
        mcs.total_charged_kwh for mcs in env.world.MCSs
    ))
    total_mcs_idle_time_min = float(sum(
        mcs.total_idle_time_min for mcs in env.world.MCSs
    ))
    reward_denominator = max(executed_steps * mcs_count, 1)
    high_durations = [
        item.duration_steps for item in high_buffer.transitions
    ]
    low_durations = [
        item.duration_steps for item in low_buffer.transitions
    ]
    low_transitions = list(low_buffer.transitions)
    moving_low_transitions = [
        item for item in low_transitions if not item.stay_selected
    ]

    def transition_mean(items, field: str) -> float:
        return float(np.mean([
            float(getattr(item, field)) for item in items
        ])) if items else 0.0

    def transition_correlation(items, field: str) -> float:
        if len(items) < 2:
            return 0.0
        values = np.asarray([
            float(getattr(item, field)) for item in items
        ], dtype=np.float64)
        rewards = np.asarray([
            float(item.option_reward) for item in items
        ], dtype=np.float64)
        if np.std(values) < 1e-12 or np.std(rewards) < 1e-12:
            return 0.0
        return float(np.corrcoef(values, rewards)[0, 1])

    low_count = max(len(low_transitions), 1)
    moving_low_count = max(len(moving_low_transitions), 1)
    low_stay_count = sum(item.stay_selected for item in low_transitions)
    low_forced_stay_count = sum(
        item.forced_stay for item in low_transitions
    )
    low_active_stay_count = low_stay_count - low_forced_stay_count
    selected_minus_available_attraction = transition_mean(
        moving_low_transitions, 'selected_attraction'
    ) - transition_mean(moving_low_transitions, 'available_avg_attraction')
    selected_minus_available_mcs_competition = transition_mean(
        moving_low_transitions, 'selected_mcs_competition'
    ) - transition_mean(
        moving_low_transitions, 'available_avg_mcs_competition'
    )
    selected_minus_available_fcs_competition = transition_mean(
        moving_low_transitions, 'selected_fcs_competition'
    ) - transition_mean(
        moving_low_transitions, 'available_avg_fcs_competition'
    )
    selected_minus_available_desirability = transition_mean(
        moving_low_transitions, 'selected_desirability'
    ) - transition_mean(
        moving_low_transitions, 'available_avg_desirability'
    )
    episode_low_step_weighted = float(
        reward_component_sums['low_step_weighted']
    )
    episode_low_event_weighted = float(
        reward_component_sums['low_event_weighted']
    )
    low_reward_absolute_mass = (
        abs(episode_low_step_weighted) + abs(episode_low_event_weighted)
    )
    row = {
        'episode': episode,
        'scenario_seed': scenario_seed,
        'steps': executed_steps,
        'time_limit_truncated': bool(executed_steps >= args.max_steps),
        'option_count': len(high_buffer),
        'high_option_count': len(high_buffer),
        'low_option_count': len(low_buffer),
        'low_move_selection_count': len(moving_low_transitions),
        'low_stay_selection_count': low_stay_count,
        'low_active_stay_selection_count': low_active_stay_count,
        'low_forced_stay_selection_count': low_forced_stay_count,
        'low_stay_selection_rate': low_stay_count / low_count,
        'low_active_stay_selection_rate': (
            low_active_stay_count / low_count
        ),
        'low_forced_stay_selection_rate': (
            low_forced_stay_count / low_count
        ),
        'low_top1_urgency_selection_rate': (
            sum(
                item.selected_urgency_rank == 1
                for item in moving_low_transitions
            ) / moving_low_count
        ),
        'avg_low_selected_urgency_rank': transition_mean(
            moving_low_transitions, 'selected_urgency_rank'
        ),
        'avg_low_selected_urgency': transition_mean(
            moving_low_transitions, 'selected_urgency'
        ),
        'avg_low_selected_remain_kwh': transition_mean(
            moving_low_transitions, 'selected_remain_kwh'
        ),
        'avg_low_selected_need_power_kwh': transition_mean(
            moving_low_transitions, 'selected_need_power_kwh'
        ),
        'avg_low_selected_distance_ratio': transition_mean(
            moving_low_transitions, 'selected_distance_ratio'
        ),
        'avg_low_selected_attraction': transition_mean(
            moving_low_transitions, 'selected_attraction'
        ),
        'avg_low_selected_mcs_competition': transition_mean(
            moving_low_transitions, 'selected_mcs_competition'
        ),
        'avg_low_selected_fcs_competition': transition_mean(
            moving_low_transitions, 'selected_fcs_competition'
        ),
        'avg_low_selected_desirability': transition_mean(
            moving_low_transitions, 'selected_desirability'
        ),
        'avg_low_available_attraction': transition_mean(
            moving_low_transitions, 'available_avg_attraction'
        ),
        'avg_low_available_mcs_competition': transition_mean(
            moving_low_transitions, 'available_avg_mcs_competition'
        ),
        'avg_low_available_fcs_competition': transition_mean(
            moving_low_transitions, 'available_avg_fcs_competition'
        ),
        'avg_low_available_desirability': transition_mean(
            moving_low_transitions, 'available_avg_desirability'
        ),
        'low_selected_minus_available_attraction': (
            selected_minus_available_attraction
        ),
        'low_selected_minus_available_mcs_competition': (
            selected_minus_available_mcs_competition
        ),
        'low_selected_minus_available_fcs_competition': (
            selected_minus_available_fcs_competition
        ),
        'low_selected_minus_available_desirability': (
            selected_minus_available_desirability
        ),
        'avg_low_raw_quasi_count': transition_mean(
            low_transitions, 'raw_quasi_count'
        ),
        'avg_low_safe_quasi_count': transition_mean(
            low_transitions, 'safe_quasi_count'
        ),
        'avg_low_topk_truncated_count': transition_mean(
            low_transitions, 'topk_truncated_count'
        ),
        'low_topk_truncation_rate': (
            sum(
                item.topk_truncated_count > 0 for item in low_transitions
            ) / low_count
        ),
        'low_selected_urgency_reward_corr': transition_correlation(
            moving_low_transitions, 'selected_urgency'
        ),
        'low_selected_need_power_reward_corr': transition_correlation(
            moving_low_transitions, 'selected_need_power_kwh'
        ),
        'low_selected_mcs_competition_reward_corr': (
            transition_correlation(
                moving_low_transitions, 'selected_mcs_competition'
            )
        ),
        'low_selected_fcs_competition_reward_corr': (
            transition_correlation(
                moving_low_transitions, 'selected_fcs_competition'
            )
        ),
        'serve_option_count': action_counts['Serve'],
        'recharge_option_count': action_counts['Recharge'],
        # High Wait 已移除；保留旧列便于历史CSV拼接。
        'wait_option_count': 0,
        'recharge_match_count': recharge_match_count,
        'recharge_match_rate': (
            recharge_match_count / recharge_request_count
            if recharge_request_count else 0.0
        ),
        'avg_option_duration': float(np.mean(high_durations)) if high_durations else 0.0,
        'avg_high_option_duration': float(np.mean(high_durations)) if high_durations else 0.0,
        'avg_low_option_duration': float(np.mean(low_durations)) if low_durations else 0.0,
        'episode_high_reward': episode_high_reward,
        'episode_low_reward': episode_low_reward,
        'episode_low_step_weighted': episode_low_step_weighted,
        'episode_low_event_weighted': episode_low_event_weighted,
        'low_event_absolute_contribution_ratio': (
            abs(episode_low_event_weighted) / low_reward_absolute_mass
            if low_reward_absolute_mass > 1e-12 else 0.0
        ),
        'episode_reward': episode_high_reward + episode_low_reward,
        'avg_high_reward': episode_high_reward / reward_denominator,
        'avg_low_reward': episode_low_reward / reward_denominator,
        'avg_reward': (
            episode_high_reward + episode_low_reward
        ) / reward_denominator,
        'charge_success_rate': success_rate,
        'charge_success_count': successes,
        'charge_success_mcs_count': mcs_successes,
        'charge_success_fcs_count': fcs_successes,
        'charge_failure_count': failures,
        'unresolved_count': unresolved_count,
        'attributed_mcs_success_count': int(
            event_audit_totals['attributed_mcs_success_count']
        ),
        'attributed_mcs_success_weight_sum': float(
            event_audit_totals['attributed_mcs_success_weight_sum']
        ),
        'low_attributed_mcs_success_weight_sum': float(
            event_audit_totals[
                'low_attributed_mcs_success_weight_sum'
            ]
        ),
        'controllable_failure_count': int(
            event_audit_totals['controllable_failure_count']
        ),
        'uncontrollable_failure_count': int(
            event_audit_totals['uncontrollable_failure_count']
        ),
        'controllable_failure_weight_sum': float(
            event_audit_totals['controllable_failure_weight_sum']
        ),
        'low_controllable_failure_weight_sum': float(
            event_audit_totals[
                'low_controllable_failure_weight_sum'
            ]
        ),
        'fcs_success_kpi_count': int(
            event_audit_totals['fcs_success_kpi_count']
        ),
        'unattributed_mcs_success_count': int(
            event_audit_totals['unattributed_mcs_success_count']
        ),
        'forced_wait_count': int(event_audit_totals['forced_wait_count']),
        'voluntary_wait_count': int(
            event_audit_totals['voluntary_wait_count']
        ),
        'energy_stranded_count': int(
            event_audit_totals['energy_stranded_count']
        ),
        'low_active_wait_count': int(
            event_audit_totals['low_active_wait_count']
        ),
        'low_passive_wait_count': int(
            event_audit_totals['low_passive_wait_count']
        ),
        'low_delayed_event_count': int(low_buffer.delayed_event_count),
        'total_mcs_profit': total_mcs_profit,
        'avg_mcs_profit': total_mcs_profit / mcs_count,
        'total_mcs_cost': total_mcs_cost,
        'avg_mcs_cost': total_mcs_cost / mcs_count,
        'total_mcs_energy_consumed_kwh': total_mcs_energy_consumed,
        'avg_mcs_energy_consumed_kwh': total_mcs_energy_consumed / mcs_count,
        'total_mcs_charged_kwh': total_mcs_charged_kwh,
        'avg_mcs_charged_kwh': total_mcs_charged_kwh / mcs_count,
        'total_mcs_idle_time_min': total_mcs_idle_time_min,
        'avg_mcs_idle_time_min': total_mcs_idle_time_min / mcs_count,
        'broken_mcs_count': sum(mcs.is_broken for mcs in env.world.MCSs),
        'energy_stranded_mcs_count': sum(
            mcs.is_energy_stranded for mcs in env.world.MCSs
        ),
        **{
            f'avg_reward_{name}': total / reward_denominator
            for name, total in reward_component_sums.items()
        },
    }
    return row, high_buffer, low_buffer


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.high_checkpoint is not None:
        raise ValueError('--resume 与 --high-checkpoint 不能同时使用')
    if args.freeze_high and args.high_checkpoint is None:
        raise ValueError('--freeze-high 必须配合 --high-checkpoint')
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError('episodes and max-steps must be positive')
    if (
        args.episodes_per_update <= 0
        or args.minibatch_size <= 0
        or args.low_minibatch_size <= 0
    ):
        raise ValueError('episodes-per-update and minibatch-size must be positive')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / 'training_log.csv'
    high_option_log_path = output_dir / 'high_option_rollout.csv'
    low_option_log_path = output_dir / 'low_option_rollout.csv'
    curve_path = output_dir / 'convergence_curves.png'
    latest_model_path = output_dir / 'latest_model.pt'
    best_model_path = output_dir / 'best_model.pt'
    if args.resume is None:
        for path in (
            training_log_path, high_option_log_path, low_option_log_path
        ):
            if path.exists():
                path.unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False')
    device = (
        'cuda'
        if args.device == 'cuda'
        or (args.device == 'auto' and torch.cuda.is_available())
        else 'cpu'
    )
    cuda_name = ''
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision('high')
        cuda_name = torch.cuda.get_device_name(0)

    agent = MCSMAPPOAgent(
        high_state_dim=MCS_HIGH_FEAT_DIM,
        low_self_dim=MCS_FEAT_DIM_self,
        low_candidate_dim=MCS_FEAT_DIM_tgt,
        critic_state_dim=MCS_CRITIC_STATE_DIM,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        low_actor_lr=args.low_actor_lr,
        low_critic_lr=args.low_critic_lr,
        device=device,
    )
    start_episode = 1
    log_rows: List[Dict] = []
    best_update_reward = -float('inf')
    if args.resume is not None:
        metadata = agent.load(args.resume, load_optimizers=True)
        start_episode = int(metadata.get('episode', 0)) + 1
        best_update_reward = float(
            metadata.get('best_update_reward', best_update_reward)
        )
        if training_log_path.exists():
            log_rows = pd.read_csv(training_log_path).to_dict('records')
    elif args.high_checkpoint is not None:
        source_metadata = agent.load_high_branch(
            args.high_checkpoint,
            load_high_critic=True,
            freeze=args.freeze_high,
        )
        source_episode = int(source_metadata.get('episode', -1))
        print(
            f'high_checkpoint={args.high_checkpoint.resolve()} '
            f'source_episode={source_episode} '
            f'high_frozen={args.freeze_high} '
            'low_initialized_from_scratch=True'
        )

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config.update({
        'output_dir': str(output_dir),
        'resolved_device': device,
        'cuda_device_name': cuda_name,
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'low_reward_design_version': (
            'v8_count_opportunity_urgency_topk'
        ),
        'serve_quasi_topk_capacity': max(
            int(TOP_K_MCS_CANDIDATES) - 1, 0
        ),
        'iev_transition_threshold_kwh': float(EV_LOW_POWER_THRESHOLD),
        'low_step_reward_weight': float(LOW_STEP_REWARD_WEIGHT),
        'low_event_reward_weight': float(LOW_EVENT_REWARD_WEIGHT),
        'low_spatial_opportunity_weight': float(
            LOW_SPATIAL_OPPORTUNITY_WEIGHT
        ),
        'low_serve_move_penalty': float(SERVE_MOVE_PENALTY),
        'low_fcs_alternative_penalty': float(
            LOW_FCS_ALTERNATIVE_PENALTY
        ),
        'low_mcs_alternative_penalty': float(
            LOW_MCS_ALTERNATIVE_PENALTY
        ),
    })
    (output_dir / 'training_config.json').write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(
        f'device={device} cuda_name={cuda_name or "NONE"} '
        f'episodes_per_update={args.episodes_per_update} '
        f'minibatch={args.minibatch_size}'
    )

    update_high_buffer = HighOptionRolloutBuffer(
        args.gamma, args.gae_lambda
    )
    update_low_buffer = LowServeRolloutBuffer(args.low_gamma)
    pending_episodes: List[tuple[
        int, Dict, HighOptionRolloutBuffer, LowServeRolloutBuffer
    ]] = []
    update_index = 0
    final_episode = start_episode + args.episodes - 1
    last_checkpoint_episode = start_episode - 1

    for episode in range(start_episode, final_episode + 1):
        episode_row, episode_high_buffer, episode_low_buffer = collect_episode(
            agent, args, episode
        )
        update_high_buffer.extend_completed(episode_high_buffer)
        update_low_buffer.extend_completed(episode_low_buffer)
        pending_episodes.append((
            episode,
            episode_row,
            episode_high_buffer,
            episode_low_buffer,
        ))
        group_complete = (
            len(pending_episodes) >= args.episodes_per_update
            or episode == final_episode
        )
        if not group_complete:
            continue

        update_index += 1
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        loss_metrics = ppo_update(
            agent,
            update_high_buffer,
            update_low_buffer,
            args.update_epochs,
            args.minibatch_size,
            args.low_minibatch_size,
            args.clip_ratio,
            args.entropy_coef,
            args.low_entropy_coef,
            args.max_grad_norm,
            train_high=not args.freeze_high,
        )
        peak_cuda_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            if device == 'cuda' else 0.0
        )
        update_avg_reward = float(np.mean([
            row['avg_reward'] for _, row, _, _ in pending_episodes
        ]))
        update_episode_count = len(pending_episodes)
        update_high_sample_count = len(update_high_buffer)
        update_low_sample_count = len(update_low_buffer)
        update_sample_count = (
            update_high_sample_count + update_low_sample_count
        )

        for (
            completed_episode,
            row,
            completed_high_buffer,
            completed_low_buffer,
        ) in pending_episodes:
            row.update(loss_metrics)
            row.update({
                'update_index': update_index,
                'episodes_in_update': update_episode_count,
                'update_sample_count': update_sample_count,
                'update_high_sample_count': update_high_sample_count,
                'update_low_sample_count': update_low_sample_count,
                'update_avg_reward': update_avg_reward,
                'training_device': device,
                'peak_cuda_memory_mb': peak_cuda_memory_mb,
            })
            log_rows.append(row)
            append_high_option_log(
                high_option_log_path,
                completed_episode,
                completed_high_buffer,
            )
            append_low_option_log(
                low_option_log_path,
                completed_episode,
                completed_low_buffer,
            )

        log_table = pd.DataFrame(log_rows)
        log_table.to_csv(training_log_path, index=False)
        persisted_log = pd.read_csv(training_log_path)
        plot_convergence(persisted_log, curve_path, args.plot_window)

        is_best = update_avg_reward > best_update_reward
        if is_best:
            best_update_reward = update_avg_reward
        metadata = {
            'episode': episode,
            'update_index': update_index,
            'best_update_reward': best_update_reward,
            'config': run_config,
        }
        agent.save(latest_model_path, metadata)
        if is_best:
            agent.save(best_model_path, metadata)
        if (
            args.checkpoint_every > 0
            and episode - last_checkpoint_episode >= args.checkpoint_every
        ):
            agent.save(
                output_dir / f'model_episode_{episode}.pt', metadata
            )
            last_checkpoint_episode = episode

        print(
            f'update={update_index} episodes='
            f'{pending_episodes[0][0]}-{episode} '
            f'samples={update_sample_count} '
            f'avg_reward={update_avg_reward:.4f} '
            f'actor_loss={loss_metrics["actor_loss"]:.4f} '
            f'critic_loss={loss_metrics["critic_loss"]:.4f} '
            f'cuda_peak_mb={peak_cuda_memory_mb:.1f}'
        )
        update_high_buffer = HighOptionRolloutBuffer(
            args.gamma, args.gae_lambda
        )
        update_low_buffer = LowServeRolloutBuffer(args.low_gamma)
        pending_episodes.clear()

    # Always retain an explicit final numbered checkpoint.
    final_metadata = {
        'episode': final_episode,
        'update_index': update_index,
        'best_update_reward': best_update_reward,
        'config': run_config,
    }
    agent.save(
        output_dir / f'model_episode_{final_episode}.pt', final_metadata
    )
    print(f'training log: {training_log_path}')
    print(f'high option rollout: {high_option_log_path}')
    print(f'low option rollout: {low_option_log_path}')
    print(f'convergence curves: {curve_path}')


if __name__ == '__main__':
    main()
