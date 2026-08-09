"""Train the MCS High/Low actors with option-aware PPO.

IEV actions remain deterministic track-following actions.  The trainable policy
contains only the MCS High Actor (Serve/Recharge/Wait), the Serve-conditioned
Low Actor (quasi candidate), and an agent-conditioned centralized critic.
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
    HIDDEN_DIM,
    MAX_STEPS_PER_EPISODE,
    MCS_CRITIC_STATE_DIM,
    MCS_FEAT_DIM_self,
    MCS_FEAT_DIM_tgt,
    MCS_HIGH_FEAT_DIM,
    TRACK_DATA_PATH,
)
from core import EV, MCS
from environment import MultiAgentEnv
from matching import RechargeMatcher
from network import MCSMAPPOAgent
from rollout import OptionAwareRolloutBuffer

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


def ppo_update(
    agent: MCSMAPPOAgent,
    buffer: OptionAwareRolloutBuffer,
    update_epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    data = buffer.as_tensors(agent.device)
    advantages = normalize_advantages(data['advantages'])
    sample_count = advantages.shape[0]
    high_losses: List[float] = []
    low_losses: List[float] = []
    critic_losses: List[float] = []
    high_entropies: List[float] = []
    low_entropies: List[float] = []

    for _ in range(update_epochs):
        permutation = torch.randperm(sample_count, device=agent.device)
        for start in range(0, sample_count, minibatch_size):
            indices = permutation[start:start + minibatch_size]
            batch_advantages = advantages[indices]

            new_high_log_probs, high_entropy = agent.evaluate_high(
                data['high_states'][indices],
                data['high_masks'][indices],
                data['high_actions'][indices],
            )
            high_ratio = torch.exp(
                new_high_log_probs - data['old_high_log_probs'][indices]
            )
            high_surrogate = high_ratio * batch_advantages
            high_clipped = torch.clamp(
                high_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * batch_advantages
            high_loss = -torch.min(high_surrogate, high_clipped).mean()
            high_objective = high_loss - entropy_coef * high_entropy.mean()
            agent.high_optimizer.zero_grad()
            high_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.high_actor.parameters(), max_grad_norm
            )
            agent.high_optimizer.step()
            high_losses.append(float(high_loss.item()))
            high_entropies.append(float(high_entropy.mean().item()))

            serve_mask = data['low_actions'][indices] >= 0
            if serve_mask.any():
                low_indices = indices[serve_mask]
                low_advantages = advantages[low_indices]
                new_low_log_probs, low_entropy = agent.evaluate_low(
                    data['low_self_states'][low_indices],
                    data['low_candidates'][low_indices],
                    data['low_masks'][low_indices],
                    data['low_actions'][low_indices],
                )
                low_ratio = torch.exp(
                    new_low_log_probs - data['old_low_log_probs'][low_indices]
                )
                low_surrogate = low_ratio * low_advantages
                low_clipped = torch.clamp(
                    low_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                ) * low_advantages
                low_loss = -torch.min(low_surrogate, low_clipped).mean()
                low_objective = low_loss - entropy_coef * low_entropy.mean()
                agent.low_optimizer.zero_grad()
                low_objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.low_actor.parameters(), max_grad_norm
                )
                agent.low_optimizer.step()
                low_losses.append(float(low_loss.item()))
                low_entropies.append(float(low_entropy.mean().item()))

            predicted_values = agent.values(data['critic_states'][indices])
            critic_loss = 0.5 * F.mse_loss(
                predicted_values, data['returns'][indices]
            )
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.critic.parameters(), max_grad_norm
            )
            agent.critic_optimizer.step()
            critic_losses.append(float(critic_loss.item()))

    high_mean = float(np.mean(high_losses)) if high_losses else 0.0
    low_mean = float(np.mean(low_losses)) if low_losses else 0.0
    return {
        'actor_loss': high_mean + low_mean,
        'high_actor_loss': high_mean,
        'low_actor_loss': low_mean,
        'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
        'high_entropy': float(np.mean(high_entropies)) if high_entropies else 0.0,
        'low_entropy': float(np.mean(low_entropies)) if low_entropies else 0.0,
    }


def append_option_log(
    path: Path,
    episode: int,
    buffer: OptionAwareRolloutBuffer,
) -> None:
    fieldnames = [
        'episode', 'mcs_id', 'high_action', 'duration_steps',
        'option_reward', 'advantage', 'return_target', 'value',
        'terminal', 'recharge_matched', 'has_low_action',
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
                'recharge_matched': item.recharge_matched,
                'has_low_action': item.low_action >= 0,
            })


def rolling(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=max(window, 1), min_periods=1).mean()


def plot_convergence(log_table: pd.DataFrame, save_path: Path, window: int) -> None:
    episodes = log_table['episode']
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].plot(episodes, rolling(log_table['actor_loss'], window), label='Actor')
    axes[0, 0].plot(episodes, rolling(log_table['critic_loss'], window), label='Critic')
    axes[0, 0].set_title('PPO Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(
        episodes, rolling(log_table['avg_reward'], window), color='#2A9D8F'
    )
    axes[0, 1].set_title('Average MCS Reward')

    axes[1, 0].plot(
        episodes,
        rolling(log_table['charge_success_rate'], window),
        color='#457B9D',
    )
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title('EV Charging Success Rate')

    axes[1, 1].plot(
        episodes,
        rolling(log_table['avg_mcs_profit'], window),
        color='#E76F51',
    )
    axes[1, 1].set_title('Average MCS total_profit')

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
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--update-epochs', type=int, default=5)
    parser.add_argument('--minibatch-size', type=int, default=1024)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--plot-window', type=int, default=20)
    parser.add_argument('--checkpoint-every', type=int, default=40)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--resume', type=Path, default=None)
    return parser.parse_args()


def collect_episode(
    agent: MCSMAPPOAgent,
    args: argparse.Namespace,
    episode: int,
) -> tuple[Dict, OptionAwareRolloutBuffer]:
    """Collect one complete scenario using batched MCS policy inference."""
    scenario_seed = args.seed + episode - 1
    env = MultiAgentEnv(scenario_seed)
    env.world.verbose = False
    recharge_matcher = RechargeMatcher()
    observations = env.reset()
    episode_buffer = OptionAwareRolloutBuffer(args.gamma, args.gae_lambda)
    episode_reward = 0.0
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
        'battery_potential',
    )
    reward_component_sums = {
        name: 0.0 for name in reward_component_names
    }
    action_counts = {'Serve': 0, 'Recharge': 0, 'Wait': 0}
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
        if acting_mcss:
            value_states = np.stack([
                critic_input(global_state, observation['high_state'])
                for observation in mcs_observations
            ])
            value_batch = agent.get_values_batch(value_states)
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
            episode_buffer.start_option(
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
            elif action['mode'] == 'Wait':
                action_n.append({
                    'mode': 'Wait',
                    'requested_mode': 'Wait',
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
                target = ev_by_id[candidate_id]
                action_n.append({
                    'mode': 'Serve',
                    'requested_mode': 'Serve',
                    'recharge_matched': False,
                    'high_action_mask': observation[
                        'high_action_mask'
                    ].tolist(),
                    'target_pos': list(target.pos),
                })

        new_observations, _, _, _ = env.step(action_n)
        executed_steps += 1
        reward_by_mcs = {
            mcs.id: env.world.last_mcs_reward_components[mcs.id]['total']
            for mcs in env.world.MCSs
        }
        episode_reward += sum(reward_by_mcs.values())
        for components in env.world.last_mcs_reward_components.values():
            for name in reward_component_names:
                reward_component_sums[name] += float(
                    components.get(name, 0.0)
                )
        for name in event_audit_totals:
            event_audit_totals[name] += env.world.last_system_reward_event.get(
                name, 0
            )
        episode_buffer.add_step_rewards(reward_by_mcs)

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
            ready_value_batch = agent.get_values_batch(ready_state_batch)
            for index, mcs in enumerate(ready_mcss):
                next_states[mcs.id] = ready_state_batch[index]
                next_values[mcs.id] = float(ready_value_batch[index])

        broken_ids = {mcs.id for mcs in env.world.MCSs if mcs.is_broken}
        final_step = step_index + 1 >= args.max_steps or env.world.get_done()
        episode_buffer.close_options(
            decision_ready_ids=ready_ids,
            next_critic_states=next_states,
            next_values=next_values,
            terminal_ids=broken_ids,
            close_all=final_step,
        )
        observations = new_observations
        if final_step:
            break

    episode_buffer.compute_returns_and_advantages()
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
    durations = [item.duration_steps for item in episode_buffer.transitions]
    row = {
        'episode': episode,
        'scenario_seed': scenario_seed,
        'steps': executed_steps,
        'option_count': len(episode_buffer),
        'serve_option_count': action_counts['Serve'],
        'recharge_option_count': action_counts['Recharge'],
        'wait_option_count': action_counts['Wait'],
        'recharge_match_count': recharge_match_count,
        'recharge_match_rate': (
            recharge_match_count / recharge_request_count
            if recharge_request_count else 0.0
        ),
        'avg_option_duration': float(np.mean(durations)) if durations else 0.0,
        'episode_reward': episode_reward,
        'avg_reward': episode_reward / reward_denominator,
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
        'controllable_failure_count': int(
            event_audit_totals['controllable_failure_count']
        ),
        'uncontrollable_failure_count': int(
            event_audit_totals['uncontrollable_failure_count']
        ),
        'controllable_failure_weight_sum': float(
            event_audit_totals['controllable_failure_weight_sum']
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
        **{
            f'avg_reward_{name}': total / reward_denominator
            for name, total in reward_component_sums.items()
        },
    }
    return row, episode_buffer


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError('episodes and max-steps must be positive')
    if args.episodes_per_update <= 0 or args.minibatch_size <= 0:
        raise ValueError('episodes-per-update and minibatch-size must be positive')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / 'training_log.csv'
    option_log_path = output_dir / 'option_rollout.csv'
    curve_path = output_dir / 'convergence_curves.png'
    latest_model_path = output_dir / 'latest_model.pt'
    best_model_path = output_dir / 'best_model.pt'
    if args.resume is None:
        for path in (training_log_path, option_log_path):
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
        device=device,
    )
    start_episode = 1
    log_rows: List[Dict] = []
    best_update_reward = -float('inf')
    if args.resume is not None:
        metadata = agent.load(args.resume)
        start_episode = int(metadata.get('episode', 0)) + 1
        best_update_reward = float(
            metadata.get('best_update_reward', best_update_reward)
        )
        if training_log_path.exists():
            log_rows = pd.read_csv(training_log_path).to_dict('records')

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

    update_buffer = OptionAwareRolloutBuffer(args.gamma, args.gae_lambda)
    pending_episodes: List[tuple[int, Dict, OptionAwareRolloutBuffer]] = []
    update_index = 0
    final_episode = start_episode + args.episodes - 1
    last_checkpoint_episode = start_episode - 1

    for episode in range(start_episode, final_episode + 1):
        episode_row, episode_buffer = collect_episode(
            agent, args, episode
        )
        update_buffer.extend_completed(episode_buffer)
        pending_episodes.append((episode, episode_row, episode_buffer))
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
            update_buffer,
            args.update_epochs,
            args.minibatch_size,
            args.clip_ratio,
            args.entropy_coef,
            args.max_grad_norm,
        )
        peak_cuda_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            if device == 'cuda' else 0.0
        )
        update_avg_reward = float(np.mean([
            row['avg_reward'] for _, row, _ in pending_episodes
        ]))
        update_episode_count = len(pending_episodes)
        update_sample_count = len(update_buffer)

        for completed_episode, row, completed_buffer in pending_episodes:
            row.update(loss_metrics)
            row.update({
                'update_index': update_index,
                'episodes_in_update': update_episode_count,
                'update_sample_count': update_sample_count,
                'update_avg_reward': update_avg_reward,
                'training_device': device,
                'peak_cuda_memory_mb': peak_cuda_memory_mb,
            })
            log_rows.append(row)
            append_option_log(
                option_log_path, completed_episode, completed_buffer
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
        update_buffer = OptionAwareRolloutBuffer(
            args.gamma, args.gae_lambda
        )
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
    print(f'option rollout: {option_log_path}')
    print(f'convergence curves: {curve_path}')


if __name__ == '__main__':
    main()
