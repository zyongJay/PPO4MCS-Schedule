"""Train the centralized single-agent PPO dispatch policy.

Examples
--------
Graph message-passing treatment::

    .venv/bin/python demo/train_centralized.py \
        --encoder graph --device cuda \
        --output-dir training_results_centralized_graph

Matched node-only control::

    .venv/bin/python demo/train_centralized.py \
        --encoder nodeonly --device cuda \
        --output-dir training_results_centralized_nodeonly

No existing v10-v14 source file is modified by this entry point.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import world as world_module
from centralized_env import (
    CentralizedDispatchEnv,
    CentralizedRewardConfig,
)
from centralized_ppo import (
    CentralizedPPOAgent,
    CentralizedRolloutBuffer,
    CentralizedStep,
    ppo_update,
)
from config import MAX_STEPS_PER_EPISODE, TRACK_DATA_PATH


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / 'training_results_centralized'
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Centralized single-agent PPO for global MCS dispatch'
    )
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grid-rows', type=int, default=4)
    parser.add_argument('--grid-columns', type=int, default=4)
    parser.add_argument('--encoder', choices=('graph', 'nodeonly'), default='graph')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--graph-hidden-dim', type=int, default=32)
    parser.add_argument('--graph-layers', type=int, default=2)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--update-epochs', type=int, default=8)
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--episodes-per-update', type=int, default=4)
    parser.add_argument('--reward-success', type=float, default=1.0)
    parser.add_argument('--reward-failure', type=float, default=1.0)
    parser.add_argument('--reward-profit', type=float, default=0.01)
    parser.add_argument('--reward-dispatch-cost', type=float, default=0.01)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    return parser.parse_args(argv)


def resolve_device(requested: str) -> str:
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA requested but torch.cuda.is_available() is False'
        )
    if requested == 'cuda' or (
        requested == 'auto' and torch.cuda.is_available()
    ):
        return 'cuda'
    return 'cpu'


def reward_config(args) -> CentralizedRewardConfig:
    return CentralizedRewardConfig(
        success=args.reward_success,
        failure=args.reward_failure,
        realised_profit=args.reward_profit,
        dispatch_cost=args.reward_dispatch_cost,
    )


def make_env(args, seed: int) -> CentralizedDispatchEnv:
    return CentralizedDispatchEnv(
        seed=seed,
        grid_rows=args.grid_rows,
        grid_columns=args.grid_columns,
        max_steps=args.max_steps,
        reward_config=reward_config(args),
        verbose=False,
    )


def collect_episode(
    agent: CentralizedPPOAgent,
    args,
    episode: int,
) -> tuple[Dict, CentralizedRolloutBuffer]:
    scenario_seed = int(args.seed + episode - 1)
    env = make_env(args, scenario_seed)
    observation = env.reset()
    buffer = CentralizedRolloutBuffer(args.gamma, args.gae_lambda)
    total_reward = 0.0
    total_decisions = 0
    decision_steps = 0
    total_dispatch_distance = 0.0
    total_realised_profit = 0.0
    total_completed_events = 0
    total_failure_events = 0
    last_info: Dict = {}

    for _step in range(args.max_steps):
        policy = agent.act(observation, deterministic=False)
        next_observation, reward, done, info = env.step(policy['actions'])
        has_decision = policy['decision_count'] > 0
        buffer.add(CentralizedStep(
            graph=np.asarray(observation['graph'], np.float32).copy(),
            pair_features=np.asarray(
                observation['pair_features'], np.float32
            ).copy(),
            action_mask=np.asarray(
                observation['action_mask'], bool
            ).copy(),
            actions=np.asarray(policy['actions'], np.int64).copy(),
            order=np.asarray(policy['order'], np.int64).copy(),
            old_log_prob=float(policy['log_prob']),
            value=float(policy['value']),
            reward=float(reward),
            done=bool(done),
            has_decision=bool(has_decision),
        ))
        total_reward += float(reward)
        total_decisions += int(policy['decision_count'])
        decision_steps += int(has_decision)
        total_dispatch_distance += float(
            info.get('dispatch_distance_km_step', 0.0)
        )
        total_realised_profit += float(
            info.get('realised_service_profit_step', 0.0)
        )
        total_completed_events += int(
            info.get('completed_service_count_step', 0)
        )
        total_failure_events += int(info.get('new_failure_count_step', 0))
        observation = next_observation
        last_info = info
        if done:
            break

    last_value = 0.0 if buffer.steps[-1].done else agent.value(
        observation['graph']
    )
    buffer.finish(last_value)
    completed = int(last_info.get('completed_ev_count', 0))
    failures = int(last_info.get('failed_ev_count', 0))
    denominator = max(completed + failures, 1)
    population = max(len(env.world.EVs), 1)
    row = {
        'episode': int(episode),
        'scenario_seed': scenario_seed,
        'steps': int(len(buffer.steps)),
        'episode_reward': float(total_reward),
        'avg_step_reward': float(total_reward / max(len(buffer.steps), 1)),
        'decision_step_count': int(decision_steps),
        'joint_mcs_decision_count': int(total_decisions),
        'avg_mcs_decisions_per_decision_step': float(
            total_decisions / max(decision_steps, 1)
        ),
        'completed_ev_count': completed,
        'failed_ev_count': failures,
        'unresolved_ev_count': int(max(population - completed - failures, 0)),
        'completed_ev_population_ratio': float(completed / population),
        'completed_ev_ratio_resolved': float(completed / denominator),
        'completed_service_event_count': int(total_completed_events),
        'failure_event_count': int(total_failure_events),
        'realised_service_profit': float(total_realised_profit),
        'dispatch_distance_km': float(total_dispatch_distance),
        'total_mcs_profit_accounting': float(
            last_info.get('total_mcs_profit', 0.0)
        ),
        'total_fcs_profit_accounting': float(
            last_info.get('total_fcs_profit', 0.0)
        ),
        **env.graph_builder.metrics(),
    }
    return row, buffer


def write_rows(path: Path, rows: List[Dict]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError('episodes and max-steps must be positive')
    if args.episodes_per_update <= 0 or args.minibatch_size <= 0:
        raise ValueError('update batching parameters must be positive')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'training_log.csv'
    config_path = output_dir / 'training_config.json'
    latest_path = output_dir / 'latest_model.pt'
    best_path = output_dir / 'best_model.pt'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision('high')

    template_env = make_env(args, args.seed)
    template_observation = template_env.reset()
    agent = CentralizedPPOAgent(
        dispatch_points=template_observation['dispatch_points'],
        encoder_mode=args.encoder,
        hidden_dim=args.hidden_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_layers=args.graph_layers,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        device=device,
    )

    rows: List[Dict] = []
    start_episode = 1
    best_completed_ratio = -float('inf')
    best_realised_profit = -float('inf')
    if args.resume is not None:
        metadata = agent.load(args.resume.resolve(), load_optimizers=True)
        start_episode = int(metadata.get('episode', 0)) + 1
        best_completed_ratio = float(metadata.get(
            'best_completed_population_ratio',
            metadata.get('best_completed_ratio', -float('inf')),
        ))
        best_realised_profit = float(metadata.get(
            'best_realised_profit', -float('inf')
        ))
        if log_path.is_file():
            import pandas as pd
            rows = pd.read_csv(log_path).to_dict('records')

    run_config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'resolved_device': device,
        'cuda_device_name': (
            torch.cuda.get_device_name(0) if device == 'cuda' else ''
        ),
        'architecture': CentralizedPPOAgent.ARCHITECTURE,
        'policy_semantics': 'one_joint_action_for_all_eligible_mcs_per_step',
        'task_semantics': (
            'dispatching_charging_recharging_mcs_excluded_from_actor'
        ),
        'step_order': 'graph_then_schedule_then_progress_then_match_then_reward',
        'reward_timing': 'physical_outcomes_at_end_of_fixed_step',
        'gae_timing': 'ordinary_fixed_step_gamma_no_option_gamma_power_duration',
        'dispatch_point_count': int(template_env.dispatch_point_count),
        'stay_action_index': int(template_env.stay_action),
        'graph_built_once_per_physical_state': True,
        'actor_graph_rebuilt_during_autoregressive_action': False,
        'manual_spatial_reward': False,
    }
    config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    update_buffer = CentralizedRolloutBuffer(args.gamma, args.gae_lambda)
    pending_rows: List[Dict] = []
    final_episode = start_episode + args.episodes - 1
    for episode in range(start_episode, final_episode + 1):
        row, episode_buffer = collect_episode(agent, args, episode)
        update_buffer.extend(episode_buffer)
        pending_rows.append(row)
        update_due = bool(
            len(pending_rows) >= args.episodes_per_update
            or episode == final_episode
        )
        if not update_due:
            continue

        losses = ppo_update(
            agent,
            update_buffer,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
        )
        for pending in pending_rows:
            pending.update(losses)
            pending['encoder_mode'] = args.encoder
            rows.append(pending)

        recent = pending_rows
        completed_ratio = float(np.mean([
            item['completed_ev_population_ratio'] for item in recent
        ]))
        realised_profit = float(np.mean([
            item['realised_service_profit'] for item in recent
        ]))
        is_best = bool(
            completed_ratio > best_completed_ratio + 1e-12
            or (
                abs(completed_ratio - best_completed_ratio) <= 1e-12
                and realised_profit > best_realised_profit
            )
        )
        if is_best:
            best_completed_ratio = completed_ratio
            best_realised_profit = realised_profit
        metadata = {
            'episode': int(episode),
            'best_completed_population_ratio': float(best_completed_ratio),
            'best_realised_profit': float(best_realised_profit),
        }
        agent.save(latest_path, metadata)
        if is_best:
            agent.save(best_path, metadata)
        if (
            args.checkpoint_interval > 0
            and episode % args.checkpoint_interval == 0
        ):
            agent.save(
                output_dir / f'checkpoint_episode_{episode}.pt', metadata
            )
        write_rows(log_path, rows)
        print(
            f'episode={episode} encoder={args.encoder} '
            f'reward={np.mean([x["episode_reward"] for x in recent]):.4f} '
            f'completed_ratio={completed_ratio:.4f} '
            f'realised_profit={realised_profit:.2f} '
            f'actor_loss={losses["actor_loss"]:.5f} '
            f'critic_loss={losses["critic_loss"]:.5f}'
        )
        update_buffer.clear()
        pending_rows = []


if __name__ == '__main__':
    main()
