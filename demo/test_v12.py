"""Paired v12 Competition-HG / v10_onlylow / Random evaluation.

Each policy receives the same 50 seeded scenarios.  The v12 environment only
augments policy observations with its competition hypergraph; it does not alter
the simulator, matching, or rewards.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

import world as world_module
from config import MAX_STEPS_PER_EPISODE, TRACK_DATA_PATH
from core import MCS
from environment import MultiAgentEnv
from test import build_summary, iev_track_action, mean_attribute, remember_charge_providers, resolve_device, load_rl_agent
from test_actor import (
    CheckpointLowAblationPolicy,
    InstrumentedRandomDecisionPolicy,
    LOW_FEATURE_NAMES,
    METRIC_DIRECTIONS,
    build_paired_comparison,
)
from train_v12 import V12MultiAgentEnv
from v12_hypergraph import V12CompetitionHypergraphAgent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_V12_CHECKPOINT = PROJECT_DIR / 'training_results_v12' / 'best_model.pt'
DEFAULT_ONLYLOW_CHECKPOINT = PROJECT_DIR / 'training_results_v10_onlylow' / 'best_model.pt'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v12'
DEFAULT_SEEDS = tuple(range(1050, 1100))
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


class V12CheckpointPolicy(CheckpointLowAblationPolicy):
    """v12 Low Actor inference with its soft occupancy and reservation pass."""

    def __init__(self, agent: V12CompetitionHypergraphAgent, seed: int):
        super().__init__(agent, 'learned', seed, high_mode='threshold')
        self.agent.deterministic_low_actions = True

    @torch.no_grad()
    def _select_low_actions(self, observations: Sequence[Dict]) -> List[int]:
        decisions = self.agent.select_low_actions_batch(list(observations))
        selected = []
        for decision, observation in zip(decisions, observations):
            index = int(decision['low_action'])
            if not bool(observation['low_candidate_mask'][index]):
                raise RuntimeError('v12 Low Actor selected an illegal candidate')
            self.low_selected_ranks.append(index + 1)
            self.low_selected_features.append(np.asarray(
                observation['low_candidates'][index], dtype=np.float64
            ).copy())
            selected.append(index)
        return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate v12, v10_onlylow and Random on paired scenarios'
    )
    parser.add_argument('--v12-checkpoint', type=Path, default=DEFAULT_V12_CHECKPOINT)
    parser.add_argument('--onlylow-checkpoint', type=Path, default=DEFAULT_ONLYLOW_CHECKPOINT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_episode(path: Path, metadata: Dict) -> int | str:
    episode = metadata.get('episode', '')
    if episode:
        return episode
    match = re.search(r'(\d+)', path.stem)
    return int(match.group(1)) if match else ''


def evaluate_scenario(
    policy_name: str,
    policy_type: str,
    scenario_seed: int,
    max_steps: int,
    checkpoint_path: Path | None,
    checkpoint_metadata: Dict,
    agent,
) -> Dict:
    """Run one policy in a seeded scenario and collect common business metrics."""
    seed_everything(scenario_seed)
    if policy_type == 'v12':
        env = V12MultiAgentEnv(scenario_seed)
        policy = V12CheckpointPolicy(agent, scenario_seed)
    elif policy_type == 'v10_onlylow':
        env = MultiAgentEnv(scenario_seed)
        policy = CheckpointLowAblationPolicy(
            agent, 'learned', scenario_seed, high_mode='threshold'
        )
    elif policy_type == 'full_random':
        env = MultiAgentEnv(scenario_seed)
        policy = InstrumentedRandomDecisionPolicy(scenario_seed)
    else:
        raise ValueError(f'unknown policy_type: {policy_type}')
    env.world.verbose = False
    observations = env.reset()

    provider_by_ev_id: Dict[int, str] = {}
    remember_charge_providers(env.world.EVs, provider_by_ev_id)
    decision_times_ms: List[float] = []
    executed_steps = 0
    for _ in range(max_steps):
        acting_agents = list(env.world.agents)
        policy.synchronize()
        started = time.perf_counter_ns()
        action_n = policy.build_actions(env, acting_agents, observations)
        policy.synchronize()
        decision_times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if len(action_n) != len(acting_agents):
            raise RuntimeError(f'{policy_name}: action count does not match acting agents')
        observations, _, _, _ = env.step(action_n)
        executed_steps += 1
        remember_charge_providers(env.world.EVs, provider_by_ev_id)
        if env.world.get_done():
            break

    evs, mcss, fcss = list(env.world.EVs), list(env.world.MCSs), list(env.world.FCSs)
    successful_evs = [ev for ev in evs if ev.is_charged]
    success_count = len(successful_evs)
    failure_count = int(sum(ev.fail_charge for ev in evs))
    finished_count = success_count + failure_count
    unresolved_count = int(sum(
        not ev.is_normal and not ev.fail_charge and not ev.is_charged for ev in evs
    ))
    provider_counts = {'MCS': 0, 'FCS': 0}
    missing = []
    for ev in successful_evs:
        provider = provider_by_ev_id.get(int(ev.id), '')
        if provider in provider_counts:
            provider_counts[provider] += 1
        else:
            missing.append(int(ev.id))
    if missing:
        raise RuntimeError(f'successful EV lacks provider record: {missing[:10]}')

    success_denominator = max(success_count, 1)
    row = {
        'policy_name': policy_name,
        'policy_type': policy_type,
        'high_mode': 'threshold40' if policy_type != 'full_random' else 'full_random',
        'low_mode': 'competition_hypergraph' if policy_type == 'v12' else (
            'learned' if policy_type == 'v10_onlylow' else 'not_applicable'
        ),
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else '',
        'checkpoint_episode': checkpoint_episode(checkpoint_path, checkpoint_metadata) if checkpoint_path else '',
        'scenario_seed': int(scenario_seed),
        'random_low_seed': int(scenario_seed),
        'steps': int(executed_steps),
        'ev_charge_success_ratio': success_count / finished_count if finished_count else 0.0,
        'ev_population_success_ratio': success_count / len(evs) if evs else 0.0,
        'ev_success_count': int(success_count),
        'ev_failure_count': failure_count,
        'ev_unresolved_count': unresolved_count,
        'avg_mcs_profit': mean_attribute(mcss, 'total_profit'),
        'avg_mcs_idle_time_min': mean_attribute(mcss, 'total_idle_time_min'),
        'avg_fcs_profit': mean_attribute(fcss, 'total_profit'),
        'avg_fcs_idle_time_min': mean_attribute(fcss, 'total_idle_time_min'),
        'avg_ev_extra_distance_km': mean_attribute(evs, 'total_extra_dist_km'),
        'avg_ev_charging_delay_min': mean_attribute(evs, 'total_wait_time_min'),
        'avg_decision_time_ms': float(np.mean(decision_times_ms)),
        'decision_time_std_ms': float(np.std(decision_times_ms, ddof=0)),
        'decision_time_p95_ms': float(np.percentile(decision_times_ms, 95)),
        'successful_ev_mcs_count': int(provider_counts['MCS']),
        'successful_ev_fcs_count': int(provider_counts['FCS']),
        'successful_ev_mcs_share': provider_counts['MCS'] / success_denominator,
        'successful_ev_fcs_share': provider_counts['FCS'] / success_denominator,
        'broken_mcs_count': int(sum(mcs.is_broken for mcs in mcss)),
        'energy_stranded_mcs_count': int(sum(mcs.is_energy_stranded for mcs in mcss)),
        'unavailable_mcs_count': int(sum(mcs.is_broken or mcs.is_energy_stranded for mcs in mcss)),
    }
    row.update(policy.diagnostic_metrics())
    if policy_type == 'v12':
        row.update(env.metrics())
    return row


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.hidden_dim <= 0 or args.torch_threads <= 0:
        raise ValueError('max-steps, hidden-dim and torch-threads must be positive')
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError('scenario seeds must be unique')
    v12_path = args.v12_checkpoint.expanduser().resolve()
    onlylow_path = args.onlylow_checkpoint.expanduser().resolve()
    for path in (v12_path, onlylow_path):
        if not path.is_file():
            raise FileNotFoundError(f'checkpoint not found: {path}')
    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    v12_agent, v12_metadata = V12CompetitionHypergraphAgent.from_checkpoint(v12_path, device=device)
    v12_agent.eval()
    onlylow_agent, onlylow_metadata = load_rl_agent(onlylow_path, args.hidden_dim, device)
    max_steps = min(args.max_steps, MAX_STEPS_PER_EPISODE)
    experiments = (
        ('v12', 'v12', v12_path, v12_metadata, v12_agent),
        ('v10_onlylow', 'v10_onlylow', onlylow_path, onlylow_metadata, onlylow_agent),
        ('Random', 'full_random', None, {}, None),
    )
    rows = []
    print(f'device={device} paired_scenarios={len(args.seeds)} max_steps={max_steps}')
    for seed in args.seeds:
        for name, kind, path, metadata, agent in experiments:
            row = evaluate_scenario(name, kind, seed, max_steps, path, metadata, agent)
            rows.append(row)
            print(
                f'{name} seed={seed} success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_success={row["successful_ev_mcs_count"]} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f}'
            )
    scenarios = pd.DataFrame(rows)
    comparisons = build_paired_comparison(scenarios, (
        ('v12', 'v10_onlylow'), ('v12', 'Random'), ('v10_onlylow', 'Random'),
    ))
    summary = build_summary(scenarios)
    if args.no_save:
        print(summary.to_string(index=False))
        return
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / 'v12_scenarios.csv', index=False)
    summary.to_csv(output_dir / 'v12_summary.csv', index=False)
    comparisons.to_csv(output_dir / 'v12_paired_comparison.csv', index=False)
    config = {
        'evaluated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'seeds': sorted(int(seed) for seed in args.seeds),
        'paired_scenario_count': len(args.seeds),
        'max_steps': max_steps,
        'device': device,
        'torch_threads': args.torch_threads,
        'v12_checkpoint': str(v12_path),
        'v12_checkpoint_metadata': v12_metadata,
        'v10_onlylow_checkpoint': str(onlylow_path),
        'v10_onlylow_checkpoint_metadata': onlylow_metadata,
        'policy_semantics': {
            'v12': 'best checkpoint Low Actor, Competition-HG local observation and deterministic sequential reservation; fixed strict remain<40 kWh High option rule',
            'v10_onlylow': 'best checkpoint Low Actor with the same fixed strict remain<40 kWh High option rule',
            'Random': 'existing complete RandomDecisionPolicy',
        },
        'fairness': 'All policies use identical seeded simulator scenarios. v12 only augments its policy observation; simulator, matcher, and reward are unchanged.',
    }
    (output_dir / 'v12_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('\nSaved:', output_dir)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
