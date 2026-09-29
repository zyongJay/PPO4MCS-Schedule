"""Paired evaluation of four v14 graph/node-only checkpoints."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from test import build_summary, resolve_device
from test_actor import build_paired_comparison
from test_v14 import evaluate_v14
from v14_global_graph import V14GlobalGraphMAPPOAgent
from v14_nodeonly import V14NodeOnlyMAPPOAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v14_nodeonly'
DEFAULT_SEEDS = tuple(range(1050, 1100))
DEFAULT_CHECKPOINTS = {
    'v14_best': PROJECT_DIR / 'training_results_v14' / 'best_model.pt',
    'v14_episode300': (
        PROJECT_DIR / 'training_results_v14' / 'model_episode_300.pt'
    ),
    'v14_nodeonly_best': (
        PROJECT_DIR / 'training_results_v14_nodeonly' / 'best_model.pt'
    ),
    'v14_nodeonly_episode300': (
        PROJECT_DIR / 'training_results_v14_nodeonly'
        / 'model_episode_300.pt'
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate v14 graph and node-only best/final checkpoints'
    )
    parser.add_argument('--v14-best', type=Path, default=DEFAULT_CHECKPOINTS['v14_best'])
    parser.add_argument('--v14-final', type=Path, default=DEFAULT_CHECKPOINTS['v14_episode300'])
    parser.add_argument('--nodeonly-best', type=Path, default=DEFAULT_CHECKPOINTS['v14_nodeonly_best'])
    parser.add_argument('--nodeonly-final', type=Path, default=DEFAULT_CHECKPOINTS['v14_nodeonly_episode300'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_steps <= 0 or args.torch_threads <= 0:
        raise ValueError('max-steps and torch-threads must be positive')
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError('scenario seeds must be unique')
    paths = {
        'v14_best': args.v14_best.expanduser().resolve(),
        'v14_episode300': args.v14_final.expanduser().resolve(),
        'v14_nodeonly_best': args.nodeonly_best.expanduser().resolve(),
        'v14_nodeonly_episode300': args.nodeonly_final.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'checkpoint not found: {path}')

    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    agents = {}
    metadata = {}
    for name in ('v14_best', 'v14_episode300'):
        agents[name], metadata[name] = (
            V14GlobalGraphMAPPOAgent.from_checkpoint(
                paths[name], device=device
            )
        )
    for name in ('v14_nodeonly_best', 'v14_nodeonly_episode300'):
        agents[name], metadata[name] = V14NodeOnlyMAPPOAgent.from_checkpoint(
            paths[name], device=device
        )
    for agent in agents.values():
        agent.eval()
        agent.deterministic_low_actions = True

    rows = []
    print(
        f'device={device} checkpoints=4 paired_scenarios={len(seeds)} '
        f'max_steps={args.max_steps}'
    )
    for seed in seeds:
        for name in paths:
            row = evaluate_v14(
                name,
                paths[name],
                metadata[name],
                agents[name],
                seed,
                int(args.max_steps),
            )
            is_nodeonly = name.startswith('v14_nodeonly')
            row['policy_type'] = (
                'v14_nodeonly_no_message_passing'
                if is_nodeonly
                else 'v14_global_graph_local_aggregation'
            )
            row['low_mode'] = (
                'v14_nodeonly_deterministic_greedy'
                if is_nodeonly
                else 'v14_global_graph_deterministic_greedy'
            )
            rows.append(row)
            print(
                f'{name} seed={seed} '
                f'success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_success={row["successful_ev_mcs_count"]} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f} '
                f'target_conflict={row["target_conflict_rate"]:.4f}'
            )

    scenarios = pd.DataFrame(rows).sort_values(
        ['scenario_seed', 'policy_name']
    ).reset_index(drop=True)
    expected = len(seeds) * len(paths)
    if len(scenarios) != expected:
        raise RuntimeError(f'unexpected row count {len(scenarios)} != {expected}')
    pairs = (
        ('v14_best', 'v14_episode300'),
        ('v14_nodeonly_best', 'v14_nodeonly_episode300'),
        ('v14_best', 'v14_nodeonly_best'),
        ('v14_episode300', 'v14_nodeonly_episode300'),
        ('v14_best', 'v14_nodeonly_episode300'),
        ('v14_episode300', 'v14_nodeonly_best'),
    )
    comparisons = build_paired_comparison(scenarios, pairs)
    summary = build_summary(scenarios)
    if args.no_save:
        print(summary.to_string(index=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(
        output_dir / 'v14_nodeonly_scenarios.csv', index=False
    )
    summary.to_csv(output_dir / 'v14_nodeonly_summary.csv', index=False)
    comparisons.to_csv(
        output_dir / 'v14_nodeonly_paired_comparison.csv', index=False
    )
    config = {
        'evaluated_at': datetime.now().astimezone().isoformat(
            timespec='seconds'
        ),
        'seeds': sorted(seeds),
        'paired_scenario_count': len(seeds),
        'max_steps': int(args.max_steps),
        'device': device,
        'torch_threads': int(args.torch_threads),
        'checkpoints': {
            name: {
                'path': str(paths[name]),
                'metadata': metadata[name],
            }
            for name in paths
        },
        'policy_semantics': {
            'v14_best': 'global graph local message passing, selected best checkpoint',
            'v14_episode300': 'global graph local message passing, episode 300',
            'v14_nodeonly_best': 'identical v14 nodes/heads with all relation matrices zeroed, selected best checkpoint',
            'v14_nodeonly_episode300': 'identical v14 nodes/heads with all relation matrices zeroed, episode 300',
        },
        'common_conditions': (
            'Identical simulator, fixed strict 40 kWh High rule, reward, '
            'candidate action set, deterministic greedy Low decisions, '
            'max steps and paired scenario seeds.'
        ),
    }
    (output_dir / 'v14_nodeonly_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    compact = [
        'policy_name', 'checkpoint_episode', 'ev_charge_success_ratio',
        'ev_success_count', 'ev_failure_count', 'avg_mcs_profit',
        'successful_ev_mcs_count', 'successful_ev_fcs_count',
        'successful_ev_mcs_share', 'avg_mcs_idle_time_min',
        'avg_ev_charging_delay_min', 'target_conflict_rate',
        'spatial_conflict_rate',
    ]
    print(summary[[c for c in compact if c in summary]].to_string(index=False))


if __name__ == '__main__':
    main()
