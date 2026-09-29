"""Paired evaluation for v14 trained with outcome-only rewards."""
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


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v14_outcome_reward'
DEFAULT_SEEDS = tuple(range(1050, 1100))
DEFAULT_BEST = PROJECT_DIR / 'training_results_v14_outcome_reward' / 'best_model.pt'
DEFAULT_FINAL = (
    PROJECT_DIR / 'training_results_v14_outcome_reward' / 'model_episode_600.pt'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate v14 outcome-reward best and episode-600 models'
    )
    parser.add_argument('--best-checkpoint', type=Path, default=DEFAULT_BEST)
    parser.add_argument('--final-checkpoint', type=Path, default=DEFAULT_FINAL)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_steps <= 0 or args.torch_threads <= 0:
        raise ValueError('max-steps and torch-threads must be positive')
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError('scenario seeds must be unique')

    paths = {
        'v14_outcome_reward_best': args.best_checkpoint.expanduser().resolve(),
        'v14_outcome_reward_episode600': args.final_checkpoint.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'checkpoint not found: {path}')

    device = resolve_device(args.device)
    torch.set_num_threads(int(args.torch_threads))
    agents, metadata = {}, {}
    for name, path in paths.items():
        agents[name], metadata[name] = V14GlobalGraphMAPPOAgent.from_checkpoint(
            path, device=device
        )
        agents[name].eval()
        agents[name].deterministic_low_actions = True

    rows = []
    print(
        f'device={device} checkpoints=2 paired_scenarios={len(seeds)} '
        f'max_steps={args.max_steps}'
    )
    for seed in seeds:
        for name, path in paths.items():
            row = evaluate_v14(
                name, path, metadata[name], agents[name], seed,
                int(args.max_steps),
            )
            row['policy_type'] = 'v14_global_graph_outcome_reward'
            row['low_mode'] = 'deterministic_greedy_fixed_40kwh_high_rule'
            rows.append(row)
            print(
                f'{name} seed={seed} '
                f'success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_success={row["successful_ev_mcs_count"]} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f}'
            )

    scenarios = pd.DataFrame(rows).sort_values(
        ['scenario_seed', 'policy_name']
    ).reset_index(drop=True)
    expected = len(seeds) * len(paths)
    if len(scenarios) != expected:
        raise RuntimeError(f'unexpected row count {len(scenarios)} != {expected}')
    summary = build_summary(scenarios)
    comparison = build_paired_comparison(scenarios, [(
        'v14_outcome_reward_best', 'v14_outcome_reward_episode600',
    )])

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(
        output_dir / 'v14_outcome_reward_scenarios.csv', index=False
    )
    summary.to_csv(
        output_dir / 'v14_outcome_reward_summary.csv', index=False
    )
    comparison.to_csv(
        output_dir / 'v14_outcome_reward_paired_comparison.csv', index=False
    )
    config = {
        'evaluated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'seeds': sorted(seeds),
        'paired_scenario_count': len(seeds),
        'max_steps': int(args.max_steps),
        'device': device,
        'torch_threads': int(args.torch_threads),
        'reward_design': 'v14_outcome_only_no_manual_spatial_shaping',
        'checkpoints': {
            name: {'path': str(path), 'metadata': metadata[name]}
            for name, path in paths.items()
        },
        'common_conditions': (
            'Same v14 simulator and canonical global graph, strict fixed '
            '40 kWh High rule, unchanged candidate actions, deterministic '
            'greedy Low decisions, max_steps=200 and paired seeds 1050-1099.'
        ),
        'baseline_reexecuted': False,
    }
    (output_dir / 'v14_outcome_reward_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
