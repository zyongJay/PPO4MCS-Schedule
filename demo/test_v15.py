"""Evaluate v15 best/final checkpoints on fixed seeds 1050--1099."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from test import build_summary, resolve_device
from test_actor import build_paired_comparison
import test_v12 as v12_test
from test_v14 import load_baselines
from train_v15 import V15MultiAgentEnv
from v15_signed_graph import V15SignedGraphMAPPOAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BEST_CHECKPOINT = PROJECT_DIR / 'training_results_v15' / 'best_model.pt'
DEFAULT_FINAL_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v15' / 'model_episode_300.pt'
)
DEFAULT_BASELINE_PATH = PROJECT_DIR / 'test_results_v12' / 'v12_scenarios.csv'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v15'
DEFAULT_SEEDS = tuple(range(1050, 1100))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate v15 best and final checkpoints'
    )
    parser.add_argument(
        '--best-checkpoint', type=Path, default=DEFAULT_BEST_CHECKPOINT
    )
    parser.add_argument(
        '--final-checkpoint', type=Path, default=DEFAULT_FINAL_CHECKPOINT
    )
    parser.add_argument(
        '--baseline-scenarios', type=Path, default=DEFAULT_BASELINE_PATH
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS)
    )
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto'
    )
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


def evaluate_v15(
    policy_name,
    checkpoint,
    metadata,
    agent,
    seed,
    max_steps,
):
    original_environment = v12_test.V12MultiAgentEnv
    v12_test.V12MultiAgentEnv = V15MultiAgentEnv
    try:
        row = v12_test.evaluate_scenario(
            policy_name=policy_name,
            policy_type='v12',
            scenario_seed=seed,
            max_steps=max_steps,
            checkpoint_path=checkpoint,
            checkpoint_metadata=metadata,
            agent=agent,
        )
    finally:
        v12_test.V12MultiAgentEnv = original_environment
        V15MultiAgentEnv.instances.pop(int(seed), None)
    row['policy_type'] = 'v15_global_binary_signed_attention_graph'
    row['low_mode'] = 'v15_signed_graph_deterministic_greedy'
    agent.low_actor.pop_diagnostics()
    return row


def main():
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError('scenario seeds must be unique')
    best_path = args.best_checkpoint.expanduser().resolve()
    final_path = args.final_checkpoint.expanduser().resolve()
    baseline_path = args.baseline_scenarios.expanduser().resolve()
    for path in (best_path, final_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(f'required file not found: {path}')

    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    best_agent, best_metadata = V15SignedGraphMAPPOAgent.from_checkpoint(
        best_path, device=device
    )
    final_agent, final_metadata = V15SignedGraphMAPPOAgent.from_checkpoint(
        final_path, device=device
    )
    for agent in (best_agent, final_agent):
        agent.eval()
        agent.deterministic_low_actions = True
    experiments = (
        ('v15_best', best_path, best_metadata, best_agent),
        ('v15_final', final_path, final_metadata, final_agent),
    )

    rows = []
    print(
        f'device={device} v15_checkpoints=2 paired_scenarios={len(seeds)} '
        f'max_steps={args.max_steps}'
    )
    for seed in seeds:
        for name, path, metadata, agent in experiments:
            row = evaluate_v15(
                name, path, metadata, agent, seed, int(args.max_steps)
            )
            rows.append(row)
            print(
                f'{name} seed={seed} '
                f'success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_success={row["successful_ev_mcs_count"]} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f}'
            )

    v15_rows = pd.DataFrame(rows)
    baseline_rows = load_baselines(baseline_path, seeds)
    scenarios = pd.concat(
        [v15_rows, baseline_rows], ignore_index=True, sort=False
    ).sort_values(['scenario_seed', 'policy_name']).reset_index(drop=True)
    comparisons = build_paired_comparison(scenarios, (
        ('v15_best', 'v15_final'),
        ('v15_best', 'v10_onlylow'),
        ('v15_best', 'Random'),
        ('v15_final', 'v10_onlylow'),
        ('v15_final', 'Random'),
        ('v10_onlylow', 'Random'),
    ))
    summary = build_summary(scenarios)
    if args.no_save:
        print(summary.to_string(index=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / 'v15_scenarios.csv', index=False)
    summary.to_csv(output_dir / 'v15_summary.csv', index=False)
    comparisons.to_csv(
        output_dir / 'v15_paired_comparison.csv', index=False
    )
    config = {
        'evaluated_at': datetime.now().astimezone().isoformat(
            timespec='seconds'
        ),
        'seeds': sorted(seeds),
        'paired_scenario_count': len(seeds),
        'max_steps': int(args.max_steps),
        'device': device,
        'v15_best_checkpoint': str(best_path),
        'v15_best_checkpoint_metadata': best_metadata,
        'v15_final_checkpoint': str(final_path),
        'v15_final_checkpoint_metadata': final_metadata,
        'baseline_path': str(baseline_path),
        'baseline_reexecuted': False,
        'fairness': (
            'V15 changes graph construction and graph encoders only.  The '
            'simulator, reward, candidate set and fixed High rule are '
            'unchanged from v14/v10-onlylow.'
        ),
    }
    (output_dir / 'v15_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
