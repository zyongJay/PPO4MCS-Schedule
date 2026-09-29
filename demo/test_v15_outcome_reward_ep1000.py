"""Evaluate the v15_outcome_reward episode-1000 checkpoint on fixed seeds 1050--1099.

Protocol is identical to ``demo/test_v15.py`` (the script that produced
``test_results_v15_outcome_reward/``): same v15 signed-graph environment, same
deterministic greedy Low decisions, same fixed strict 40 kWh High rule, same
paired seeds 1050--1099, same ``max_steps=200``.  Random and v10_onlylow rows
are read from the validated paired baseline file rather than re-executed.

The only difference from ``test_v15.py`` is that a single checkpoint is
evaluated (instead of a best/final pair), so the episode-1000 model can be
scored on exactly the same fixed scenarios.
"""
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
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v15_outcome_reward' / 'model_episode_1000.pt'
)
DEFAULT_BASELINE_PATH = PROJECT_DIR / 'test_results_v12' / 'v12_scenarios.csv'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v15_outcome_reward_ep1000'
DEFAULT_SEEDS = tuple(range(1050, 1100))
POLICY_NAME = 'v15_ep1000'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the v15_outcome_reward episode-1000 checkpoint'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--policy-name', default=POLICY_NAME)
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


def evaluate_v15(policy_name, checkpoint, metadata, agent, seed, max_steps):
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
    checkpoint_path = args.checkpoint.expanduser().resolve()
    baseline_path = args.baseline_scenarios.expanduser().resolve()
    for path in (checkpoint_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(f'required file not found: {path}')

    device = resolve_device(args.device)
    torch.set_num_threads(int(args.torch_threads))
    agent, metadata = V15SignedGraphMAPPOAgent.from_checkpoint(
        checkpoint_path, device=device
    )
    agent.eval()
    agent.deterministic_low_actions = True

    rows = []
    print(
        f'device={device} checkpoint={checkpoint_path.name} '
        f'policy={args.policy_name} paired_scenarios={len(seeds)} '
        f'max_steps={args.max_steps}'
    )
    for seed in seeds:
        row = evaluate_v15(
            args.policy_name, checkpoint_path, metadata, agent, seed,
            int(args.max_steps),
        )
        rows.append(row)
        print(
            f'{args.policy_name} seed={seed} '
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
        (args.policy_name, 'v10_onlylow'),
        (args.policy_name, 'Random'),
        ('v10_onlylow', 'Random'),
    ))
    summary = build_summary(scenarios)
    if args.no_save:
        print(summary.to_string(index=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.policy_name
    scenarios.to_csv(output_dir / f'{stem}_scenarios.csv', index=False)
    summary.to_csv(output_dir / f'{stem}_summary.csv', index=False)
    comparisons.to_csv(
        output_dir / f'{stem}_paired_comparison.csv', index=False
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
        'policy_name': args.policy_name,
        'checkpoint': str(checkpoint_path),
        'checkpoint_metadata': metadata,
        'baseline_path': str(baseline_path),
        'baseline_reexecuted': False,
        'protocol': (
            'Identical to demo/test_v15.py: v15 signed-graph environment, '
            'deterministic greedy Low decisions, fixed strict 40 kWh High '
            'rule, paired seeds 1050-1099, max_steps=200.  Only the number of '
            'evaluated checkpoints differs (single vs best/final pair).'
        ),
        'fairness': (
            'V15 changes graph construction and graph encoders only.  The '
            'simulator, reward, candidate set and fixed High rule are '
            'unchanged from v14/v10-onlylow.'
        ),
    }
    (output_dir / f'{stem}_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
