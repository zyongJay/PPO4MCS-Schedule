"""Evaluate the v15 node-only episode-300 checkpoint on fixed seeds 1050--1099.

The canonical v15 node slots, primitive node features, candidate actions,
Actor/Critic heads, reward and the fixed strict 40 kWh High rule are identical
to v15.  The only ablated operation is inter-node signed message passing: all
packed positive/negative edge bits are cleared before every encoder forward.

Random and v10_onlylow rows are read from the validated paired baseline file
rather than re-executed.
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
from v15_nodeonly import V15NodeOnlyMAPPOAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v15_nodeonly' / 'model_episode_300.pt'
)
DEFAULT_BASELINE_PATH = PROJECT_DIR / 'test_results_v12' / 'v12_scenarios.csv'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v15_nodeonly'
DEFAULT_SEEDS = tuple(range(1050, 1100))
POLICY_NAME = 'v15_nodeonly_episode300'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the v15 node-only episode-300 checkpoint'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        '--baseline-scenarios', type=Path, default=DEFAULT_BASELINE_PATH,
        help='已有的 Random 与 v10_onlylow 配对场景记录；不重新执行它们',
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


def evaluate_v15_nodeonly(
    policy_name, checkpoint, metadata, agent, seed, max_steps,
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
    row['policy_type'] = 'v15_nodeonly_no_signed_message_passing'
    row['low_mode'] = 'v15_nodeonly_deterministic_greedy'
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
    agent, metadata = V15NodeOnlyMAPPOAgent.from_checkpoint(
        checkpoint_path, device=device
    )
    agent.eval()
    agent.deterministic_low_actions = True

    rows = []
    print(
        f'device={device} checkpoint={checkpoint_path.name} '
        f'paired_scenarios={len(seeds)} max_steps={args.max_steps}'
    )
    for seed in seeds:
        row = evaluate_v15_nodeonly(
            POLICY_NAME, checkpoint_path, metadata, agent, seed,
            int(args.max_steps),
        )
        rows.append(row)
        print(
            f'{POLICY_NAME} seed={seed} '
            f'success={row["ev_charge_success_ratio"]:.4f} '
            f'mcs_success={row["successful_ev_mcs_count"]} '
            f'mcs_profit={row["avg_mcs_profit"]:.2f}'
        )

    nodeonly_rows = pd.DataFrame(rows)
    baseline_rows = load_baselines(baseline_path, seeds)
    scenarios = pd.concat(
        [nodeonly_rows, baseline_rows], ignore_index=True, sort=False
    ).sort_values(['scenario_seed', 'policy_name']).reset_index(drop=True)
    comparisons = build_paired_comparison(scenarios, (
        (POLICY_NAME, 'v10_onlylow'),
        (POLICY_NAME, 'Random'),
        ('v10_onlylow', 'Random'),
    ))
    summary = build_summary(scenarios)
    if args.no_save:
        print(summary.to_string(index=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / 'v15_nodeonly_scenarios.csv', index=False)
    summary.to_csv(output_dir / 'v15_nodeonly_summary.csv', index=False)
    comparisons.to_csv(
        output_dir / 'v15_nodeonly_paired_comparison.csv', index=False
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
        'v15_nodeonly_checkpoint': str(checkpoint_path),
        'v15_nodeonly_checkpoint_metadata': metadata,
        'baseline_path': str(baseline_path),
        'baseline_reexecuted': False,
        'ablation': (
            'All packed positive/negative signed-edge bits are cleared before '
            'every encoder forward; nodes, candidate actions, heads, reward '
            'and the fixed strict 40 kWh High rule are identical to v15.'
        ),
        'fairness': (
            'V15 node-only changes inter-node message passing only.  The '
            'simulator, reward, candidate action set and fixed High rule are '
            'unchanged from v15/v14/v10-onlylow.'
        ),
    }
    (output_dir / 'v15_nodeonly_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
