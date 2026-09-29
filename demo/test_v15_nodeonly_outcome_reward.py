"""Evaluate multiple v15 node-only outcome-reward checkpoints together.

The evaluation uses the paired fixed scenarios 1050--1099 by default.  Each
checkpoint receives an unambiguous policy name derived from its checkpoint
role/metadata, so a best checkpoint and numbered checkpoints can coexist in
the same scenario, summary and paired-comparison outputs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch

from test import build_summary, resolve_device
from test_actor import build_paired_comparison
import test_v12 as v12_test
from test_v14 import load_baselines
from train_v15_outcome_reward import V15OutcomeRewardEnv
from v15_nodeonly import V15NodeOnlyMAPPOAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINTS = (
    PROJECT_DIR / 'training_results_v15_nodeonly_outcome_reward' / 'best_model.pt',
    PROJECT_DIR / 'training_results_v15_nodeonly_outcome_reward'
    / 'model_episode_1000.pt',
)
DEFAULT_BASELINE_PATH = PROJECT_DIR / 'test_results_v12' / 'v12_scenarios.csv'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v15_nodeonly_outcome_reward'
DEFAULT_SEEDS = tuple(range(1050, 1100))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate one or more v15 node-only outcome-reward checkpoints '
            'on paired fixed scenarios'
        )
    )
    parser.add_argument(
        '--checkpoints',
        type=Path,
        nargs='+',
        default=list(DEFAULT_CHECKPOINTS),
        help='One or more node-only outcome-reward .pt checkpoints.',
    )
    parser.add_argument(
        '--baseline-scenarios', type=Path, default=DEFAULT_BASELINE_PATH,
        help='Existing paired Random and v10_onlylow scenario records.',
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


def policy_name_for_checkpoint(path: Path, metadata: dict) -> str:
    """Create a stable, human-readable label without a hard-coded episode."""
    if path.name == 'best_model.pt':
        return 'v15_nodeonly_outcome_reward_best'
    episode = metadata.get('episode')
    if episode is not None:
        return f'v15_nodeonly_outcome_reward_episode{int(episode)}'
    safe_stem = ''.join(
        character if character.isalnum() else '_'
        for character in path.stem
    ).strip('_')
    return f'v15_nodeonly_outcome_reward_{safe_stem}'


def evaluate_checkpoint(policy_name, checkpoint, metadata, agent, seed, max_steps):
    """Run one node-only policy through the shared fixed-scenario harness."""
    original_environment = v12_test.V12MultiAgentEnv
    v12_test.V12MultiAgentEnv = V15OutcomeRewardEnv
    try:
        row = v12_test.evaluate_scenario(
            policy_name=policy_name,
            # The shared scenario harness dispatches the supplied agent under
            # its legacy v12 branch; the externally reported type is set
            # below to the precise v15 node-only outcome-reward label.
            policy_type='v12',
            scenario_seed=seed,
            max_steps=max_steps,
            checkpoint_path=checkpoint,
            checkpoint_metadata=metadata,
            agent=agent,
        )
    finally:
        v12_test.V12MultiAgentEnv = original_environment
        V15OutcomeRewardEnv.instances.pop(int(seed), None)
    row['policy_type'] = (
        'v15_nodeonly_outcome_reward_no_signed_message_passing'
    )
    row['low_mode'] = 'v15_nodeonly_deterministic_greedy'
    agent.low_actor.pop_diagnostics()
    return row


def main():
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('scenario seeds must be non-empty and unique')
    if args.max_steps <= 0 or args.torch_threads <= 0:
        raise ValueError('max-steps and torch-threads must be positive')

    checkpoint_paths = [
        path.expanduser().resolve() for path in args.checkpoints
    ]
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise ValueError('checkpoint paths must be unique')
    baseline_path = args.baseline_scenarios.expanduser().resolve()
    for path in (*checkpoint_paths, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(f'required file not found: {path}')

    device = resolve_device(args.device)
    torch.set_num_threads(int(args.torch_threads))
    experiments = []
    seen_names = set()
    for path in checkpoint_paths:
        agent, metadata = V15NodeOnlyMAPPOAgent.from_checkpoint(
            path, device=device
        )
        agent.eval()
        agent.deterministic_low_actions = True
        policy_name = policy_name_for_checkpoint(path, metadata)
        if policy_name in seen_names:
            raise ValueError(
                f'checkpoint labels are not unique: {policy_name}'
            )
        seen_names.add(policy_name)
        experiments.append((policy_name, path, metadata, agent))

    rows = []
    print(
        f'device={device} checkpoints={len(experiments)} '
        f'paired_scenarios={len(seeds)} max_steps={args.max_steps}'
    )
    for seed in seeds:
        for policy_name, path, metadata, agent in experiments:
            row = evaluate_checkpoint(
                policy_name, path, metadata, agent, seed, args.max_steps
            )
            rows.append(row)
            print(
                f'{policy_name} seed={seed} '
                f'success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_success={row["successful_ev_mcs_count"]} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f}'
            )

    checkpoint_rows = pd.DataFrame(rows)
    baseline_rows = load_baselines(baseline_path, seeds)
    scenarios = pd.concat(
        [checkpoint_rows, baseline_rows], ignore_index=True, sort=False
    ).sort_values(['scenario_seed', 'policy_name']).reset_index(drop=True)
    expected_rows = (len(experiments) + 2) * len(seeds)
    if len(scenarios) != expected_rows:
        raise RuntimeError(
            f'unexpected scenario row count {len(scenarios)} != {expected_rows}'
        )

    policy_names = [item[0] for item in experiments]
    comparison_names = [*policy_names, 'v10_onlylow', 'Random']
    comparisons = build_paired_comparison(
        scenarios, list(combinations(comparison_names, 2))
    )
    summary = build_summary(scenarios)
    if args.no_save:
        print(summary.to_string(index=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(
        output_dir / 'v15_nodeonly_outcome_reward_scenarios.csv', index=False
    )
    summary.to_csv(
        output_dir / 'v15_nodeonly_outcome_reward_summary.csv', index=False
    )
    comparisons.to_csv(
        output_dir / 'v15_nodeonly_outcome_reward_paired_comparison.csv',
        index=False,
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
        'reward_design': 'v14_outcome_only_no_manual_spatial_shaping',
        'checkpoints': {
            policy_name: {'path': str(path), 'metadata': metadata}
            for policy_name, path, metadata, _ in experiments
        },
        'baseline_path': str(baseline_path),
        'baseline_reexecuted': False,
        'ablation': (
            'All packed positive/negative signed-edge bits are cleared before '
            'every encoder forward; nodes, candidate actions, heads, reward '
            'and the fixed strict 40 kWh High rule are identical to v15.'
        ),
        'comparison_protocol': (
            'Every supplied checkpoint is evaluated on each identical fixed '
            'seed. Random and v10_onlylow rows are loaded from the validated '
            'paired baseline scenario file.'
        ),
    }
    (output_dir / 'v15_nodeonly_outcome_reward_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
