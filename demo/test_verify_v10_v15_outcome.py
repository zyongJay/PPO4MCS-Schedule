"""Paired verification for v10_onlylow and v15 outcome-reward checkpoints.

All requested policies are re-executed on every supplied seed.  The v15
policies use their signed-graph observation adapter; v10_onlylow retains its
original observation codec required by its checkpoint.  Both adapters share
the unchanged physical simulator, matcher, candidate set and fixed 40 kWh
High-option rule.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch

from test import build_summary, load_rl_agent, resolve_device
from test_actor import build_paired_comparison
import test_v12 as v12_test
from train_v15_outcome_reward import V15OutcomeRewardEnv
from v15_signed_graph import V15SignedGraphMAPPOAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_V10_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v10_onlylow' / 'model_episode_300.pt'
)
DEFAULT_V15_EP280_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v15_outcome_reward' / 'model_episode_280.pt'
)
DEFAULT_V15_EP1000_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v15_outcome_reward' / 'model_episode_1000.pt'
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_verify'
DEFAULT_SEEDS = tuple(range(2000, 2050))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Paired fixed-scenario verification for v10 and v15 outcome reward'
    )
    parser.add_argument('--v10-checkpoint', type=Path, default=DEFAULT_V10_CHECKPOINT)
    parser.add_argument('--v15-ep280-checkpoint', type=Path, default=DEFAULT_V15_EP280_CHECKPOINT)
    parser.add_argument('--v15-ep1000-checkpoint', type=Path, default=DEFAULT_V15_EP1000_CHECKPOINT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


def evaluate_v15(policy_name, checkpoint, metadata, agent, seed, max_steps):
    """Evaluate one v15 checkpoint through the established option harness."""
    original_environment = v12_test.V12MultiAgentEnv
    v12_test.V12MultiAgentEnv = V15OutcomeRewardEnv
    try:
        row = v12_test.evaluate_scenario(
            policy_name=policy_name,
            policy_type='v12',
            scenario_seed=int(seed),
            max_steps=int(max_steps),
            checkpoint_path=checkpoint,
            checkpoint_metadata=metadata,
            agent=agent,
        )
    finally:
        v12_test.V12MultiAgentEnv = original_environment
        V15OutcomeRewardEnv.instances.pop(int(seed), None)
    row['policy_type'] = 'v15_global_binary_signed_attention_graph'
    row['low_mode'] = 'v15_signed_graph_deterministic_greedy'
    agent.low_actor.pop_diagnostics()
    return row


def main():
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('seeds must be non-empty and unique')
    if args.max_steps <= 0 or args.torch_threads <= 0:
        raise ValueError('max-steps and torch-threads must be positive')

    v10_path = args.v10_checkpoint.expanduser().resolve()
    v15_ep280_path = args.v15_ep280_checkpoint.expanduser().resolve()
    v15_ep1000_path = args.v15_ep1000_checkpoint.expanduser().resolve()
    for path in (v10_path, v15_ep280_path, v15_ep1000_path):
        if not path.is_file():
            raise FileNotFoundError(f'checkpoint not found: {path}')

    device = resolve_device(args.device)
    torch.set_num_threads(int(args.torch_threads))
    v10_agent, v10_metadata = load_rl_agent(v10_path, 128, device)
    v15_ep280_agent, v15_ep280_metadata = V15SignedGraphMAPPOAgent.from_checkpoint(
        v15_ep280_path, device=device
    )
    v15_ep1000_agent, v15_ep1000_metadata = V15SignedGraphMAPPOAgent.from_checkpoint(
        v15_ep1000_path, device=device
    )
    for agent in (v10_agent, v15_ep280_agent, v15_ep1000_agent):
        agent.eval()
    for agent in (v15_ep280_agent, v15_ep1000_agent):
        agent.deterministic_low_actions = True

    experiments = (
        ('v10_onlylow_ep300', 'v10', v10_path, v10_metadata, v10_agent),
        ('v15_outcome_reward_ep280', 'v15', v15_ep280_path, v15_ep280_metadata, v15_ep280_agent),
        ('v15_outcome_reward_ep1000', 'v15', v15_ep1000_path, v15_ep1000_metadata, v15_ep1000_agent),
    )
    rows = []
    print(
        f'device={device} policies={len(experiments)} '
        f'paired_scenarios={len(seeds)} max_steps={args.max_steps}'
    )
    for seed in seeds:
        for name, kind, checkpoint, metadata, agent in experiments:
            if kind == 'v10':
                row = v12_test.evaluate_scenario(
                    policy_name=name,
                    policy_type='v10_onlylow',
                    scenario_seed=seed,
                    max_steps=int(args.max_steps),
                    checkpoint_path=checkpoint,
                    checkpoint_metadata=metadata,
                    agent=agent,
                )
            else:
                row = evaluate_v15(
                    name, checkpoint, metadata, agent, seed, args.max_steps
                )
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
    names = [item[0] for item in experiments]
    comparisons = build_paired_comparison(
        scenarios, list(combinations(names, 2))
    )
    summary = build_summary(scenarios)
    if args.no_save:
        print(summary.to_string(index=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / 'verify_scenarios.csv', index=False)
    summary.to_csv(output_dir / 'verify_summary.csv', index=False)
    comparisons.to_csv(output_dir / 'verify_paired_comparison.csv', index=False)
    config = {
        'evaluated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'seeds': sorted(seeds),
        'paired_scenario_count': len(seeds),
        'max_steps': int(args.max_steps),
        'device': device,
        'torch_threads': int(args.torch_threads),
        'checkpoints': {
            name: {'path': str(path), 'metadata': metadata}
            for name, _, path, metadata, _ in experiments
        },
        'protocol': (
            'All policies are re-executed on the identical supplied seeds. '
            'V15 uses its signed-graph observation adapter; v10_onlylow uses '
            'its original checkpoint-compatible observation adapter. Both '
            'share the unchanged physical simulator, matching, candidate set, '
            'option lifecycle and fixed strict 40 kWh High rule.'
        ),
        'v15_ep280_note': (
            'Episode 300 checkpoint does not exist; the user selected the '
            'nearest earlier checkpoint model_episode_280.pt.'
        ),
    }
    (output_dir / 'verify_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
