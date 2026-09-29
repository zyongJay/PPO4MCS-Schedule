"""Evaluate v14 best/final checkpoints on seeds 1050--1099.

The two v14 checkpoints are executed with the same deterministic Low action
selection and fixed strict 40 kWh High threshold.  Existing v10_onlylow and
Random rows are read rather than rerun.  The requested
``test_results_v10_onlylow`` file currently contains seeds 1001--1050, so the
validated 1050--1099 baseline rows are loaded from ``test_results_v12``.
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
from train_v14 import V14MultiAgentEnv
from v14_global_graph import V14GlobalGraphMAPPOAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BEST_CHECKPOINT = PROJECT_DIR / 'training_results_v14' / 'best_model.pt'
DEFAULT_FINAL_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v14' / 'model_episode_300.pt'
)
REQUESTED_BASELINE_PATH = (
    PROJECT_DIR / 'test_results_v10_onlylow' / 'actor_scenarios.csv'
)
VALIDATED_BASELINE_PATH = (
    PROJECT_DIR / 'test_results_v12' / 'v12_scenarios.csv'
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v14'
DEFAULT_SEEDS = tuple(range(1050, 1100))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate v14 best and episode-300 checkpoints'
    )
    parser.add_argument(
        '--best-checkpoint', type=Path, default=DEFAULT_BEST_CHECKPOINT
    )
    parser.add_argument(
        '--final-checkpoint', type=Path, default=DEFAULT_FINAL_CHECKPOINT
    )
    parser.add_argument(
        '--baseline-scenarios', type=Path, default=VALIDATED_BASELINE_PATH
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


def load_baselines(path: Path, seeds: list[int]) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {'policy_name', 'scenario_seed'}
    if not required.issubset(table.columns):
        raise ValueError(f'baseline file lacks columns: {required - set(table.columns)}')
    selected = table[
        table['policy_name'].isin(('v10_onlylow', 'Random'))
        & table['scenario_seed'].isin(seeds)
    ].copy()
    expected = pd.MultiIndex.from_product(
        [('v10_onlylow', 'Random'), seeds],
        names=('policy_name', 'scenario_seed'),
    )
    actual = pd.MultiIndex.from_frame(
        selected[['policy_name', 'scenario_seed']]
    )
    missing = expected.difference(actual)
    duplicated = selected.duplicated(
        ['policy_name', 'scenario_seed'], keep=False
    )
    if len(missing) or bool(duplicated.any()):
        raise ValueError(
            f'baseline rows do not form a unique paired set; '
            f'missing={list(missing[:5])}, duplicated={int(duplicated.sum())}'
        )
    return selected.sort_values(['scenario_seed', 'policy_name'])


def evaluate_v14(
    policy_name: str,
    checkpoint: Path,
    metadata: dict,
    agent: V14GlobalGraphMAPPOAgent,
    seed: int,
    max_steps: int,
):
    # The established evaluator is reused only for simulation/accounting.  Its
    # environment adapter is temporarily replaced by v14's observation-only
    # adapter.  V14 has no reservation or action postprocessing.
    original_environment = v12_test.V12MultiAgentEnv
    v12_test.V12MultiAgentEnv = V14MultiAgentEnv
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
        V14MultiAgentEnv.instances.pop(int(seed), None)
    row['policy_type'] = 'v14_global_graph_local_aggregation'
    row['low_mode'] = 'v14_global_graph_deterministic_greedy'
    # Evaluation forwards also generate encoder health rows.  They are not
    # training diagnostics and should not accumulate across all 50 scenarios.
    agent.low_actor.pop_diagnostics()
    return row


def main():
    args = parse_args()
    if args.max_steps <= 0 or args.torch_threads <= 0:
        raise ValueError('max-steps and torch-threads must be positive')
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
    best_agent, best_metadata = V14GlobalGraphMAPPOAgent.from_checkpoint(
        best_path, device=device
    )
    final_agent, final_metadata = V14GlobalGraphMAPPOAgent.from_checkpoint(
        final_path, device=device
    )
    best_agent.eval()
    final_agent.eval()
    best_agent.deterministic_low_actions = True
    final_agent.deterministic_low_actions = True
    experiments = (
        ('v14_best', best_path, best_metadata, best_agent),
        ('v14_episode300', final_path, final_metadata, final_agent),
    )

    rows = []
    print(
        f'device={device} v14_checkpoints=2 paired_scenarios={len(seeds)} '
        f'max_steps={args.max_steps}'
    )
    for seed in seeds:
        for name, path, metadata, agent in experiments:
            row = evaluate_v14(
                name, path, metadata, agent, seed, int(args.max_steps)
            )
            rows.append(row)
            print(
                f'{name} seed={seed} '
                f'success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_success={row["successful_ev_mcs_count"]} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f} '
                f'target_conflict={row["target_conflict_rate"]:.4f}'
            )

    v14_rows = pd.DataFrame(rows)
    baseline_rows = load_baselines(baseline_path, seeds)
    scenarios = pd.concat(
        [v14_rows, baseline_rows], ignore_index=True, sort=False
    ).sort_values(['scenario_seed', 'policy_name']).reset_index(drop=True)
    expected_rows = len(seeds) * 4
    if len(scenarios) != expected_rows:
        raise RuntimeError(
            f'unexpected scenario row count {len(scenarios)} != {expected_rows}'
        )

    comparisons = build_paired_comparison(scenarios, (
        ('v14_best', 'v14_episode300'),
        ('v14_best', 'v10_onlylow'),
        ('v14_best', 'Random'),
        ('v14_episode300', 'v10_onlylow'),
        ('v14_episode300', 'Random'),
        ('v10_onlylow', 'Random'),
    ))
    summary = build_summary(scenarios)
    if args.no_save:
        print(summary.to_string(index=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / 'v14_scenarios.csv', index=False)
    summary.to_csv(output_dir / 'v14_summary.csv', index=False)
    comparisons.to_csv(
        output_dir / 'v14_paired_comparison.csv', index=False
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
        'v14_best_checkpoint': str(best_path),
        'v14_best_checkpoint_metadata': best_metadata,
        'v14_episode300_checkpoint': str(final_path),
        'v14_episode300_checkpoint_metadata': final_metadata,
        'requested_baseline_path': str(REQUESTED_BASELINE_PATH),
        'requested_baseline_seed_range': '1001-1050',
        'actual_baseline_path': str(baseline_path),
        'actual_baseline_seed_range': '1050-1099',
        'baseline_reexecuted': False,
        'policy_semantics': {
            'v14_best': (
                'v14 episode-168 selected best checkpoint; deterministic '
                'greedy Low Actor; fixed strict remain<40 kWh High rule'
            ),
            'v14_episode300': (
                'v14 episode-300 checkpoint; deterministic greedy Low Actor; '
                'fixed strict remain<40 kWh High rule'
            ),
            'v10_onlylow': (
                'existing paired 1050-1099 deterministic checkpoint results'
            ),
            'Random': 'existing paired 1050-1099 complete Random results',
        },
        'fairness': (
            'All four policies use the identical 1050-1099 simulator seeds. '
            'V14 changes observations and policy networks only; simulator, '
            'matching and physical rules are unchanged.'
        ),
    }
    (output_dir / 'v14_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
