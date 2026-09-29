"""Paired evaluation of four policies on identical scenario seeds.

All four policies are re-executed from scratch (no cached baseline rows):

  1. Random                                    -- complete random decision policy
  2. v10_onlylow                               -- v10-onlylow learned Low Actor
  3. v14_outcome_reward_episode1000            -- v14 global graph, outcome reward
  4. v14_nodeonly_outcome_reward_episode1000   -- v14 node-only, outcome reward

The simulator, matching, reward, candidate action set, fixed strict 40 kWh High
rule, deterministic greedy Low decisions, max steps and paired scenario seeds
are identical across all four policies.  Random and v10_onlylow are run through
the same established ``test_v12.evaluate_scenario`` harness that produced the
original validated baselines, so their protocol is unchanged.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from test import build_summary, load_rl_agent, resolve_device
from test_actor import build_paired_comparison
import test_v12 as v12_test
from test_v14 import evaluate_v14
from v14_global_graph import V14GlobalGraphMAPPOAgent
from v14_nodeonly import V14NodeOnlyMAPPOAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v14_nodeonly_outcome_reward'
DEFAULT_SEEDS = tuple(range(1050, 1100))
DEFAULT_GRAPH = (
    PROJECT_DIR / 'training_results_v14_outcome_reward'
    / 'model_episode_1000.pt'
)
DEFAULT_NODEONLY = (
    PROJECT_DIR / 'training_results_v14_nodeonly_outcome_reward'
    / 'model_episode_1000.pt'
)
DEFAULT_ONLYLOW = (
    PROJECT_DIR / 'training_results_v10_onlylow' / 'best_model.pt'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Re-evaluate Random, v10_onlylow and the two outcome-reward v14 '
            'episode-1000 models on paired scenarios'
        )
    )
    parser.add_argument('--graph-checkpoint', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument(
        '--nodeonly-checkpoint', type=Path, default=DEFAULT_NODEONLY
    )
    parser.add_argument(
        '--onlylow-checkpoint', type=Path, default=DEFAULT_ONLYLOW
    )
    parser.add_argument('--hidden-dim', type=int, default=128)
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


def evaluate_onlylow(
    policy_name, checkpoint, metadata, agent, seed, max_steps
):
    """Run the v10-onlylow learned Low Actor through the shared harness."""
    row = v12_test.evaluate_scenario(
        policy_name=policy_name,
        policy_type='v10_onlylow',
        scenario_seed=seed,
        max_steps=max_steps,
        checkpoint_path=checkpoint,
        checkpoint_metadata=metadata,
        agent=agent,
    )
    row['policy_type'] = 'v10_onlylow_learned'
    row['low_mode'] = 'learned_deterministic_greedy'
    return row


def evaluate_random(policy_name, seed, max_steps):
    """Run the complete random decision policy through the shared harness."""
    row = v12_test.evaluate_scenario(
        policy_name=policy_name,
        policy_type='full_random',
        scenario_seed=seed,
        max_steps=max_steps,
        checkpoint_path=None,
        checkpoint_metadata={},
        agent=None,
    )
    row['policy_type'] = 'random'
    row['low_mode'] = 'not_applicable'
    return row


def _print_row(name, seed, row):
    print(
        f'{name} seed={seed} '
        f'success={row["ev_charge_success_ratio"]:.4f} '
        f'mcs_success={row["successful_ev_mcs_count"]} '
        f'mcs_profit={row["avg_mcs_profit"]:.2f} '
        f'target_conflict={row.get("target_conflict_rate", float("nan")):.4f}'
    )


def main():
    args = parse_args()
    if args.max_steps <= 0 or args.torch_threads <= 0 or args.hidden_dim <= 0:
        raise ValueError(
            'max-steps, torch-threads and hidden-dim must be positive'
        )
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError('scenario seeds must be unique')
    paths = {
        'v14_outcome_reward_episode1000': (
            args.graph_checkpoint.expanduser().resolve()
        ),
        'v14_nodeonly_outcome_reward_episode1000': (
            args.nodeonly_checkpoint.expanduser().resolve()
        ),
    }
    onlylow_path = args.onlylow_checkpoint.expanduser().resolve()
    for path in (*paths.values(), onlylow_path):
        if not path.is_file():
            raise FileNotFoundError(f'checkpoint not found: {path}')

    device = resolve_device(args.device)
    torch.set_num_threads(int(args.torch_threads))

    agents = {}
    metadata = {}
    agents['v14_outcome_reward_episode1000'], metadata[
        'v14_outcome_reward_episode1000'
    ] = V14GlobalGraphMAPPOAgent.from_checkpoint(
        paths['v14_outcome_reward_episode1000'], device=device
    )
    agents['v14_nodeonly_outcome_reward_episode1000'], metadata[
        'v14_nodeonly_outcome_reward_episode1000'
    ] = V14NodeOnlyMAPPOAgent.from_checkpoint(
        paths['v14_nodeonly_outcome_reward_episode1000'], device=device
    )
    for agent in agents.values():
        agent.eval()
        agent.deterministic_low_actions = True
    onlylow_agent, onlylow_metadata = load_rl_agent(
        onlylow_path, int(args.hidden_dim), device
    )

    rows = []
    print(
        f'device={device} reexecuted_policies=4 paired_scenarios={len(seeds)} '
        f'max_steps={args.max_steps}'
    )
    for seed in seeds:
        for name, path in paths.items():
            row = evaluate_v14(
                name, path, metadata[name], agents[name], seed,
                int(args.max_steps),
            )
            is_nodeonly = name.startswith('v14_nodeonly')
            row['policy_type'] = (
                'v14_nodeonly_outcome_reward_no_message_passing'
                if is_nodeonly else 'v14_outcome_reward_global_graph'
            )
            row['low_mode'] = 'deterministic_greedy_fixed_40kwh_high_rule'
            rows.append(row)
            _print_row(name, seed, row)
        row = evaluate_onlylow(
            'v10_onlylow', onlylow_path, onlylow_metadata, onlylow_agent,
            seed, int(args.max_steps),
        )
        rows.append(row)
        _print_row('v10_onlylow', seed, row)
        row = evaluate_random('Random', seed, int(args.max_steps))
        rows.append(row)
        _print_row('Random', seed, row)

    scenarios = pd.DataFrame(rows).sort_values(
        ['scenario_seed', 'policy_name']
    ).reset_index(drop=True)
    expected = 4 * len(seeds)
    if len(scenarios) != expected:
        raise RuntimeError(
            f'unexpected row count {len(scenarios)} != {expected}'
        )
    summary = build_summary(scenarios)
    comparisons = build_paired_comparison(scenarios, [
        (
            'v14_outcome_reward_episode1000',
            'v14_nodeonly_outcome_reward_episode1000',
        ),
        ('v14_outcome_reward_episode1000', 'v10_onlylow'),
        ('v14_outcome_reward_episode1000', 'Random'),
        ('v14_nodeonly_outcome_reward_episode1000', 'v10_onlylow'),
        ('v14_nodeonly_outcome_reward_episode1000', 'Random'),
        ('v10_onlylow', 'Random'),
    ])
    if args.no_save:
        print(summary.to_string(index=False))
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(
        output_dir / 'v14_nodeonly_outcome_reward_scenarios.csv', index=False
    )
    summary.to_csv(
        output_dir / 'v14_nodeonly_outcome_reward_summary.csv', index=False
    )
    comparisons.to_csv(
        output_dir / 'v14_nodeonly_outcome_reward_paired_comparison.csv',
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
        'baseline_reexecuted': True,
        'all_policies_reexecuted': True,
        'checkpoints': {
            name: {'path': str(path), 'metadata': metadata[name]}
            for name, path in paths.items()
        },
        'v10_onlylow_checkpoint': {
            'path': str(onlylow_path),
            'metadata': onlylow_metadata,
        },
        'comparison_variable': 'inter_node_relation_messages',
        'graph_model': (
            'canonical global heterogeneous graph with two local relation '
            'aggregation layers'
        ),
        'nodeonly_model': (
            'identical nodes, candidates, heads and parameter shapes; all '
            'relation matrices are zeroed before every encoder forward'
        ),
        'common_conditions': (
            'Same simulator, matching, reward and candidate action set.  '
            'All four policies are re-executed on the same paired seeds '
            '1050-1099 with deterministic greedy Low decisions and the '
            'strict fixed 40 kWh High rule.'
        ),
    }
    (output_dir / 'v14_nodeonly_outcome_reward_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved: {output_dir}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
