"""Add v12-compatible physical MCS dispersion metrics to v10_onlylow tests.

The wrapper records only the post-step distance statistic used by
``V12MultiAgentEnv``.  It does not change simulator observations, actions,
matching, rewards, or the v10_onlylow policy.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core import euclidean_distance
from environment import MultiAgentEnv
from test import load_rl_agent, resolve_device
import test_actor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v10_onlylow' / 'best_model.pt'
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v12'
DEFAULT_SEEDS = tuple(range(1050, 1100))


class SpatialAuditMultiAgentEnv(MultiAgentEnv):
    """Normal environment plus the exact v12 physical-pair audit metric."""

    instances = {}

    def __init__(self, seed=42):
        super().__init__(seed)
        self.seed = int(seed)
        self.spatial_conflict_numerator = 0
        self.spatial_conflict_denominator = 0
        self.__class__.instances[self.seed] = self

    def _record_spatial_conflicts(self):
        active = [
            mcs for mcs in self.world.MCSs
            if not (mcs.is_broken or mcs.is_energy_stranded)
        ]
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                self.spatial_conflict_numerator += int(
                    euclidean_distance(*left.pos, *right.pos) / 1000.0 <= 3.0
                )
                self.spatial_conflict_denominator += 1

    def step(self, action_n):
        result = super().step(action_n)
        # Same timing as V12MultiAgentEnv: after the normal world transition.
        self._record_spatial_conflicts()
        return result

    def spatial_metrics(self):
        return {
            'spatial_conflict_rate': (
                self.spatial_conflict_numerator
                / max(self.spatial_conflict_denominator, 1)
            ),
            'spatial_conflict_numerator': self.spatial_conflict_numerator,
            'spatial_conflict_denominator': self.spatial_conflict_denominator,
            'spatial_conflict_definition': (
                'post_step_active_mcs_pair_distance_km_lte_3'
            ),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description='v10_onlylow physical-MCS-dispersion audit for v12 seeds'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError('scenario seeds must be unique')
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint}')
    device = resolve_device(args.device)
    import torch
    torch.set_num_threads(args.torch_threads)
    agent, metadata = load_rl_agent(checkpoint, args.hidden_dim, device)

    # Reuse the established v10_onlylow option lifecycle and policy.  Only its
    # environment symbol is replaced for the duration of this standalone run.
    original_environment = test_actor.MultiAgentEnv
    test_actor.MultiAgentEnv = SpatialAuditMultiAgentEnv
    rows = []
    try:
        for seed in args.seeds:
            row = test_actor.evaluate_scenario(
                experiment_name='v10_onlylow',
                policy_mode='threshold_high_learned_low',
                scenario_seed=int(seed),
                random_low_seed=int(seed),
                max_steps=int(args.max_steps),
                checkpoint_path=checkpoint,
                checkpoint_metadata=metadata,
                agent=agent,
            )
            environment = SpatialAuditMultiAgentEnv.instances.pop(int(seed))
            row.update(environment.spatial_metrics())
            rows.append(row)
            print(
                f'seed={seed} spatial_conflict='
                f'{row["spatial_conflict_rate"]:.4f} '
                f'success={row["ev_charge_success_ratio"]:.4f}'
            )
    finally:
        test_actor.MultiAgentEnv = original_environment

    table = pd.DataFrame(rows)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = output_dir / 'v10_onlylow_spatial_scenarios.csv'
    summary_path = output_dir / 'v10_onlylow_spatial_summary.csv'
    config_path = output_dir / 'v10_onlylow_spatial_config.json'
    table.to_csv(scenario_path, index=False)
    summary = pd.DataFrame([{
        'policy_name': 'v10_onlylow',
        'scenario_count': len(table),
        'seed_min': int(table['scenario_seed'].min()),
        'seed_max': int(table['scenario_seed'].max()),
        'spatial_conflict_rate': float(table['spatial_conflict_rate'].mean()),
        'spatial_conflict_numerator': int(table['spatial_conflict_numerator'].sum()),
        'spatial_conflict_denominator': int(table['spatial_conflict_denominator'].sum()),
        'weighted_spatial_conflict_rate': float(
            table['spatial_conflict_numerator'].sum()
            / max(table['spatial_conflict_denominator'].sum(), 1)
        ),
        'spatial_conflict_definition': (
            'post_step_active_mcs_pair_distance_km_lte_3'
        ),
    }])
    summary.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps({
        'evaluated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'checkpoint': str(checkpoint),
        'checkpoint_metadata': metadata,
        'seeds': sorted(int(seed) for seed in args.seeds),
        'max_steps': int(args.max_steps),
        'device': device,
        'measurement_equivalence': (
            'Same post-step active-MCS pair distance <= 3 km definition and '
            'timing as demo.train_v12.V12MultiAgentEnv._record_spatial_conflicts'
        ),
        'simulator_changes': 'none; audit wrapper only',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
