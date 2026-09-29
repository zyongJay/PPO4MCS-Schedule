"""Evaluate the v11 HG-MAPPO no-reservation ablation on fixed scenarios."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from test import build_summary, resolve_device
import test_v12 as v12_test
from train_v11 import V11MultiAgentEnv
from v12_hypergraph import V11CompetitionHypergraphAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_DIR / 'training_results_v11' / 'best_model.pt'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v11'
DEFAULT_SEEDS = tuple(range(1050, 1100))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate v11 on fixed scenarios')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
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
    torch.set_num_threads(args.torch_threads)
    agent, metadata = V11CompetitionHypergraphAgent.from_checkpoint(
        checkpoint, device=device
    )
    agent.eval()

    # Reuse the established v12 evaluator but substitute v11's observation
    # adapter.  The agent itself disables both occupancy and reservation.
    original_environment = v12_test.V12MultiAgentEnv
    v12_test.V12MultiAgentEnv = V11MultiAgentEnv
    rows = []
    try:
        for seed in args.seeds:
            row = v12_test.evaluate_scenario(
                policy_name='v11',
                policy_type='v12',
                scenario_seed=int(seed),
                max_steps=int(args.max_steps),
                checkpoint_path=checkpoint,
                checkpoint_metadata=metadata,
                agent=agent,
            )
            row['policy_type'] = 'v11_hypergraph_no_reservation'
            row['low_mode'] = 'competition_hypergraph_no_reservation'
            rows.append(row)
            print(
                f'v11 seed={seed} success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f} '
                f'target_conflict={row["target_conflict_rate"]:.4f}'
            )
    finally:
        v12_test.V12MultiAgentEnv = original_environment

    scenarios = pd.DataFrame(rows)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / 'v11_scenarios.csv', index=False)
    build_summary(scenarios).to_csv(output_dir / 'v11_summary.csv', index=False)
    (output_dir / 'v11_config.json').write_text(json.dumps({
        'evaluated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'checkpoint': str(checkpoint),
        'checkpoint_metadata': metadata,
        'seeds': sorted(int(seed) for seed in args.seeds),
        'paired_scenario_count': len(args.seeds),
        'max_steps': int(args.max_steps),
        'device': device,
        'policy_semantics': (
            'v11 B3 Competition-HG Low MAPPO; fixed strict remain<40 kWh '
            'High threshold; no local soft occupancy and no sequential reservation'
        ),
        'simulator_changes': 'none',
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
