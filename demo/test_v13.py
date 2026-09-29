"""Evaluate v13 system-profit v10_onlylow on fixed scenarios."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from test import build_summary, load_rl_agent, resolve_device
import test_actor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_DIR / 'training_results_v13' / 'best_model.pt'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results_v13'
DEFAULT_SEEDS = tuple(range(1050, 1100))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate v13 on fixed scenarios')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=128)
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
    agent, metadata = load_rl_agent(checkpoint, args.hidden_dim, device)
    rows = []
    for seed in args.seeds:
        row = test_actor.evaluate_scenario(
            experiment_name='v13',
            policy_mode='threshold_high_learned_low',
            scenario_seed=int(seed),
            random_low_seed=int(seed),
            max_steps=int(args.max_steps),
            checkpoint_path=checkpoint,
            checkpoint_metadata=metadata,
            agent=agent,
        )
        row['policy_type'] = 'v13_system_total_profit_onlylow'
        row['low_mode'] = 'learned_system_total_profit'
        rows.append(row)
        print(
            f'v13 seed={seed} success={row["ev_charge_success_ratio"]:.4f} '
            f'mcs_profit={row["avg_mcs_profit"]:.2f}'
        )

    scenarios = pd.DataFrame(rows)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(output_dir / 'v13_scenarios.csv', index=False)
    build_summary(scenarios).to_csv(output_dir / 'v13_summary.csv', index=False)
    (output_dir / 'v13_config.json').write_text(json.dumps({
        'evaluated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'checkpoint': str(checkpoint),
        'checkpoint_metadata': metadata,
        'seeds': sorted(int(seed) for seed in args.seeds),
        'paired_scenario_count': len(args.seeds),
        'max_steps': int(args.max_steps),
        'device': device,
        'policy_semantics': (
            'v13 Low MAPPO checkpoint selected by success then system-total-profit; '
            'fixed strict remain<40 kWh High threshold'
        ),
        'simulator_changes': 'none',
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
