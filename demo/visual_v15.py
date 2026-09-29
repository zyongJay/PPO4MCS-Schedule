"""Render v15-final scheduling GIFs and cumulative coverage maps.

This module reuses the rendering and accounting conventions in ``visual.py``.
Only the environment and policy adapter are replaced with the v15 signed-graph
environment and its fixed 40 kWh High-level rule.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

import visual as base_visual
from test import remember_charge_providers, resolve_device
from test_v12 import V12CheckpointPolicy
from train_v15 import V15MultiAgentEnv
from v15_signed_graph import V15SignedGraphMAPPOAgent


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_DIR / 'training_results_v15' / 'model_episode_300.pt'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'visual_results_v15_final'
DEFAULT_SEEDS = (1001, 1002, 1003)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate v15-final micro scheduling GIFs and coverage maps'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--frame-duration-ms', type=int, default=250)
    return parser.parse_args()


def run_scenario(
    seed: int,
    max_steps: int,
    policy: V12CheckpointPolicy,
    output_dir: Path,
    frame_duration_ms: int,
    checkpoint_label: str,
) -> Dict:
    """Execute one v15-final fixed scenario and render the two visual outputs."""
    base_visual.set_scenario_seed(seed)
    env = V15MultiAgentEnv(seed)
    env.world.verbose = False
    observations = env.reset()
    providers: Dict[int, str] = {}
    remember_charge_providers(env.world.EVs, providers)
    snapshots: List[Dict] = [base_visual.capture_snapshot(env, step=0)]

    executed_steps = 0
    for step in range(1, max_steps + 1):
        acting_agents = list(env.world.agents)
        policy.synchronize()
        actions = policy.build_actions(env, acting_agents, observations)
        if len(actions) != len(acting_agents):
            raise RuntimeError('action count does not match acting agents')
        observations, _, _, _ = env.step(actions)
        policy.synchronize()
        executed_steps = step
        remember_charge_providers(env.world.EVs, providers)
        snapshots.append(base_visual.capture_snapshot(env, step=step))
        if env.world.get_done():
            break

    fcs_ids = [int(fcs.id) for fcs in env.world.FCSs]
    fcs_positions = [list(fcs.pos) for fcs in env.world.FCSs]
    stem = f'scenario_seed_{seed}'
    base_visual.write_count_csv(
        output_dir / f'{stem}_fcs_nearby_mcs.csv', seed, fcs_ids, snapshots
    )
    base_visual.render_micro_gif(
        output_dir / f'{stem}_micro.gif',
        seed,
        fcs_positions,
        fcs_ids,
        snapshots,
        frame_duration_ms,
        checkpoint_label,
    )
    base_visual.render_macro_png(
        output_dir / f'{stem}_macro.png',
        seed,
        fcs_positions,
        snapshots,
        checkpoint_label,
    )

    successful = [ev for ev in env.world.EVs if ev.is_charged]
    failed_count = sum(bool(ev.fail_charge) for ev in env.world.EVs)
    completed = len(successful) + failed_count
    summary = {
        'scenario_seed': int(seed),
        'steps': int(executed_steps),
        'ev_success_count': len(successful),
        'ev_failure_count': int(failed_count),
        'ev_charge_success_ratio': len(successful) / completed if completed else 0.0,
        'successful_ev_mcs_count': sum(
            providers.get(int(ev.id)) == 'MCS' for ev in successful
        ),
        'successful_ev_fcs_count': sum(
            providers.get(int(ev.id)) == 'FCS' for ev in successful
        ),
    }
    for fcs_id in fcs_ids:
        values = [row['nearby_counts'][fcs_id] for row in snapshots]
        summary[f'FCS_{fcs_id}_nearby_mcs_mean'] = float(sum(values) / len(values))
        summary[f'FCS_{fcs_id}_nearby_mcs_max'] = int(max(values))
    V15MultiAgentEnv.instances.pop(int(seed), None)
    return summary


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.torch_threads <= 0:
        raise ValueError('max-steps and torch-threads must be positive')
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError('scenario seeds must be unique')
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint does not exist: {checkpoint}')

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.set_num_threads(int(args.torch_threads))
    agent, metadata = V15SignedGraphMAPPOAgent.from_checkpoint(
        checkpoint, device=device
    )
    agent.eval()
    agent.deterministic_low_actions = True
    policy = V12CheckpointPolicy(agent, 0)
    episode = int(metadata.get('episode', 300))
    label = f'V15 final checkpoint episode {episode}'

    summaries = []
    for seed in args.seeds:
        print(f'Rendering v15_final seed={seed} ...')
        summary = run_scenario(
            int(seed),
            int(args.max_steps),
            policy,
            output_dir,
            int(args.frame_duration_ms),
            label,
        )
        summaries.append(summary)
        print(
            f"seed={seed} completed: success={summary['ev_charge_success_ratio']:.4f}, "
            f"MCS/FCS={summary['successful_ev_mcs_count']}/"
            f"{summary['successful_ev_fcs_count']}"
        )
    summary_path = output_dir / 'scenario_summary.csv'
    existing = {}
    if summary_path.is_file():
        with summary_path.open(newline='', encoding='utf-8-sig') as handle:
            existing = {
                int(row['scenario_seed']): row
                for row in csv.DictReader(handle)
            }
    existing.update({int(row['scenario_seed']): row for row in summaries})
    base_visual.write_rows(
        summary_path,
        [existing[seed] for seed in sorted(existing)],
    )
    print(f'Saved GIFs, coverage maps and summaries to: {output_dir}')


if __name__ == '__main__':
    main()
