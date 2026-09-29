"""Render macro/micro views for centralized Random dispatch.

This keeps the visualization style of ``visual.py`` while adapting its
MCS-state snapshot to the centralized dispatch task lifecycle.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import world as world_module
from centralized_env import CentralizedDispatchEnv
from config import COMM_RANGE, MAX_STEPS_PER_EPISODE, TRACK_DATA_PATH
from core import euclidean_distance
from test_centralized_random import random_joint_action
from visual import render_macro_png, render_micro_gif, write_count_csv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'visual_results_centralized_random'
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Render centralized Random dispatch for one fixed seed'
    )
    parser.add_argument('--seed', type=int, default=1003)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument('--grid-rows', type=int, default=4)
    parser.add_argument('--grid-columns', type=int, default=4)
    parser.add_argument('--frame-duration-ms', type=int, default=180)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _mcs_state(mcs) -> str:
    """Map persistent centralized dispatch to the existing task colour."""
    if mcs.is_broken:
        return 'broken'
    if mcs.is_recharging:
        return 'recharge'
    if bool(getattr(mcs, 'centralized_is_dispatching', False)) or mcs.is_task:
        return 'task'
    return 'idle'


def _nearby_mcs_counts(env) -> Dict[int, int]:
    radius_m = float(COMM_RANGE) * 1000.0
    return {
        int(fcs.id): sum(
            euclidean_distance(*fcs.pos, *mcs.pos) <= radius_m
            for mcs in env.world.MCSs
        )
        for fcs in env.world.FCSs
    }


def capture_snapshot(env, step: int) -> Dict:
    return {
        'step': int(step),
        'mcs': [
            {
                'id': int(mcs.id),
                'x': float(mcs.pos[0]),
                'y': float(mcs.pos[1]),
                'state': _mcs_state(mcs),
            }
            for mcs in env.world.MCSs
        ],
        'failed_evs': [
            {'x': float(ev.pos[0]), 'y': float(ev.pos[1])}
            for ev in env.world.EVs if ev.fail_charge
        ],
        'all_evs': [
            {'x': float(ev.pos[0]), 'y': float(ev.pos[1])}
            for ev in env.world.EVs
        ],
        'nearby_counts': _nearby_mcs_counts(env),
    }


def write_rows(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> Dict:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = CentralizedDispatchEnv(
        seed=args.seed,
        grid_rows=args.grid_rows,
        grid_columns=args.grid_columns,
        max_steps=args.max_steps,
        verbose=False,
    )
    observation = env.reset()
    rng = np.random.default_rng(args.seed)
    snapshots = [capture_snapshot(env, 0)]
    dispatch_counts = np.zeros(env.dispatch_point_count, dtype=np.int64)

    for step in range(1, args.max_steps + 1):
        actions = random_joint_action(
            observation, env.dispatch_point_count, env.stay_action, rng
        )
        selected = actions[(actions >= 0) & (actions < env.dispatch_point_count)]
        for point in selected:
            dispatch_counts[int(point)] += 1
        observation, _reward, done, _info = env.step(actions)
        snapshots.append(capture_snapshot(env, step))
        if done:
            break

    fcs_ids = [int(fcs.id) for fcs in env.world.FCSs]
    fcs_positions = [list(fcs.pos) for fcs in env.world.FCSs]
    stem = f'centralized_random_seed_{args.seed}'
    label = (
        f'Centralized Random | {args.grid_rows}x{args.grid_columns} '
        'dispatch grid'
    )
    write_count_csv(
        output_dir / f'{stem}_fcs_nearby_mcs.csv',
        args.seed,
        fcs_ids,
        snapshots,
    )
    render_micro_gif(
        output_dir / f'{stem}_micro.gif',
        args.seed,
        fcs_positions,
        fcs_ids,
        snapshots,
        args.frame_duration_ms,
        checkpoint_label=label,
    )
    render_macro_png(
        output_dir / f'{stem}_macro.png',
        args.seed,
        fcs_positions,
        snapshots,
        checkpoint_label=label,
    )

    completed = [
        ev for ev in env.world.EVs
        if ev.is_charged and float(ev.charge_time_remain_min) <= 1e-8
    ]
    failures = [ev for ev in env.world.EVs if ev.fail_charge]
    summary = {
        'policy': 'centralized_random_dispatch_point',
        'scenario_seed': int(args.seed),
        'steps': int(len(snapshots) - 1),
        'grid_rows': int(args.grid_rows),
        'grid_columns': int(args.grid_columns),
        'dispatch_point_count': int(env.dispatch_point_count),
        'completed_ev_count': int(len(completed)),
        'failed_ev_count': int(len(failures)),
        'unresolved_ev_count': int(len(env.world.EVs) - len(completed) - len(failures)),
        'completed_ev_population_ratio': float(len(completed) / len(env.world.EVs)),
        'mcs_completed_ev_count': int(sum(
            ev.charge_provider_type == 'MCS' for ev in completed
        )),
        'fcs_completed_ev_count': int(sum(
            ev.charge_provider_type == 'FCS' for ev in completed
        )),
        'dispatch_selection_count': int(dispatch_counts.sum()),
        'unique_dispatch_points_selected': int(np.count_nonzero(dispatch_counts)),
    }
    write_rows(output_dir / f'{stem}_summary.csv', [summary])
    (output_dir / f'{stem}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return summary


if __name__ == '__main__':
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
