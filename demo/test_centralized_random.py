"""Evaluate a reproducible Random policy in the centralized dispatch world.

For every eligible MCS the policy samples uniformly from currently legal
dispatch points.  Stay is used only if no dispatch point is energy-feasible.
Task MCSs retain the environment's implicit Continue action ``-1``.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import world as world_module
from centralized_env import CentralizedDispatchEnv
from config import MAX_STEPS_PER_EPISODE, TRACK_DATA_PATH


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / 'test_results_centralized_random'
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Random dispatch-point baseline in centralized environment'
    )
    parser.add_argument('--seed-start', type=int, default=1050)
    parser.add_argument('--num-scenarios', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument('--grid-rows', type=int, default=4)
    parser.add_argument('--grid-columns', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def random_joint_action(
    observation: Dict[str, np.ndarray],
    dispatch_count: int,
    stay_action: int,
    rng: np.random.Generator,
) -> np.ndarray:
    eligible = np.asarray(observation['eligible_mask'], dtype=bool)
    legal = np.asarray(observation['action_mask'], dtype=bool)
    actions = np.full(eligible.shape, -1, dtype=np.int64)
    for mcs_index in np.flatnonzero(eligible):
        dispatch_choices = np.flatnonzero(
            legal[mcs_index, :dispatch_count]
        )
        if dispatch_choices.size:
            actions[mcs_index] = int(rng.choice(dispatch_choices))
        elif legal[mcs_index, stay_action]:
            actions[mcs_index] = int(stay_action)
        else:
            raise RuntimeError(
                f'eligible MCS index {mcs_index} has no legal action'
            )
    return actions


def run_scenario(args, scenario_seed: int) -> Dict:
    env = CentralizedDispatchEnv(
        seed=scenario_seed,
        grid_rows=args.grid_rows,
        grid_columns=args.grid_columns,
        max_steps=args.max_steps,
        verbose=False,
    )
    observation = env.reset()
    rng = np.random.default_rng(scenario_seed)
    episode_reward = 0.0
    decision_count = 0
    decision_step_count = 0
    stay_count = 0
    dispatch_counts = np.zeros(env.dispatch_point_count, dtype=np.int64)
    dispatch_distance_km = 0.0
    completed_event_count = 0
    failure_event_count = 0
    realised_service_profit = 0.0
    forced_recharge_requests = 0
    started = time.perf_counter()
    last_info: Dict = {}

    for _step in range(args.max_steps):
        actions = random_joint_action(
            observation,
            env.dispatch_point_count,
            env.stay_action,
            rng,
        )
        selected = actions[actions >= 0]
        if selected.size:
            decision_step_count += 1
            decision_count += int(selected.size)
            stay_count += int(np.count_nonzero(selected == env.stay_action))
            for point in selected[selected < env.dispatch_point_count]:
                dispatch_counts[int(point)] += 1
        observation, reward, done, info = env.step(actions)
        episode_reward += float(reward)
        dispatch_distance_km += float(
            info.get('dispatch_distance_km_step', 0.0)
        )
        completed_event_count += int(
            info.get('completed_service_count_step', 0)
        )
        failure_event_count += int(info.get('new_failure_count_step', 0))
        realised_service_profit += float(
            info.get('realised_service_profit_step', 0.0)
        )
        forced_recharge_requests += int(
            info.get('forced_recharge_request_count_step', 0)
        )
        last_info = info
        if done:
            break

    completed_evs = [
        ev for ev in env.world.EVs
        if ev.is_charged and float(ev.charge_time_remain_min) <= 1e-8
    ]
    failures = [ev for ev in env.world.EVs if ev.fail_charge]
    unresolved = max(
        len(env.world.EVs) - len(completed_evs) - len(failures), 0
    )
    completed_mcs = sum(
        ev.charge_provider_type == 'MCS' for ev in completed_evs
    )
    completed_fcs = sum(
        ev.charge_provider_type == 'FCS' for ev in completed_evs
    )
    mcs_count = max(len(env.world.MCSs), 1)
    fcs_count = max(len(env.world.FCSs), 1)
    resolved = max(len(completed_evs) + len(failures), 1)
    elapsed = time.perf_counter() - started
    row = {
        'policy_name': 'centralized_random_dispatch_point',
        'scenario_seed': int(scenario_seed),
        'random_action_seed': int(scenario_seed),
        'steps': int(last_info.get('step', args.max_steps)),
        'episode_reward': float(episode_reward),
        'completed_ev_count': int(len(completed_evs)),
        'failed_ev_count': int(len(failures)),
        'unresolved_ev_count': int(unresolved),
        'completed_ev_population_ratio': float(
            len(completed_evs) / max(len(env.world.EVs), 1)
        ),
        'completed_ev_resolved_ratio': float(len(completed_evs) / resolved),
        'mcs_completed_ev_count': int(completed_mcs),
        'fcs_completed_ev_count': int(completed_fcs),
        'mcs_completed_ev_share': float(
            completed_mcs / max(len(completed_evs), 1)
        ),
        'decision_step_count': int(decision_step_count),
        'joint_mcs_decision_count': int(decision_count),
        'stay_selection_count': int(stay_count),
        'dispatch_selection_count': int(dispatch_counts.sum()),
        'unique_dispatch_points_selected': int(np.count_nonzero(dispatch_counts)),
        'max_dispatch_point_selection_count': int(dispatch_counts.max(initial=0)),
        'dispatch_distance_km': float(dispatch_distance_km),
        'completed_service_event_count': int(completed_event_count),
        'failure_event_count': int(failure_event_count),
        'realised_service_profit': float(realised_service_profit),
        'forced_recharge_request_count': int(forced_recharge_requests),
        'total_mcs_profit_accounting': float(sum(
            mcs.total_profit for mcs in env.world.MCSs
        )),
        'avg_mcs_profit_accounting': float(sum(
            mcs.total_profit for mcs in env.world.MCSs
        ) / mcs_count),
        'total_fcs_profit_accounting': float(sum(
            fcs.total_profit for fcs in env.world.FCSs
        )),
        'avg_fcs_profit_accounting': float(sum(
            fcs.total_profit for fcs in env.world.FCSs
        ) / fcs_count),
        'avg_mcs_idle_time_min': float(sum(
            mcs.total_idle_time_min for mcs in env.world.MCSs
        ) / mcs_count),
        'avg_fcs_idle_time_min': float(sum(
            fcs.total_idle_time_min for fcs in env.world.FCSs
        ) / fcs_count),
        'avg_completed_ev_delay_min': float(np.mean([
            ev.total_wait_time_min for ev in completed_evs
        ])) if completed_evs else 0.0,
        'broken_mcs_count': int(sum(
            mcs.is_broken for mcs in env.world.MCSs
        )),
        'energy_stranded_mcs_count': int(sum(
            mcs.is_energy_stranded for mcs in env.world.MCSs
        )),
        'wall_time_seconds': float(elapsed),
        **env.graph_builder.metrics(),
    }
    return row


def write_csv(path: Path, rows: List[Dict]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: List[Dict]) -> Dict:
    excluded = {
        'policy_name', 'scenario_seed', 'random_action_seed'
    }
    result: Dict[str, float | int | str] = {
        'policy_name': 'centralized_random_dispatch_point',
        'scenario_count': int(len(rows)),
        'seed_start': int(min(row['scenario_seed'] for row in rows)),
        'seed_end': int(max(row['scenario_seed'] for row in rows)),
    }
    for key in rows[0]:
        if key in excluded:
            continue
        values = np.asarray([float(row[key]) for row in rows], np.float64)
        result[f'{key}_mean'] = float(values.mean())
        result[f'{key}_std'] = float(values.std(ddof=0))
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.num_scenarios <= 0 or args.max_steps <= 0:
        raise ValueError('num-scenarios and max-steps must be positive')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(args.num_scenarios):
        seed = int(args.seed_start + offset)
        row = run_scenario(args, seed)
        rows.append(row)
        print(
            f'seed={seed} completed={row["completed_ev_count"]} '
            f'failed={row["failed_ev_count"]} '
            f'population_success={row["completed_ev_population_ratio"]:.4f} '
            f'dispatch_km={row["dispatch_distance_km"]:.2f}'
        )
    summary = summarise(rows)
    write_csv(output_dir / 'centralized_random_scenarios.csv', rows)
    write_csv(output_dir / 'centralized_random_summary.csv', [summary])
    (output_dir / 'centralized_random_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    config = {
        'policy': 'uniform_random_legal_dispatch_point',
        'stay_semantics': 'fallback_only_when_no_dispatch_point_is_legal',
        'seed_start': args.seed_start,
        'num_scenarios': args.num_scenarios,
        'max_steps': args.max_steps,
        'grid_rows': args.grid_rows,
        'grid_columns': args.grid_columns,
        'dispatch_point_count': args.grid_rows * args.grid_columns,
        'environment': 'centralized_fixed_step_schedule_progress_match_reward',
    }
    (output_dir / 'test_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

