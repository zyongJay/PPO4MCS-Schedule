"""Render learning curves from a centralized PPO training log.

The script is intentionally independent of training: it can be safely run
while training is active and rerun after the final checkpoint is written.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = SCRIPT_DIR.parent / 'training_results_centralized_v1'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Plot centralized PPO learning curves from training_log.csv'
    )
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--output', type=Path)
    return parser.parse_args(argv)


def _read_rows(path: Path) -> list[dict[str, float]]:
    # The trainer rewrites the CSV after each PPO update.  A plot requested
    # concurrently can briefly observe only its header, so retry a few times
    # rather than treating that transient state as an empty experiment.
    for attempt in range(20):
        with path.open(newline='', encoding='utf-8') as handle:
            raw_rows = list(csv.DictReader(handle))
        rows = []
        for raw in raw_rows:
            try:
                row = {}
                for key, value in raw.items():
                    if value in (None, ''):
                        continue
                    try:
                        row[key] = float(value)
                    except ValueError:
                        # Metadata columns such as encoder_mode are not curve
                        # values and should not discard an otherwise valid row.
                        continue
                row['episode'] = int(raw['episode'])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
        if rows or attempt == 19:
            return sorted(rows, key=lambda item: item['episode'])
        time.sleep(0.25)
    return []


def _series(rows: list[dict[str, float]], name: str) -> np.ndarray:
    return np.asarray([row.get(name, np.nan) for row in rows], dtype=float)


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        segment = values[start:index + 1]
        if np.isfinite(segment).any():
            result[index] = np.nanmean(segment)
    return result


def _plot_with_rolling(axis, episode, values, label, window, color):
    axis.plot(episode, values, color=color, alpha=0.26, linewidth=0.9)
    axis.plot(
        episode, _rolling(values, window), color=color, linewidth=1.8,
        label=f'{label} ({window}-episode mean)',
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)


def render(input_dir: Path, output: Path, window: int) -> None:
    log_path = input_dir / 'training_log.csv'
    rows = _read_rows(log_path)
    if not rows:
        raise RuntimeError(f'no valid rows found in {log_path}')

    episode = _series(rows, 'episode')
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    figure.suptitle(
        f'Centralized PPO training curves ({len(rows)} episodes logged)',
        fontsize=14,
    )

    _plot_with_rolling(
        axes[0, 0], episode, _series(rows, 'episode_reward'),
        'episode reward', window, '#1f77b4',
    )
    axes[0, 0].set_title('Outcome reward')
    axes[0, 0].set_xlabel('episode')

    success_axis = axes[0, 1]
    _plot_with_rolling(
        success_axis, episode, _series(rows, 'completed_ev_population_ratio'),
        'population completion ratio', window, '#2a9d8f',
    )
    success_axis.plot(
        episode, _rolling(_series(rows, 'completed_ev_ratio_resolved'), window),
        color='#e9c46a', linewidth=1.5, label='resolved completion ratio',
    )
    success_axis.set_ylim(-0.02, 1.02)
    success_axis.set_title('EV completion ratios')
    success_axis.set_xlabel('episode')
    success_axis.legend(fontsize=8)

    counts_axis = axes[1, 0]
    for name, label, color in (
        ('completed_ev_count', 'completed EV', '#2a9d8f'),
        ('failed_ev_count', 'failed EV', '#e76f51'),
        ('unresolved_ev_count', 'unresolved EV', '#6c757d'),
    ):
        counts_axis.plot(
            episode, _rolling(_series(rows, name), window),
            linewidth=1.7, color=color, label=label,
        )
    counts_axis.set_title('EV outcomes')
    counts_axis.set_xlabel('episode')
    counts_axis.set_ylabel('EV count')
    counts_axis.legend(fontsize=8)
    counts_axis.grid(alpha=0.25)

    profit_axis = axes[1, 1]
    _plot_with_rolling(
        profit_axis, episode, _series(rows, 'realised_service_profit'),
        'realised service profit', window, '#264653',
    )
    distance_axis = profit_axis.twinx()
    distance_axis.plot(
        episode, _rolling(_series(rows, 'dispatch_distance_km'), window),
        color='#f4a261', linewidth=1.7, label='dispatch distance (km)',
    )
    profit_axis.set_title('Realised profit and dispatch distance')
    profit_axis.set_xlabel('episode')
    profit_axis.set_ylabel('profit')
    distance_axis.set_ylabel('km')
    handles, labels = profit_axis.get_legend_handles_labels()
    second_handles, second_labels = distance_axis.get_legend_handles_labels()
    profit_axis.legend(handles + second_handles, labels + second_labels, fontsize=8)

    loss_axis = axes[2, 0]
    for name, label, color in (
        ('actor_loss', 'actor loss', '#457b9d'),
        ('critic_loss', 'critic loss', '#9d4edd'),
    ):
        loss_axis.plot(
            episode, _rolling(_series(rows, name), window),
            linewidth=1.7, color=color, label=label,
        )
    loss_axis.set_title('PPO losses')
    loss_axis.set_xlabel('episode')
    loss_axis.legend(fontsize=8)
    loss_axis.grid(alpha=0.25)

    diagnostic_axis = axes[2, 1]
    for name, label, color in (
        ('entropy', 'policy entropy', '#0077b6'),
        ('approx_kl', 'approx. KL', '#d62828'),
        ('clip_fraction', 'clip fraction', '#8338ec'),
    ):
        diagnostic_axis.plot(
            episode, _rolling(_series(rows, name), window),
            linewidth=1.7, color=color, label=label,
        )
    diagnostic_axis.set_title('PPO update diagnostics')
    diagnostic_axis.set_xlabel('episode')
    diagnostic_axis.legend(fontsize=8)
    diagnostic_axis.grid(alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv=None) -> None:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else input_dir / 'training_curves.png'
    )
    if args.window <= 0:
        raise ValueError('--window must be positive')
    render(input_dir, output, args.window)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
