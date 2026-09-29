"""Draw initial MCS/FCS positions and fixed dispatch points for one seed."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

import world as world_module
from centralized_env import CentralizedDispatchEnv
from config import (
    AREA_LAT_MAX, AREA_LAT_MIN, AREA_LON_MAX, AREA_LON_MIN, TRACK_DATA_PATH,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Plot centralized dispatch initial layout'
    )
    parser.add_argument('--seed', type=int, default=1003)
    parser.add_argument('--grid-rows', type=int, default=4)
    parser.add_argument('--grid-columns', type=int, default=4)
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_DIR / 'visual_results_centralized_random'
        / 'centralized_seed_1003_initial_layout.png',
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    env = CentralizedDispatchEnv(
        seed=args.seed,
        grid_rows=args.grid_rows,
        grid_columns=args.grid_columns,
        verbose=False,
    )
    env.reset()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mean_lat = 0.5 * (AREA_LAT_MIN + AREA_LAT_MAX)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    ax.set_xlim(AREA_LON_MIN, AREA_LON_MAX)
    ax.set_ylim(AREA_LAT_MIN, AREA_LAT_MAX)
    ax.set_aspect(1.0 / max(math.cos(math.radians(mean_lat)), 1e-6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(
        f'Centralized dispatch initial layout | seed {args.seed} | '
        f'{args.grid_rows}x{args.grid_columns} grid'
    )
    ax.grid(True, linewidth=0.55, alpha=0.3)

    lon_step = (AREA_LON_MAX - AREA_LON_MIN) / args.grid_columns
    lat_step = (AREA_LAT_MAX - AREA_LAT_MIN) / args.grid_rows
    for column in range(args.grid_columns + 1):
        x = AREA_LON_MIN + column * lon_step
        ax.axvline(x, color='#8a8a8a', linewidth=0.7, alpha=0.55, zorder=0)
    for row in range(args.grid_rows + 1):
        y = AREA_LAT_MIN + row * lat_step
        ax.axhline(y, color='#8a8a8a', linewidth=0.7, alpha=0.55, zorder=0)

    points = env.dispatch_points
    ax.scatter(
        points[:, 0], points[:, 1], marker='x', s=72, linewidths=1.8,
        c='#6a3d9a', label='Dispatch point', zorder=2,
    )
    for point_id, point in enumerate(points, start=1):
        ax.annotate(
            f'D{point_id}', xy=point, xytext=(4, 4),
            textcoords='offset points', fontsize=8, color='#4b1e70', zorder=3,
        )

    fcs_x = [fcs.pos[0] for fcs in env.world.FCSs]
    fcs_y = [fcs.pos[1] for fcs in env.world.FCSs]
    ax.scatter(
        fcs_x, fcs_y, marker='^', s=125, c='black', edgecolors='black',
        label='FCS', zorder=5,
    )
    for fcs in env.world.FCSs:
        ax.annotate(
            f'FCS {fcs.id}', xy=fcs.pos, xytext=(5, 5),
            textcoords='offset points', fontsize=8, color='black', zorder=6,
        )

    mcs_x = [mcs.pos[0] for mcs in env.world.MCSs]
    mcs_y = [mcs.pos[1] for mcs in env.world.MCSs]
    ax.scatter(
        mcs_x, mcs_y, marker='o', s=78, c='#2ca02c', edgecolors='white',
        linewidths=0.8, label='Initial MCS', zorder=7,
    )
    for mcs in env.world.MCSs:
        ax.annotate(
            f'M{mcs.id}', xy=mcs.pos, xytext=(5, -10),
            textcoords='offset points', fontsize=8, color='#176b2d', zorder=8,
        )

    ax.legend(loc='upper right', framealpha=0.95)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
