"""导出固定场景的微观调度 GIF、宏观覆盖 PNG 与统计表。

微观图只展示 AREA 内的 FCS、MCS 及其编号；宏观图展示整个仿真期间
MCS/FCS 通信范围的覆盖并集、Broken MCS 与失败 EV。默认使用 V5 的
Episode 300 checkpoint，并运行场景种子 1001、1002、1003。
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image

import world as world_module
from config import (
    AREA_LAT_MAX,
    AREA_LAT_MIN,
    AREA_LON_MAX,
    AREA_LON_MIN,
    COMM_RANGE,
    MAX_STEPS_PER_EPISODE,
    TRACK_DATA_PATH,
)
from core import MCS, euclidean_distance
from environment import MultiAgentEnv
from test import (
    RLDecisionPolicy,
    load_rl_agent,
    remember_charge_providers,
    resolve_device,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "training_results_v5" / "model_episode_300.pt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "visual_results_v5"
DEFAULT_SCENARIO_SEEDS = (1001, 1002, 1003)

# config.py 中的轨迹路径以 demo 目录为基准；转为绝对路径以避免启动目录影响。
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())

# 颜色和形状严格对应用户指定的状态编码。
FCS_COLOR = "#000000"
IDLE_MCS_COLOR = "#2ca02c"
TASK_MCS_COLOR = "#ff8c00"
RECHARGE_MCS_COLOR = "#808080"
BROKEN_MCS_COLOR = "#d62728"
FAILED_EV_COLOR = "#f28cb1"

# FCS 邻域统计和图中虚线领域均使用一个通信半径。
FCS_NEIGHBOR_RADIUS_KM = float(COMM_RANGE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 V5 模型生成微观调度 GIF 和宏观覆盖 PNG"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCENARIO_SEEDS),
        help="固定场景种子，默认 1001 1002 1003",
    )
    parser.add_argument(
        "--max-steps", type=int, default=MAX_STEPS_PER_EPISODE
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=250,
        help="GIF 每帧显示时间（毫秒）",
    )
    return parser.parse_args()


def set_scenario_seed(seed: int) -> None:
    """统一 Python、NumPy 和 Torch 随机种子，保证固定场景可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mcs_state(mcs: MCS) -> str:
    """按 broken > recharge > task > idle 的优先级返回唯一显示状态。"""
    if mcs.is_broken:
        return "broken"
    if mcs.is_recharging:
        return "recharge"
    if mcs.is_task:
        return "task"
    return "idle"


def nearby_mcs_counts(env: MultiAgentEnv) -> Dict[int, int]:
    """统计每个 FCS 在一个通信半径内的全部 MCS 数量。"""
    radius_m = FCS_NEIGHBOR_RADIUS_KM * 1000.0
    return {
        int(fcs.id): sum(
            euclidean_distance(
                fcs.pos[0], fcs.pos[1], mcs.pos[0], mcs.pos[1]
            )
            <= radius_m
            for mcs in env.world.MCSs
        )
        for fcs in env.world.FCSs
    }


def capture_snapshot(env: MultiAgentEnv, step: int) -> Dict:
    """保存绘图所需的最小状态，避免 GIF 渲染阶段依赖可变环境对象。"""
    return {
        "step": int(step),
        "mcs": [
            {
                "id": int(mcs.id),
                "x": float(mcs.pos[0]),
                "y": float(mcs.pos[1]),
                "state": mcs_state(mcs),
            }
            for mcs in env.world.MCSs
        ],
        # EV 仅在 fail_charge 变为 True 后进入失败列表。宏观图不显示 EV ID，
        # 只用粉色正方形标记其失败位置；微观图不消费该列表。
        "failed_evs": [
            {
                "x": float(ev.pos[0]),
                "y": float(ev.pos[1]),
            }
            for ev in env.world.EVs
            if ev.fail_charge
        ],
        # 宏观图边界需要覆盖所有 EV 在整个仿真过程中的位置；普通 EV
        # 不作为图标绘制，仅用于计算完整画布范围。
        "all_evs": [
            {
                "x": float(ev.pos[0]),
                "y": float(ev.pos[1]),
            }
            for ev in env.world.EVs
        ],
        "nearby_counts": nearby_mcs_counts(env),
    }


def write_count_csv(
    path: Path, seed: int, fcs_ids: Sequence[int], snapshots: Sequence[Dict]
) -> None:
    """写出单场景宽表：每行一个 step，每列一个 FCS。"""
    fieldnames = ["step"] + [f"FCS_{fcs_id}" for fcs_id in fcs_ids]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in snapshots:
            row = {"step": snapshot["step"]}
            row.update({
                f"FCS_{fcs_id}": snapshot["nearby_counts"][fcs_id]
                for fcs_id in fcs_ids
            })
            writer.writerow(row)


def _macro_plot_bounds(
    fcs_positions: Sequence[Sequence[float]], snapshots: Sequence[Dict]
):
    """返回覆盖 AREA、所有 EV、FCS 和 MCS 历史位置的宏观画布范围。"""
    xs = [
        float(AREA_LON_MIN),
        float(AREA_LON_MAX),
        *[float(pos[0]) for pos in fcs_positions],
    ]
    ys = [
        float(AREA_LAT_MIN),
        float(AREA_LAT_MAX),
        *[float(pos[1]) for pos in fcs_positions],
    ]
    for snapshot in snapshots:
        xs.extend(mcs["x"] for mcs in snapshot["mcs"])
        ys.extend(mcs["y"] for mcs in snapshot["mcs"])
        xs.extend(ev["x"] for ev in snapshot.get("all_evs", []))
        ys.extend(ev["y"] for ev in snapshot.get("all_evs", []))

    x_span = max(max(xs) - min(xs), 1e-4)
    y_span = max(max(ys) - min(ys), 1e-4)
    mean_lat = 0.5 * (min(ys) + max(ys))
    cos_lat = max(math.cos(math.radians(mean_lat)), 1e-6)
    # 画布不仅覆盖所有实体中心，也要完整容纳边缘实体的通信圆域。
    x_padding = max(0.06 * x_span, COMM_RANGE / (111.32 * cos_lat))
    y_padding = max(0.06 * y_span, COMM_RANGE / 111.32)
    return (
        min(xs) - x_padding,
        max(xs) + x_padding,
        min(ys) - y_padding,
        max(ys) + y_padding,
    )


def _boxes_overlap(first, second, padding_px: float = 2.0) -> bool:
    """判断两个显示坐标包围盒是否在保留间距后重叠。"""
    return not (
        first.x1 + padding_px <= second.x0
        or second.x1 + padding_px <= first.x0
        or first.y1 + padding_px <= second.y0
        or second.y1 + padding_px <= first.y0
    )


def place_mcs_id_labels(
    ax,
    mcs_rows: Sequence[Dict],
    label_artists: Dict[int, object],
    reserved_artists: Sequence[object],
) -> None:
    """在屏幕坐标中为 MCS ID 贪心选位，保证各 ID 包围盒不重叠。"""
    renderer = ax.figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    inverse = ax.transData.inverted()

    # FCS 文本也是不可占用区域，避免 MCS ID 与 FCS ID 混在一起。
    occupied = [
        artist.get_window_extent(renderer=renderer)
        for artist in reserved_artists
        if artist.get_visible()
    ]
    angle_count = 24
    base_angles = [2.0 * math.pi * index / angle_count for index in range(angle_count)]

    for row in sorted(mcs_rows, key=lambda item: int(item["id"])):
        mcs_id = int(row["id"])
        artist = label_artists[mcs_id]
        marker_px = ax.transData.transform((row["x"], row["y"]))
        # 不同 ID 从不同角度开始尝试，重合圆圈的标签会自然向四周展开。
        rotation = mcs_id % angle_count
        angles = base_angles[rotation:] + base_angles[:rotation]
        chosen_box = None

        for radius_px in range(28, 169, 12):
            for angle in angles:
                label_px = (
                    marker_px[0] + radius_px * math.cos(angle),
                    marker_px[1] + radius_px * math.sin(angle),
                )
                label_data = inverse.transform(label_px)
                artist.set_position(label_data)
                artist.set_visible(True)
                candidate_box = artist.get_window_extent(renderer=renderer)
                if (
                    candidate_box.x0 < axes_box.x0 + 2
                    or candidate_box.x1 > axes_box.x1 - 2
                    or candidate_box.y0 < axes_box.y0 + 2
                    or candidate_box.y1 > axes_box.y1 - 2
                ):
                    continue
                if any(_boxes_overlap(candidate_box, box) for box in occupied):
                    continue
                chosen_box = candidate_box
                break
            if chosen_box is not None:
                break

        if chosen_box is None:
            raise RuntimeError(f"无法为 MCS {mcs_id} 找到无重叠 ID 位置")
        occupied.append(chosen_box)


def _is_in_area(x: float, y: float) -> bool:
    """判断经纬度坐标是否位于配置的 AREA 闭区间内。"""
    return (
        AREA_LON_MIN <= float(x) <= AREA_LON_MAX
        and AREA_LAT_MIN <= float(y) <= AREA_LAT_MAX
    )


def render_micro_gif(
    path: Path,
    seed: int,
    fcs_positions: Sequence[Sequence[float]],
    fcs_ids: Sequence[int],
    snapshots: Sequence[Dict],
    frame_duration_ms: int,
    checkpoint_label: str = "checkpoint",
) -> None:
    """生成 AREA 内 FCS/MCS 状态微观 GIF，不绘制失败 EV。"""
    area_fcs = [
        (int(fcs_id), pos)
        for fcs_id, pos in zip(fcs_ids, fcs_positions)
        if _is_in_area(pos[0], pos[1])
    ]
    mean_lat = 0.5 * (AREA_LAT_MIN + AREA_LAT_MAX)
    cos_lat = max(math.cos(math.radians(mean_lat)), 1e-6)

    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=100)
    # 图例放到坐标轴外，避免遮挡 FCS/MCS 标记。
    fig.subplots_adjust(right=0.78)
    ax.set_xlim(AREA_LON_MIN, AREA_LON_MAX)
    ax.set_ylim(AREA_LAT_MIN, AREA_LAT_MAX)
    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.5, alpha=0.25)

    fcs_x = [pos[0] for _fcs_id, pos in area_fcs]
    fcs_y = [pos[1] for _fcs_id, pos in area_fcs]
    ax.scatter(
        fcs_x,
        fcs_y,
        marker="^",
        s=125,
        c=FCS_COLOR,
        edgecolors=FCS_COLOR,
        zorder=5,
    )

    # 用虚线椭圆表示球面坐标下的一个通信半径，便于肉眼核对统计口径。
    radius_km = FCS_NEIGHBOR_RADIUS_KM
    lat_radius_deg = radius_km / 111.32
    lon_radius_deg = radius_km / (111.32 * cos_lat)
    fcs_label_artists = []
    for fcs_id, pos in area_fcs:
        ax.add_patch(Ellipse(
            xy=pos,
            width=2.0 * lon_radius_deg,
            height=2.0 * lat_radius_deg,
            fill=False,
            edgecolor=FCS_COLOR,
            linewidth=0.8,
            linestyle="--",
            alpha=0.22,
            zorder=1,
        ))
        fcs_label_artists.append(ax.annotate(
            f"FCS {fcs_id}",
            xy=pos,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=FCS_COLOR,
            zorder=6,
        ))

    state_style = {
        "idle": (IDLE_MCS_COLOR, "Idle MCS"),
        "task": (TASK_MCS_COLOR, "Task MCS"),
        "recharge": (RECHARGE_MCS_COLOR, "Recharge MCS"),
        "broken": (BROKEN_MCS_COLOR, "Broken MCS"),
    }
    state_zorders = {
        # 所有 MCS 圆点都绘制在失败 EV 方块之上。
        "idle": 8,
        "task": 8,
        # recharge MCS 必须覆盖在 FCS 三角形和其文字之上。
        "recharge": 8,
        "broken": 8,
    }
    state_scatters = {
        state: ax.scatter(
            [], [], marker="o", s=72, c=color, edgecolors="white",
            linewidths=0.6, zorder=state_zorders[state]
        )
        for state, (color, _label) in state_style.items()
    }
    mcs_ids = sorted({
        int(mcs["id"])
        for snapshot in snapshots for mcs in snapshot["mcs"]
    })
    mcs_id_labels = {
        mcs_id: ax.text(
            0.0,
            0.0,
            f"MCS{mcs_id}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            zorder=10,
            path_effects=[
                path_effects.withStroke(linewidth=2.5, foreground="white")
            ],
        )
        for mcs_id in mcs_ids
    }
    legend_handles = [
        Line2D([], [], marker="^", linestyle="None", markersize=8,
               markerfacecolor=FCS_COLOR, markeredgecolor=FCS_COLOR,
               label="FCS"),
        *[
            Line2D([], [], marker="o", linestyle="None", markersize=7,
                   markerfacecolor=color, markeredgecolor="white", label=label)
            for color, label in state_style.values()
        ],
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )
    frames: List[Image.Image] = []
    try:
        for snapshot in snapshots:
            area_mcs_rows = [
                mcs for mcs in snapshot["mcs"]
                if _is_in_area(mcs["x"], mcs["y"])
            ]
            for state, scatter in state_scatters.items():
                positions = [
                    (mcs["x"], mcs["y"])
                    for mcs in area_mcs_rows if mcs["state"] == state
                ]
                scatter.set_offsets(
                    np.asarray(positions, dtype=float).reshape(-1, 2)
                    if positions else np.empty((0, 2), dtype=float)
                )

            counts = snapshot["nearby_counts"]
            count_line = "MCS within R: " + "  ".join(
                f"FCS {fcs_id}={counts[fcs_id]}"
                for fcs_id, _pos in area_fcs
            )
            ax.set_title(
                f"{checkpoint_label} | seed {seed} | step {snapshot['step']}\n"
                f"{count_line}"
            )
            # 先刷新坐标变换，再按真实像素包围盒布置 ID。
            fig.canvas.draw()
            for artist in mcs_id_labels.values():
                artist.set_visible(False)
            place_mcs_id_labels(
                ax,
                area_mcs_rows,
                mcs_id_labels,
                fcs_label_artists,
            )
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
            frame = Image.fromarray(rgba, mode="RGBA").convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=256
            )
            frames.append(frame)

        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=max(int(frame_duration_ms), 20),
            loop=0,
            optimize=False,
            disposal=2,
        )
    finally:
        for frame in frames:
            frame.close()
        plt.close(fig)


def _coverage_counts(
    bounds: Sequence[float],
    centers: Sequence[Sequence[float]],
    resolution: int = 900,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """累计每个栅格被实体通信圆覆盖的次数。

    ``centers`` 中的每次出现都代表一次覆盖。同一 MCS 在不同 step
    停留于相同位置时仍逐次累加，确保灰度反映真实 MCS-step 覆盖频次。
    """
    x_min, x_max, y_min, y_max = map(float, bounds)
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    aspect = x_span / y_span
    if aspect >= 1.0:
        nx = resolution
        ny = max(int(round(resolution / aspect)), 300)
    else:
        ny = resolution
        nx = max(int(round(resolution * aspect)), 300)

    x_values = np.linspace(x_min, x_max, nx)
    y_values = np.linspace(y_min, y_max, ny)
    counts = np.zeros((ny, nx), dtype=np.uint32)
    mean_lat = 0.5 * (y_min + y_max)
    km_per_lon = 111.32 * max(math.cos(math.radians(mean_lat)), 1e-6)
    km_per_lat = 111.32
    radius_km = float(COMM_RANGE)

    # 先聚合同一栅格中心的出现次数，再一次性加权写入，避免静止 MCS
    # 在 200 个 step 中重复计算相同圆形掩码。
    grid_center_weights: Dict[tuple[int, int], int] = {}
    for center_x, center_y in centers:
        ix = int(np.clip(round((center_x - x_min) / x_span * (nx - 1)), 0, nx - 1))
        iy = int(np.clip(round((center_y - y_min) / y_span * (ny - 1)), 0, ny - 1))
        key = (ix, iy)
        grid_center_weights[key] = grid_center_weights.get(key, 0) + 1

    x_radius = radius_km / km_per_lon
    y_radius = radius_km / km_per_lat
    for (ix, iy), weight in grid_center_weights.items():
        center_x = x_values[ix]
        center_y = y_values[iy]
        x0 = max(int(np.searchsorted(x_values, center_x - x_radius)) - 1, 0)
        x1 = min(int(np.searchsorted(x_values, center_x + x_radius)) + 1, nx)
        y0 = max(int(np.searchsorted(y_values, center_y - y_radius)) - 1, 0)
        y1 = min(int(np.searchsorted(y_values, center_y + y_radius)) + 1, ny)
        local_x = (x_values[x0:x1] - center_x) * km_per_lon
        local_y = (y_values[y0:y1] - center_y) * km_per_lat
        local_covered = (
            local_y[:, None] ** 2 + local_x[None, :] ** 2
            <= radius_km ** 2
        )
        counts[y0:y1, x0:x1] += local_covered.astype(np.uint32) * weight
    return x_values, y_values, counts


def render_macro_png(
    path: Path,
    seed: int,
    fcs_positions: Sequence[Sequence[float]],
    snapshots: Sequence[Dict],
    checkpoint_label: str = "checkpoint",
) -> None:
    """生成全场景累计服务覆盖宏观 PNG。"""
    bounds = _macro_plot_bounds(fcs_positions, snapshots)
    x_min, x_max, y_min, y_max = bounds
    mean_lat = 0.5 * (y_min + y_max)
    cos_lat = max(math.cos(math.radians(mean_lat)), 1e-6)

    # FCS 是固定基础覆盖，仅绘制一层浅灰；MCS 按 step 顺序逐次累计，
    # 每覆盖一次对应栅格的灰度计数增加 1。
    fcs_centers = [tuple(map(float, pos)) for pos in fcs_positions]
    mcs_coverage_centers = []
    for snapshot in snapshots:
        mcs_coverage_centers.extend(
            (float(mcs["x"]), float(mcs["y"]))
            for mcs in snapshot["mcs"]
        )
    x_values, y_values, fcs_coverage_counts = _coverage_counts(
        bounds, fcs_centers
    )
    _, _, mcs_coverage_counts = _coverage_counts(
        bounds, mcs_coverage_centers
    )

    broken_positions = sorted({
        (float(mcs["x"]), float(mcs["y"]))
        for snapshot in snapshots
        for mcs in snapshot["mcs"]
        if mcs["state"] == "broken"
    })
    failed_ev_positions = sorted({
        (float(ev["x"]), float(ev["y"]))
        for snapshot in snapshots
        for ev in snapshot.get("failed_evs", [])
    })

    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    # FCS 固定覆盖作为最浅的灰色底层，不参与 MCS 频次色条。
    ax.contourf(
        x_values,
        y_values,
        (fcs_coverage_counts > 0).astype(np.uint8),
        levels=[0.5, 1.5],
        colors=["#e3e3e3"],
        alpha=0.8,
        antialiased=True,
        zorder=1,
    )
    # 只显示累计次数大于 0 的 MCS 覆盖区。一次覆盖使用浅灰，累计次数
    # 越高逐步加深；线性 Normalize 保留“每覆盖一次增加一点灰度”的语义。
    visible_mcs_counts = np.ma.masked_equal(mcs_coverage_counts, 0)
    max_mcs_count = int(mcs_coverage_counts.max())
    serve_coverage_cmap = LinearSegmentedColormap.from_list(
        "serve_coverage_greys",
        plt.cm.Greys(np.linspace(0.28, 0.88, 256)),
    )
    coverage_norm = Normalize(vmin=1, vmax=max(max_mcs_count, 2))
    coverage_mesh = ax.pcolormesh(
        x_values,
        y_values,
        visible_mcs_counts,
        cmap=serve_coverage_cmap,
        norm=coverage_norm,
        shading="auto",
        alpha=0.82,
        rasterized=True,
        zorder=2,
    )
    colorbar = fig.colorbar(coverage_mesh, ax=ax, pad=0.02, shrink=0.86)
    colorbar.set_label("MCS Serve Coverage Count (MCS-steps)")
    ax.add_patch(Rectangle(
        (AREA_LON_MIN, AREA_LAT_MIN),
        AREA_LON_MAX - AREA_LON_MIN,
        AREA_LAT_MAX - AREA_LAT_MIN,
        fill=False,
        edgecolor="black",
        linewidth=1.2,
        linestyle="--",
        zorder=5,
    ))
    ax.scatter(
        [pos[0] for pos in fcs_positions],
        [pos[1] for pos in fcs_positions],
        marker="^",
        s=75,
        c=FCS_COLOR,
        edgecolors=FCS_COLOR,
        zorder=6,
    )
    if broken_positions:
        ax.scatter(
            [pos[0] for pos in broken_positions],
            [pos[1] for pos in broken_positions],
            marker="o",
            s=45,
            c=BROKEN_MCS_COLOR,
            edgecolors="white",
            linewidths=0.6,
            zorder=7,
        )
    if failed_ev_positions:
        ax.scatter(
            [pos[0] for pos in failed_ev_positions],
            [pos[1] for pos in failed_ev_positions],
            marker="s",
            s=20,
            c=FAILED_EV_COLOR,
            edgecolors=FAILED_EV_COLOR,
            linewidths=0.4,
            zorder=6,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{checkpoint_label} | seed {seed} | cumulative service coverage"
    )
    ax.grid(True, linewidth=0.45, alpha=0.2, zorder=0)
    legend_handles = [
        Line2D([], [], linewidth=8, color="#e3e3e3",
               alpha=0.8, label="FCS Base Coverage"),
        Line2D([], [], linestyle="--", linewidth=1.2, color="black",
               label="Simulation Area"),
        Line2D([], [], marker="^", linestyle="None", markersize=7,
               markerfacecolor=FCS_COLOR, markeredgecolor=FCS_COLOR,
               label="FCS"),
        Line2D([], [], marker="o", linestyle="None", markersize=6,
               markerfacecolor=BROKEN_MCS_COLOR, markeredgecolor="white",
               label="Broken MCS"),
        Line2D([], [], marker="s", linestyle="None", markersize=5,
               markerfacecolor=FAILED_EV_COLOR,
               markeredgecolor=FAILED_EV_COLOR, label="Failed EV"),
    ]
    ax.legend(handles=legend_handles, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_scenario(
    seed: int,
    max_steps: int,
    policy: RLDecisionPolicy,
    output_dir: Path,
    frame_duration_ms: int,
    checkpoint_label: str,
) -> tuple[List[Dict], Dict]:
    """运行场景并生成微观 GIF、宏观 PNG、邻域计数和业务指标。"""
    set_scenario_seed(seed)
    env = MultiAgentEnv(seed)
    env.world.verbose = False
    observations = env.reset()

    provider_by_ev_id: Dict[int, str] = {}
    remember_charge_providers(env.world.EVs, provider_by_ev_id)
    snapshots = [capture_snapshot(env, step=0)]

    executed_steps = 0
    for step in range(1, max_steps + 1):
        acting_agents = list(env.world.agents)
        action_n = policy.build_actions(env, acting_agents, observations)
        if len(action_n) != len(acting_agents):
            raise RuntimeError("action_n 与当前 acting_agents 数量不一致")
        observations, _, _, _ = env.step(action_n)
        policy.synchronize()
        executed_steps = step
        remember_charge_providers(env.world.EVs, provider_by_ev_id)
        snapshots.append(capture_snapshot(env, step=step))
        if env.world.get_done():
            break

    fcs_ids = [int(fcs.id) for fcs in env.world.FCSs]
    fcs_positions = [list(fcs.pos) for fcs in env.world.FCSs]
    stem = f"scenario_seed_{seed}"
    write_count_csv(
        output_dir / f"{stem}_fcs_nearby_mcs.csv", seed, fcs_ids, snapshots
    )
    render_micro_gif(
        output_dir / f"{stem}_micro.gif",
        seed,
        fcs_positions,
        fcs_ids,
        snapshots,
        frame_duration_ms,
        checkpoint_label,
    )
    render_macro_png(
        output_dir / f"{stem}_macro.png",
        seed,
        fcs_positions,
        snapshots,
        checkpoint_label,
    )

    successful_evs = [ev for ev in env.world.EVs if ev.is_charged]
    failed_count = sum(bool(ev.fail_charge) for ev in env.world.EVs)
    mcs_success = sum(
        provider_by_ev_id.get(int(ev.id)) == "MCS" for ev in successful_evs
    )
    fcs_success = sum(
        provider_by_ev_id.get(int(ev.id)) == "FCS" for ev in successful_evs
    )
    finished_count = len(successful_evs) + failed_count
    summary = {
        "scenario_seed": seed,
        "steps": executed_steps,
        "ev_success_count": len(successful_evs),
        "ev_failure_count": failed_count,
        "ev_charge_success_ratio": (
            len(successful_evs) / finished_count if finished_count else 0.0
        ),
        "successful_ev_mcs_count": mcs_success,
        "successful_ev_fcs_count": fcs_success,
        "broken_mcs_count": sum(mcs.is_broken for mcs in env.world.MCSs),
    }
    for fcs_id in fcs_ids:
        values = [row["nearby_counts"][fcs_id] for row in snapshots]
        summary[f"FCS_{fcs_id}_nearby_mcs_mean"] = float(np.mean(values))
        summary[f"FCS_{fcs_id}_nearby_mcs_max"] = int(max(values))
    return snapshots, summary


def write_rows(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    agent, metadata = load_rl_agent(
        checkpoint, hidden_dim=args.hidden_dim, device=device
    )
    policy = RLDecisionPolicy(agent)
    max_steps = min(max(int(args.max_steps), 1), MAX_STEPS_PER_EPISODE)
    checkpoint_episode = int(metadata.get("episode", 0))
    parent_name = checkpoint.parent.name
    if parent_name.startswith("training_results_v"):
        model_label = parent_name.removeprefix("training_results_").upper()
    elif parent_name == "training_results":
        model_label = "V1"
    else:
        model_label = parent_name
    checkpoint_label = (
        f"{model_label} checkpoint episode {checkpoint_episode}"
        if checkpoint_episode > 0
        else f"{model_label} checkpoint"
    )

    combined_rows = []
    summaries = []
    for seed in args.seeds:
        print(f"正在仿真固定场景 seed={seed} ...")
        snapshots, summary = run_scenario(
            seed=int(seed),
            max_steps=max_steps,
            policy=policy,
            output_dir=output_dir,
            frame_duration_ms=args.frame_duration_ms,
            checkpoint_label=checkpoint_label,
        )
        for snapshot in snapshots:
            row = {"scenario_seed": int(seed), "step": snapshot["step"]}
            row.update({
                f"FCS_{fcs_id}": value
                for fcs_id, value in snapshot["nearby_counts"].items()
            })
            combined_rows.append(row)
        summary["checkpoint_episode"] = metadata.get("episode", 300)
        summaries.append(summary)
        print(
            f"seed={seed} 完成：success={summary['ev_charge_success_ratio']:.4f}, "
            f"MCS/FCS={summary['successful_ev_mcs_count']}/"
            f"{summary['successful_ev_fcs_count']}"
        )

    write_rows(output_dir / "fcs_nearby_mcs_all_scenarios.csv", combined_rows)
    write_rows(output_dir / "scenario_summary.csv", summaries)
    print(f"微观 GIF、宏观 PNG 与统计表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
