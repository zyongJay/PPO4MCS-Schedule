"""
非 RL 仿真测试脚本。

调度策略：
1. IEV 沿自身轨迹移动。
2. MCS 从通信范围内随机选择一个 quasi EV 前往跟踪。
3. 低电量 MCS 使用 RechargeMatcher 前往 FCS 补电。
4. 为每辆 EV/MCS 保存逐 step 属性与观测 CSV，并生成汇总 CSV 和调度 GIF。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

import world as world_module
from config import (
    AREA_LAT_MAX,
    AREA_LAT_MIN,
    AREA_LON_MAX,
    AREA_LON_MIN,
    MAX_STEPS_PER_EPISODE,
    MCS_RECHARGE_THRESHOLD,
    TRACK_DATA_PATH,
)
from core import EV, MCS, euclidean_distance
from environment import MultiAgentEnv
from matching import RechargeMatcher
from observation import MCS_HIGH_FEATURE_NAMES

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "simulation_results"

# config.py 中的轨迹路径以 demo 目录为基准。转换为绝对路径后，
# 从项目根目录或 demo 目录运行本脚本都能正确读取轨迹数据。
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


def to_printable(value: Any) -> Any:
    """将观测中的 NumPy 对象转换为适合写入 CSV 的 Python 对象。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_printable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_printable(item) for item in value]
    return value


def capture_frame(env: MultiAgentEnv) -> dict:
    """记录当前 step 中全部实体的位置，供 GIF 绘制使用。"""
    ev_positions = {}
    for ev in env.world.EVs:
        ev_positions[ev.id] = [float(ev.pos[0]), float(ev.pos[1])]

    mcs_positions = {}
    for mcs in env.world.MCSs:
        mcs_positions[mcs.id] = [float(mcs.pos[0]), float(mcs.pos[1])]

    fcs_positions = {}
    for fcs in env.world.FCSs:
        fcs_positions[fcs.id] = [float(fcs.pos[0]), float(fcs.pos[1])]

    return {
        "step": env.world.current_step,
        "ev": ev_positions,
        "mcs": mcs_positions,
        "fcs": fcs_positions,
    }


def save_animation(
        frames: list[dict],
        save_path: Path,
        tracked_ev_ids: list[int],
        tracked_mcs_ids: list[int],
) -> None:
    """将选定 EV 和 MCS 的逐 step 轨迹保存为 GIF。"""
    figure, axes = plt.subplots(figsize=(9, 7))
    # axes.set_xlim(AREA_LON_MIN, AREA_LON_MAX)
    # axes.set_ylim(AREA_LAT_MIN, AREA_LAT_MAX)
    axes.set_xlim(103.9, AREA_LON_MAX)
    axes.set_ylim(30.5, AREA_LAT_MAX)
    axes.set_xlabel("Longitude")
    axes.set_ylabel("Latitude")
    axes.grid(alpha=0.25)

    ev_lines = {}
    for ev_id in tracked_ev_ids:
        line = axes.plot(
            [],
            [],
            "-o",
            markersize=3,
            linewidth=1.2,
            label=f"EV-{ev_id}",
        )[0]
        ev_lines[ev_id] = line

    mcs_lines = {}
    for mcs_id in tracked_mcs_ids:
        line = axes.plot(
            [],
            [],
            "--s",
            markersize=5,
            linewidth=1.6,
            label=f"MCS-{mcs_id}",
        )[0]
        mcs_lines[mcs_id] = line

    fcs_positions = list(frames[0]["fcs"].values())
    if fcs_positions:
        axes.scatter(
            [position[0] for position in fcs_positions],
            [position[1] for position in fcs_positions],
            marker="^",
            s=120,
            color="black",
            label="FCS",
        )

    axes.legend(fontsize=8)

    def update(frame_index: int) -> list:
        artists = []

        for ev_id, line in ev_lines.items():
            positions = []
            for frame in frames[: frame_index + 1]:
                positions.append(frame["ev"][ev_id])
            line.set_data(
                [position[0] for position in positions],
                [position[1] for position in positions],
            )
            artists.append(line)

        for mcs_id, line in mcs_lines.items():
            positions = []
            for frame in frames[: frame_index + 1]:
                positions.append(frame["mcs"][mcs_id])
            line.set_data(
                [position[0] for position in positions],
                [position[1] for position in positions],
            )
            artists.append(line)

        step = frames[frame_index]["step"]
        axes.set_title(f"Dynamic scheduling | step {step}")
        return artists

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=250,
        repeat=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=4))
    plt.close(figure)


if __name__ == "__main__":
    # 1. 读取运行参数。
    parser = argparse.ArgumentParser(description="运行非 RL 多智能体充电调度仿真")
    parser.add_argument("--steps", type=int, default=200, help="仿真 step 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="CSV 和 GIF 输出目录",
    )
    parser.add_argument(
        "--tracked-count",
        type=int,
        default=3,
        help="GIF 中最多跟踪的 EV 和 MCS 数量",
    )
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps 必须大于 0")
    if args.tracked_count < 0:
        parser.error("--tracked-count 不能小于 0")

    # 2. 初始化随机数、输出目录和仿真环境。
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    env = MultiAgentEnv(args.seed)
    # 不在终端逐 step 输出匹配信息、观测或奖励。
    env.world.verbose = False
    recharge_matcher = RechargeMatcher()
    obs_n = env.reset()

    statistics_rows = []
    frames = []
    max_steps = min(args.steps, MAX_STEPS_PER_EPISODE)

    # 每个实体单独维护时序记录，最后分别写入 CSV。
    ev_history = {}
    for ev in env.world.EVs:
        ev_history[ev.id] = []

    mcs_history = {}
    for mcs in env.world.MCSs:
        mcs_history[mcs.id] = []

    # reward_n 与执行上一轮动作的 agents 对齐。step 0 尚未产生奖励。
    latest_reward_by_agent = {}
    # 保存上一轮送入 World.update() 的动作，仅用于写入 history CSV。
    latest_action_by_agent = {}

    # 3. 记录当前状态，然后执行下一 step。多循环一次以保留最终状态。
    for simulation_index in range(max_steps + 1):
        current_agents = list(env.world.agents)
        current_obs_by_agent = {}
        for index, agent in enumerate(current_agents):
            if index < len(obs_n):
                current_obs_by_agent[agent] = obs_n[index]

        # 3.1 记录每辆 EV 的位置、电量、状态、收益和当前观测。
        for ev in env.world.EVs:
            observation = current_obs_by_agent.get(ev, {})
            reward = latest_reward_by_agent.get(ev, 0.0)
            last_action = latest_action_by_agent.get(ev, {})

            previous_remain = float(ev.remain)
            previous_longitude = float(ev.pos[0])
            previous_latitude = float(ev.pos[1])
            if ev_history[ev.id]:
                previous_row = ev_history[ev.id][-1]
                previous_remain = previous_row["remain_kwh"]
                previous_longitude = previous_row["longitude"]
                previous_latitude = previous_row["latitude"]
            remain_change = float(ev.remain) - previous_remain
            moved_distance_km = euclidean_distance(
                previous_longitude,
                previous_latitude,
                ev.pos[0],
                ev.pos[1],
            ) / 1000.0

            if ev.is_fail:
                state = "fail"
            elif ev.is_charged and ev.charge_pos is not None:
                state = "charging"
            elif ev.is_charged:
                state = "success"
            elif ev.is_iev:
                state = "iev"
            elif ev.is_quasi:
                state = "quasi"
            else:
                state = "normal"

            ev_row = {
                "step": env.world.current_step,
                "ev_id": ev.id,
                "state": state,
                "is_agent": ev in current_obs_by_agent,
                "longitude": float(ev.pos[0]),
                "latitude": float(ev.pos[1]),
                "remain_kwh": float(ev.remain),
                "remain_change_kwh": remain_change,
                "energy_used_kwh": max(previous_remain - float(ev.remain), 0.0),
                "moved_distance_km": moved_distance_km,
                "action_target_longitude": last_action.get("target_pos", [None, None])[0],
                "action_target_latitude": last_action.get("target_pos", [None, None])[1],
                "reward": float(reward),
                "total_reward": float(ev.total_reward),
                "need_power_kwh": float(ev.need_power),
                "need_charge": ev.need_charge,
                "is_normal": ev.is_normal,
                "is_quasi": ev.is_quasi,
                "is_iev": ev.is_iev,
                "is_charged": ev.is_charged,
                "fail_charge": ev.fail_charge,
                "arrived": ev.arrived,
                "track_index": ev.track_index,
                "wait_time_steps": ev.wait_time_steps,
                "total_wait_time_min": float(ev.total_wait_time_min),
                "total_extra_dist_km": float(ev.total_extra_dist_km),
                "expense": float(ev.expense),
                "charge_provider_type": ev.charge_provider_type,
                "charge_provider_id": ev.charge_provider_id,
                "charge_power_kwh": float(ev.charge_power_kwh),
                "charge_time_remain_min": float(ev.charge_time_remain_min),
                "near_quasi_count": len(ev.near_quasi),
                "near_iev_count": len(ev.near_iev),
                "near_idle_mcs_count": len(ev.near_idle_mcs),
                "near_task_mcs_count": len(ev.near_task_mcs),
                "near_available_fcs_count": len(ev.near_available_fcs),
                "near_busy_fcs_count": len(ev.near_busy_fcs),
                "obs_self": json.dumps(
                    to_printable(observation.get("obs_self", [])),
                    ensure_ascii=False,
                ),
                "obs_tgt": json.dumps(
                    to_printable(observation.get("obs_tgt", [])),
                    ensure_ascii=False,
                ),
                "obs_mask": json.dumps(
                    to_printable(observation.get("mask", [])),
                    ensure_ascii=False,
                ),
            }
            ev_history[ev.id].append(ev_row)

        # 3.2 记录每辆 MCS 的位置、电量、任务、运营指标和当前观测。
        for mcs in env.world.MCSs:
            observation = current_obs_by_agent.get(mcs, {})
            reward_components = env.world.last_mcs_reward_components.get(mcs.id, {})
            reward = reward_components.get('total', 0.0)
            event = env.world.mcs_step_events.get(mcs.id, {})
            last_action = latest_action_by_agent.get(mcs, {})

            previous_remain = float(mcs.remain)
            previous_longitude = float(mcs.pos[0])
            previous_latitude = float(mcs.pos[1])
            if mcs_history[mcs.id]:
                previous_row = mcs_history[mcs.id][-1]
                previous_remain = previous_row["remain_kwh"]
                previous_longitude = previous_row["longitude"]
                previous_latitude = previous_row["latitude"]
            remain_change = float(mcs.remain) - previous_remain
            moved_distance_km = euclidean_distance(
                previous_longitude,
                previous_latitude,
                mcs.pos[0],
                mcs.pos[1],
            ) / 1000.0

            if mcs.is_broken:
                state = "broken"
            elif mcs.is_recharging:
                state = "recharging"
            elif mcs.is_task:
                state = "task"
            else:
                state = "idle"

            mcs_row = {
                "step": env.world.current_step,
                "mcs_id": mcs.id,
                "state": state,
                "is_agent": mcs in current_obs_by_agent,
                "longitude": float(mcs.pos[0]),
                "latitude": float(mcs.pos[1]),
                "remain_kwh": float(mcs.remain),
                "remain_change_kwh": remain_change,
                "energy_used_kwh": max(previous_remain - float(mcs.remain), 0.0),
                "moved_distance_km": moved_distance_km,
                "action_mode": last_action.get("mode", ""),
                "action_target_longitude": last_action.get("target_pos", [None, None])[0],
                "action_target_latitude": last_action.get("target_pos", [None, None])[1],
                "reward": float(reward),
                "reward_service": float(reward_components.get("service", 0.0)),
                "reward_serve_attraction": float(
                    reward_components.get("serve_attraction", 0.0)
                ),
                "reward_serve_competition": float(
                    reward_components.get("serve_competition", 0.0)
                ),
                "reward_serve_potential_improvement": float(
                    reward_components.get("serve_potential_improvement", 0.0)
                ),
                "reward_recharge": float(reward_components.get("recharge", 0.0)),
                "reward_movement": float(reward_components.get("movement", 0.0)),
                "reward_wait": float(reward_components.get("wait", 0.0)),
                "reward_recharge_match_failure": float(
                    reward_components.get("recharge_match_failure", 0.0)
                ),
                "reward_broken": float(reward_components.get("broken", 0.0)),
                "reward_battery_potential": float(
                    reward_components.get("battery_potential", 0.0)
                ),
                "total_reward": float(mcs.total_reward),
                "event_mode": event.get("mode", ""),
                "event_requested_mode": event.get("requested_mode", ""),
                "event_recharge_matched": bool(event.get("recharge_matched", False)),
                "event_waited": bool(event.get("waited", False)),
                "event_movement_energy_kwh": float(
                    event.get("movement_energy_kwh", 0.0)
                ),
                "event_service_kwh": float(event.get("service_kwh", 0.0)),
                "event_recharged_kwh": float(event.get("recharged_kwh", 0.0)),
                "event_previous_attraction": float(
                    event.get("previous_attraction", 0.0)
                ),
                "event_post_attraction": float(event.get("post_attraction", 0.0)),
                "event_previous_competition": float(
                    event.get("previous_competition", 0.0)
                ),
                "event_post_competition": float(
                    event.get("post_competition", 0.0)
                ),
                "event_previous_spatial_potential": float(
                    event.get("previous_spatial_potential", 0.0)
                ),
                "event_post_spatial_potential": float(
                    event.get("post_spatial_potential", 0.0)
                ),
                "event_newly_broken": bool(event.get("newly_broken", False)),
                "is_idle": mcs.is_idle,
                "is_task": mcs.is_task,
                "is_recharging": mcs.is_recharging,
                "is_broken": mcs.is_broken,
                "is_arrive": mcs.is_arrive,
                "current_target_type": mcs.current_target_type,
                "current_target_id": mcs.current_target_id,
                "charge_power_kwh": float(mcs.charge_power_kwh),
                "charge_time_remain_min": float(mcs.charge_time_remain_min),
                "total_energy_consumed": float(mcs.total_energy_consumed),
                "total_charged_kwh": float(mcs.total_charged_kwh),
                "total_cost": float(mcs.total_cost),
                "total_profit": float(mcs.total_profit),
                "total_idle_time_min": float(mcs.total_idle_time_min),
                "near_quasi_count": len(mcs.near_quasi),
                "near_iev_count": len(mcs.near_iev),
                "near_idle_mcs_count": len(mcs.near_idle_mcs),
                "near_task_mcs_count": len(mcs.near_task_mcs),
                "near_available_fcs_count": len(mcs.near_available_fcs),
                "near_busy_fcs_count": len(mcs.near_busy_fcs),
                "obs_self": json.dumps(
                    to_printable(observation.get("obs_self", [])),
                    ensure_ascii=False,
                ),
                "obs_tgt": json.dumps(
                    to_printable(observation.get("obs_tgt", [])),
                    ensure_ascii=False,
                ),
                "obs_mask": json.dumps(
                    to_printable(observation.get("mask", [])),
                    ensure_ascii=False,
                ),
                "high_state": json.dumps(
                    to_printable(observation.get("high_state", [])),
                    ensure_ascii=False,
                ),
                "high_action_mask": json.dumps(
                    to_printable(observation.get("high_action_mask", [])),
                    ensure_ascii=False,
                ),
                "low_self_state": json.dumps(
                    to_printable(observation.get("low_self_state", [])),
                    ensure_ascii=False,
                ),
                "low_candidates": json.dumps(
                    to_printable(observation.get("low_candidates", [])),
                    ensure_ascii=False,
                ),
                "low_candidate_mask": json.dumps(
                    to_printable(observation.get("low_candidate_mask", [])),
                    ensure_ascii=False,
                ),
                "candidate_ids": json.dumps(
                    to_printable(observation.get("candidate_ids", [])),
                    ensure_ascii=False,
                ),
            }
            high_state = observation.get("high_state", [])
            for feature_index, feature_name in enumerate(MCS_HIGH_FEATURE_NAMES):
                value = high_state[feature_index] if feature_index < len(high_state) else None
                mcs_row[f"high_{feature_name}"] = value
            high_mask = observation.get("high_action_mask", [])
            for action_index, action_name in enumerate(("serve", "recharge", "wait")):
                value = high_mask[action_index] if action_index < len(high_mask) else None
                mcs_row[f"mask_{action_name}_valid"] = value
            mcs_history[mcs.id].append(mcs_row)

        frames.append(capture_frame(env))

        # 已记录最终状态，不再执行新的动作。
        if simulation_index == max_steps or env.world.get_done():
            break

        acting_agents = list(env.world.agents)

        # 4. 低电量且空闲的 MCS 优先申请 FCS 补电。
        recharge_mcss = []
        for agent in acting_agents:
            if not isinstance(agent, MCS):
                continue
            if not agent.is_idle:
                continue
            if agent.remain >= MCS_RECHARGE_THRESHOLD:
                continue
            observation = current_obs_by_agent.get(agent, {})
            high_mask = observation.get("high_action_mask", [])
            recharge_valid = len(high_mask) > 1 and bool(high_mask[1])
            if not recharge_valid:
                continue
            recharge_mcss.append(agent)

        recharge_results = recharge_matcher.match_all(
            recharge_mcss,
            env.world.FCSs,
        )
        recharge_request_mcs_ids = {mcs.id for mcs in recharge_mcss}
        recharging_mcs_ids = set()
        for result in recharge_results:
            if result.get("success"):
                recharging_mcs_ids.add(result["mcs_id"])

        # 5. 按 acting_agents 的顺序构造动作。
        action_n = []

        for agent in acting_agents:
            if isinstance(agent, EV):
                # IEV 沿自身轨迹移动一个路点。
                if agent.track and agent.track_index + 1 < len(agent.track):
                    next_position = agent.track[agent.track_index + 1]
                    target_position = [
                        float(next_position[0]),
                        float(next_position[1]),
                    ]
                else:
                    target_position = [
                        float(agent.pos[0]),
                        float(agent.pos[1]),
                    ]

                action_n.append({"target_pos": target_position})
                continue

            if agent.id in recharging_mcs_ids:
                action_n.append(
                    {
                        "mode": "Recharge",
                        "requested_mode": "Recharge",
                        "recharge_matched": True,
                        "target_pos": list(agent.current_target_pos),
                    }
                )
                continue

            if agent.id in recharge_request_mcs_ids:
                # A valid request may still lose slot competition.  Its actual
                # physical outcome for this step is Wait; retry next step.
                action_n.append(
                    {
                        "mode": "Wait",
                        "requested_mode": "Recharge",
                        "recharge_matched": False,
                        "target_pos": list(agent.pos),
                    }
                )
                continue

            # near_quasi 已由 World 按通信范围构建。
            quasi_candidates = []
            for ev in agent.near_quasi:
                if ev.is_quasi:
                    quasi_candidates.append(ev)

            if quasi_candidates:
                target_ev = rng.choice(quasi_candidates)
                target_position = [
                    float(target_ev.pos[0]),
                    float(target_ev.pos[1]),
                ]
                action_n.append(
                    {
                        "mode": "Serve",
                        "requested_mode": "Serve",
                        "recharge_matched": False,
                        "target_pos": target_position,
                    }
                )
            else:
                action_n.append(
                    {
                        "mode": "Wait",
                        "requested_mode": "Wait",
                        "recharge_matched": False,
                        "target_pos": list(agent.pos),
                    }
                )

        # 每个 agent 必须有且仅有一个同位置索引的动作。
        if len(action_n) != len(acting_agents):
            raise RuntimeError("action_n 与 world.agents 的数量不一致")

        # 仅保存动作副本用于下一行 history CSV，不在此修改任何实体属性。
        latest_action_by_agent = {}
        for index, agent in enumerate(acting_agents):
            latest_action_by_agent[agent] = dict(action_n[index])

        # 6. 将 action_n 交给环境。env.step() 内部调用 World.update()，
        # 位置、电量、轨迹游标和状态变化全部由 World.update() 执行。
        new_obs_n, _old_obs_n, reward_n, _done_n = env.step(action_n)

        latest_reward_by_agent = {}
        for index, agent in enumerate(acting_agents):
            if index < len(reward_n):
                latest_reward_by_agent[agent] = reward_n[index]

        obs_n = new_obs_n

        # 7. 记录全局逐 step 统计数据。
        info = env.world.build_info()
        available_fcs_count = 0
        busy_fcs_count = 0

        for fcs in env.world.FCSs:
            if fcs.has_available_slot():
                available_fcs_count += 1
            if fcs.is_busy:
                busy_fcs_count += 1

        statistics_rows.append(
            {
                "step": info["step"],
                "num_iev": info["num_iev"],
                "num_quasi": info["num_quasi"],
                "num_charging": info["num_charging"],
                "num_success": info["num_success"],
                "num_fail": info["num_fail"],
                "num_idle_mcs": info["num_idle_mcs"],
                "num_task_mcs": info["num_task_mcs"],
                "num_broken_mcs": info["num_broken"],
                "num_avail_fcs": available_fcs_count,
                "num_busy_fcs": busy_fcs_count,
            }
        )

    # 8. 覆盖保存全局统计 CSV。
    statistics_table = pd.DataFrame(statistics_rows)
    statistics_path = output_dir / "simulation_statistics.csv"
    statistics_table.to_csv(statistics_path, index=False)

    # 9. 为每辆 EV/MCS 分别覆盖保存时序 CSV。
    for ev_id, rows in ev_history.items():
        ev_path = output_dir / f"ev_{ev_id}_history.csv"
        pd.DataFrame(rows).to_csv(ev_path, index=False)

    for mcs_id, rows in mcs_history.items():
        mcs_path = output_dir / f"mcs_{mcs_id}_history.csv"
        pd.DataFrame(rows).to_csv(mcs_path, index=False)

    # 10. 覆盖保存动态调度 GIF。
    tracked_ev_ids = []
    for ev in env.world.EVs[: args.tracked_count]:
        tracked_ev_ids.append(ev.id)

    tracked_mcs_ids = []
    for mcs in env.world.MCSs[: args.tracked_count]:
        tracked_mcs_ids.append(mcs.id)

    gif_path = output_dir / "simulation.gif"
    save_animation(
        frames,
        gif_path,
        tracked_ev_ids,
        tracked_mcs_ids,
    )

    print(f"统计汇总已保存: {statistics_path}")
    print(f"实体时序 CSV 已保存至: {output_dir}")
    print(f"动态调度图已保存: {gif_path}")
