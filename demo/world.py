"""
demo/world.py — 仿真世界实现 (Target-based Replanning 架构)

仿照 env/world.py 的 World 类设计, 提供 5 个核心函数:
  update(action_n)          — 执行动作、移动智能体、更新环境
  step_finish()             — old_agents = agents, 清理逐步状态
  match_and_get_neibor()    — 即时匹配 + 构建邻居列表 + 产生新 agents
  get_obs_n()               — 构建 new/old agents 的观测
  mix_get_reward_n()        — 为 last_agents 计算奖励

供 MultiAgentEnv 调用。
"""

import random
from typing import Dict, List

import numpy as np
import pandas as pd

from config import *
from core import EV, MCS, FCS, euclidean_distance
from matching import ImmediateMatcher
from observation import ObservationBuilder
from reward import (
    FAILURE_RESPONSIBILITY_DECAY,
    FAILURE_RESPONSIBILITY_MAX_AGE_STEPS,
    MCS_SUCCESS_BASE_CREDIT,
    MCS_SUCCESS_RESCUE_CREDIT,
    RESCUE_SUCCESS_THRESHOLD,
    RewardBuilder,
)


# ============================================================
# World 类
# ============================================================

class World:
    """仿真世界 — EV-FCS-MCS 协同充电调度"""

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed_val = seed
        self.verbose = verbose
        random.seed(seed)
        np.random.seed(seed)

        self.EVs: List[EV] = []
        self.MCSs: List[MCS] = []
        self.FCSs: List[FCS] = []

        self.agents: List = []  # 当前 step 的决策智能体 (waiting IEV + idle MCS)
        self.last_agents: List = []  # 上一 step 的智能体 (用于奖励分配)

        self.obs_builder = ObservationBuilder()
        self.reward_builder = RewardBuilder()
        self.immediate_matcher = ImmediateMatcher()  # IEV 与 idle MCS / Avail FCS 的即时匹配

        self.mcs_positions = [np.random.uniform(AREA_LON_MIN, AREA_LON_MAX, NUM_MCS),
                              np.random.uniform(AREA_LAT_MIN, AREA_LAT_MAX, NUM_MCS)]
        self.fcs_positions = [np.random.uniform(AREA_LON_MIN, AREA_LON_MAX, NUM_FCS),
                              np.random.uniform(AREA_LAT_MIN, AREA_LAT_MAX, NUM_FCS)]

        self.current_step = 0
        self.mcs_step_events: Dict[int, Dict] = {}
        self.last_mcs_reward_components: Dict[int, Dict[str, float]] = {}
        self.last_high_reward_by_option: Dict[int, float] = {}
        self.last_immediate_results: List[Dict] = []
        # 保存当前step动作执行前的EV结果状态，用于识别系统级新增成功/失败。
        self.ev_outcomes_before_step: Dict[int, tuple[bool, bool]] = {}
        self.last_system_reward_event: Dict[str, float] = {}
        # Low 奖励按唯一 Serve 决策 ID 路由，不再按当前 mcs_id 猜测。
        self.last_low_reward_by_decision: Dict[int, float] = {}
        self.last_low_event_records: List[Dict] = []
        # 新失败在下一轮匹配前计算责任，避免用匹配后的资源状态回溯责任。
        self.pending_failure_responsibilities: Dict[int, Dict[int, float]] = {}
        self.pending_failure_low_decisions: Dict[int, Dict[int, int]] = {}
        self.pending_failure_causal_records: Dict[int, Dict[int, Dict]] = {}
        # 在有限回看窗口内保存最近一次真实可行的服务机会。IEV 在
        # MCS 随后转去 Recharge/其他任务后失败时，仍可沿原 Low 决策回传。
        self.failure_responsibility_history: Dict[int, Dict] = {}
        self.init_world()

    def _would_mcs_be_energy_stranded(self, mcs: MCS) -> bool:
        """判断可用MCS当前是否因电量不足而无法到达任何物理FCS。

        这里只检查物理能量可达性，不把 FCS 当前无空闲槽、匹配竞争等
        外生因素算作受困。任务中、补电中和 broken 的 MCS 也不重复记账。
        """
        if (
            not self.FCSs
            or not mcs.is_idle
            or mcs.is_recharging
            or mcs.is_broken
        ):
            return False
        nearest_required_energy = min(
            euclidean_distance(
                mcs.pos[0], mcs.pos[1], fcs.pos[0], fcs.pos[1]
            ) / 1000.0 * POWER_UNIT
            for fcs in self.FCSs
        )
        return bool(not float(mcs.remain) > float(nearest_required_energy))

    def _is_mcs_energy_stranded(self, mcs: MCS) -> bool:
        """返回持久受困状态，兼容尚未固化状态的边界检查。"""
        return bool(
            getattr(mcs, 'is_energy_stranded', False)
            or self._would_mcs_be_energy_stranded(mcs)
        )

    # ============================================================
    # 初始化
    # ============================================================

    def init_world(self):
        """初始化所有实体"""
        self.EVs.clear()
        self.MCSs.clear()
        self.FCSs.clear()

        # MCS
        for i in range(NUM_MCS):
            pos = [float(self.mcs_positions[0][i]), float(self.mcs_positions[1][i])]
            mcs = MCS(mcs_id=i + 1, pos=pos, remain_kwh=MCS_BATTERY_CAPACITY)
            self.MCSs.append(mcs)

        # FCS
        for i in range(NUM_FCS):
            pos = [float(self.fcs_positions[0][i]), float(self.fcs_positions[1][i])]
            fcs = FCS(fcs_id=i + 1, pos=pos, num_slots=FCS_SLOTS_PER_STATION)
            self.FCSs.append(fcs)

        # EV
        ev_track_data = pd.read_csv(TRACK_DATA_PATH)
        shuffled_indices = ev_track_data.index.tolist()
        random.shuffle(shuffled_indices)
        counter = 1
        for index in shuffled_indices:
            if counter > NUM_EV:
                break
            row = ev_track_data.loc[index]
            remain = np.random.normal(loc=EV_POWER_MEAN, scale=EV_POWER_STD)
            if remain - row['distance'] * POWER_UNIT >= 0:
                continue
            ev = EV(counter, [float(row['lng']), float(row['lat'])], remain, row['distance'])
            tracks = row['track'].split(',')
            # 插值修复轨迹
            for i in range(len(tracks)):
                if i < len(tracks) - 1:
                    cur_track, nxt_track = tracks[i], tracks[i + 1]
                    x1, y1 = cur_track.split(' ')
                    x2, y2 = nxt_track.split(' ')
                    x1, y1 = float(x1), float(y1)
                    x2, y2 = float(x2), float(y2)
                    ev.track.append([float(x1), float(y1)])  # 加入tracks[i]
                    interval_km = euclidean_distance(x1, y1, x2, y2) / 1000.0
                    if interval_km > COMM_RANGE:
                        itv_left = interval_km
                        while itv_left > COMM_RANGE:
                            ratio = (COMM_RANGE - 0.001) / itv_left  # 由于使用的球面距离计算，故而增加 1米 误差
                            x_ = x1 + ratio * (x2 - x1)
                            y_ = y1 + ratio * (y2 - y1)
                            ev.track.append([float(x_), float(y_)])  # 插入新点
                            x1, y1 = x_, y_
                            itv_left -= (COMM_RANGE - 0.001)
                else:
                    x1, y1 = tracks[i].split(' ')
                    ev.track.append([float(x1), float(y1)])  # 加入末尾节点

            ev.track_index = 0
            ev.destination = ev.track[-1]
            ev.set_charge()
            self.EVs.append(ev)
            counter += 1

    def reset_world(self):
        """重置世界状态 — 保留实体当前位置/电量，仅清除运行时状态"""
        self.agents.clear()
        self.last_agents.clear()
        self.current_step = 0
        self.mcs_step_events.clear()
        self.last_mcs_reward_components.clear()
        self.last_high_reward_by_option.clear()
        self.last_immediate_results.clear()
        self.ev_outcomes_before_step.clear()
        self.last_system_reward_event.clear()
        self.last_low_reward_by_decision.clear()
        self.last_low_event_records.clear()
        self.pending_failure_responsibilities.clear()
        self.pending_failure_low_decisions.clear()
        self.pending_failure_causal_records.clear()
        self.failure_responsibility_history.clear()

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)

        self.mcs_positions = [np.random.uniform(AREA_LON_MIN, AREA_LON_MAX, NUM_MCS),
                              np.random.uniform(AREA_LAT_MIN, AREA_LAT_MAX, NUM_MCS)]
        self.fcs_positions = [np.random.uniform(AREA_LON_MIN, AREA_LON_MAX, NUM_FCS),
                              np.random.uniform(AREA_LAT_MIN, AREA_LAT_MAX, NUM_FCS)]

        # reset mcs
        for index, mcs in enumerate(self.MCSs):
            init_pos = [float(self.mcs_positions[0][index]), float(self.mcs_positions[1][index])]
            mcs.reset(init_pos)

        # reset fcs
        for fcs in self.FCSs:
            fcs.reset()

        # reset ev
        ev_track_data = pd.read_csv(TRACK_DATA_PATH)
        shuffled_indices = ev_track_data.index.tolist()
        random.shuffle(shuffled_indices)
        for idx, ev in enumerate(self.EVs):
            index = shuffled_indices[idx]
            row = ev_track_data.loc[index]
            remain = np.random.normal(loc=EV_POWER_MEAN, scale=EV_POWER_STD)
            ev.reset([float(row['lng']), float(row['lat'])], remain)

    # ============================================================
    # ① update(action_n) — 执行动作, 移动智能体, 更新环境
    # ============================================================

    def update(self, action_n: List[dict]):
        """Execute actions and record one auditable event for every MCS."""
        # reset阶段的自动匹配发生在任何策略动作之前。每个正式step在
        # 动作执行前记录结果状态，随后只奖励本step新发生的成功/失败，
        # 从而避免初始匹配或历史结果被重复计奖。
        self.ev_outcomes_before_step = {
            ev.id: (bool(ev.is_charged), bool(ev.fail_charge))
            for ev in self.EVs
        }

        # FCS 只有在所有充电槽均空闲时，is_idle 才为 True。
        # 按完整 step 统计该状态的持续时间，单位为分钟。
        for fcs in self.FCSs:
            if fcs.is_idle:
                fcs.total_idle_time_min += STEP_DURATION_MIN

        previous_mcs_state = {}
        for mcs in self.MCSs:
            spatial = self.reward_builder.compute_mcs_spatial_features(
                mcs, self.EVs, self.MCSs, self.FCSs
            )
            low_spatial = self.reward_builder.compute_low_spatial_features(
                mcs, self.EVs, self.FCSs
            )
            best_serve_potential = (
                self.reward_builder.compute_best_available_serve_potential(
                    mcs, self.MCSs, self.FCSs
                )
            )
            previous_mcs_state[mcs.id] = {
                'remain': float(mcs.remain),
                'spatial_attraction': spatial['attraction'],
                'spatial_competition': spatial['competition'],
                'spatial_potential': spatial['potential'],
                'low_spatial_attraction': low_spatial['attraction'],
                'low_spatial_mcs_competition': low_spatial[
                    'mcs_competition'
                ],
                'low_spatial_fcs_competition': low_spatial[
                    'fcs_competition'
                ],
                'low_spatial_desirability': low_spatial['desirability'],
                'best_serve_potential': best_serve_potential,
                'pos': list(mcs.pos),
                'is_broken': bool(mcs.is_broken),
                'is_idle': bool(mcs.is_idle),
                'is_task': bool(mcs.is_task),
                'is_recharging': bool(mcs.is_recharging),
                'is_energy_stranded': bool(
                    getattr(mcs, 'is_energy_stranded', False)
                ),
                'total_profit': float(mcs.total_profit),
                'total_cost': float(mcs.total_cost),
            }

        # 在策略动作改变资源状态前冻结真实可行服务机会。具体 High/Low ID
        # 必须在解析本 step 动作后再写入，避免把新 Option 的机会错挂到上一个
        # 已关闭 Option。
        pre_action_failure_opportunities: Dict[int, Dict[int, float]] = {}
        for ev in self.EVs:
            if ev.is_charged or ev.fail_charge or ev.is_normal:
                continue
            weights = self.reward_builder.compute_failure_responsibility_weights(
                ev, self.MCSs
            )
            if not weights:
                continue
            pre_action_failure_opportunities[int(ev.id)] = dict(weights)

        action_by_mcs_id = {}
        for index, agent in enumerate(self.agents):
            if index >= len(action_n):
                continue
            action = action_n[index]
            agent.last_pos = list(agent.pos)
            target_pos = action.get('target_pos')
            if target_pos is None:
                continue

            dist_km = euclidean_distance(
                agent.pos[0], agent.pos[1], target_pos[0], target_pos[1]
            ) / 1000.0
            if isinstance(agent, EV):
                is_next_track_move = False
                if agent.track_index < len(agent.track) - 1:
                    next_track_pos = agent.track[agent.track_index + 1]
                    is_next_track_move = bool(np.allclose(
                        target_pos,
                        next_track_pos,
                        rtol=0.0,
                        atol=1e-10,
                    ))
                    if is_next_track_move:
                        agent.track_index += 1
                agent.pos = list(target_pos)
                agent.remain -= dist_km * POWER_UNIT
                agent.set_charge()                                  # 更新EV状态
                if not is_next_track_move:
                    agent.total_extra_dist_km += dist_km
                continue

            if not isinstance(agent, MCS):
                continue

            mode = action.get('mode', 'Wait')
            high_action_mask = np.asarray(
                action.get('high_action_mask', []), dtype=bool
            ).reshape(-1)
            if high_action_mask.size < 2:
                # 兼容旧调用方：按动作执行前的局部状态重建有效动作掩码。
                physical_fcss = (
                    self.obs_builder.get_physically_reachable_fcss(
                        agent, self.FCSs
                    )
                )
                recharge_available = bool(physical_fcss)
                nearest_energy = (
                    float(physical_fcss[0][2])
                    if physical_fcss else float('inf')
                )
                serve_available = bool(
                    float(agent.remain)
                    > float(MCS_SERVE_SAFETY_RESERVE_KWH) + nearest_energy
                )
                high_action_mask = np.asarray(
                    [serve_available, recharge_available], dtype=bool
                )
            else:
                # 旧三动作调用方只读取前两个动作。不得在这里
                # 强制打开 Serve，否则会绕过 recharge-only 安全域。
                high_action_mask = high_action_mask[:2].copy()
            action_by_mcs_id[agent.id] = {
                'mode': mode,
                'requested_mode': action.get('requested_mode', mode),
                'recharge_matched': bool(action.get('recharge_matched', False)),
                'high_action_mask': high_action_mask,
                'low_decision_id': int(action.get('low_decision_id', -1)),
                'serve_option_id': int(action.get('serve_option_id', -1)),
                'high_option_id': int(action.get(
                    'high_option_id', action.get('serve_option_id', -1)
                )),
                'low_candidate_id': int(action.get('low_candidate_id', -1)),
                'low_stay_selected': bool(
                    action.get('low_stay_selected', False)
                ),
                'low_forced_stay': bool(action.get('low_forced_stay', False)),
                'has_quasi_candidate': bool(
                    action.get('has_quasi_candidate', False)
                ),
                'low_candidate_priority_reward': float(
                    action.get('low_candidate_priority_reward', 0.0)
                ),
            }
            high_option_id = int(
                action_by_mcs_id[agent.id]['high_option_id']
            )
            if high_option_id >= 0:
                agent.active_high_option_id = high_option_id
                agent.active_high_mode = str(
                    action_by_mcs_id[agent.id]['requested_mode']
                )
            if action_by_mcs_id[agent.id]['requested_mode'] == 'Serve':
                low_decision_id = action_by_mcs_id[agent.id][
                    'low_decision_id'
                ]
                serve_option_id = action_by_mcs_id[agent.id][
                    'serve_option_id'
                ]
                previous_serve_option_id = int(
                    getattr(agent, 'active_serve_option_id', -1)
                )
                # 训练使用稳定的非负 option_id；旧评测脚本使用 -1，
                # 但每次显式 Serve 动作同样代表一个新的可匹配窗口。
                if (
                    serve_option_id < 0
                    or serve_option_id != previous_serve_option_id
                ):
                    agent.active_serve_has_matched = False
                # 一个 High Serve option 可包含多个 Low 局部子决策，
                # 因此两个 ID 必须分离：serve_option_id 在整个 High
                # option 内稳定，low_decision_id 在每次重规划时更新。
                # 训练流程始终提供 ID；旧 test/test_actor 推理脚本没有
                # rollout，因此允许 -1 并仅跳过 Low 学习归因。
                agent.active_low_decision_id = low_decision_id
                agent.active_serve_option_id = serve_option_id
                agent.active_low_candidate_id = action_by_mcs_id[agent.id][
                    'low_candidate_id'
                ]
                agent.active_low_started_step = self.current_step
            else:
                # 下一次明确 High=Recharge 时，旧空间决策不再生效。
                agent.active_low_decision_id = -1
                agent.active_serve_option_id = -1
                agent.active_low_candidate_id = -1
                agent.active_low_started_step = -1
                agent.active_serve_has_matched = False
            if mode == 'Recharge':
                continue
            if mode == 'Wait':
                continue
            if mode != 'Serve':
                raise ValueError(f'Unsupported MCS action mode: {mode}')

            required_energy = dist_km * POWER_UNIT
            available_energy = max(float(agent.remain), 0.0)
            if required_energy <= 1e-12:
                agent.pos = list(target_pos)
            elif available_energy <= required_energy:
                ratio = available_energy / required_energy
                agent.pos[0] += ratio * (target_pos[0] - agent.pos[0])
                agent.pos[1] += ratio * (target_pos[1] - agent.pos[1])
                agent.remain = 0.0
                agent.is_broken = True
                agent.is_idle = False
                agent.total_energy_consumed += available_energy
                agent.total_cost += available_energy * RC_PRICE
            else:
                agent.pos = list(target_pos)
                agent.remain -= required_energy
                agent.total_energy_consumed += required_energy
                agent.total_cost += required_energy * RC_PRICE

        mcs_by_id = {int(mcs.id): mcs for mcs in self.MCSs}
        for ev_id, weights in pre_action_failure_opportunities.items():
            causal_records: Dict[int, Dict] = {}
            low_decisions: Dict[int, int] = {}
            high_options: Dict[int, int] = {}
            for raw_mcs_id, raw_weight in weights.items():
                mcs_id = int(raw_mcs_id)
                mcs = mcs_by_id.get(mcs_id)
                action = action_by_mcs_id.get(mcs_id, {})
                requested_mode = str(action.get(
                    'requested_mode', getattr(mcs, 'active_high_mode', '')
                ))
                high_option_id = int(action.get(
                    'high_option_id',
                    getattr(mcs, 'active_high_option_id', -1),
                ))
                low_decision_id = int(action.get(
                    'low_decision_id',
                    getattr(mcs, 'active_low_decision_id', -1),
                ))
                if requested_mode == 'Recharge':
                    cause_type = 'high_recharge_deferral'
                    high_share, low_share = 1.0, 0.0
                elif requested_mode == 'Serve' and low_decision_id >= 0:
                    cause_type = (
                        'low_wait_deferral'
                        if bool(action.get('low_stay_selected', False))
                        else 'low_reposition_deferral'
                    )
                    high_share, low_share = 0.0, 1.0
                elif high_option_id >= 0:
                    cause_type = 'high_unroutable_serve_deferral'
                    high_share, low_share = 1.0, 0.0
                else:
                    cause_type = 'orphan_opportunity'
                    high_share, low_share = 0.0, 0.0
                low_decisions[mcs_id] = low_decision_id
                high_options[mcs_id] = high_option_id
                causal_records[mcs_id] = {
                    'responsibility_weight': float(raw_weight),
                    'high_option_id': high_option_id,
                    'low_decision_id': low_decision_id,
                    'requested_mode': requested_mode,
                    'cause_type': cause_type,
                    'high_share': float(high_share),
                    'low_share': float(low_share),
                }
            self.failure_responsibility_history[ev_id] = {
                'step': int(self.current_step),
                'weights': dict(weights),
                'low_decisions': low_decisions,
                'high_options': high_options,
                'causal_records': causal_records,
            }

        for mcs in self.MCSs:
            if (
                not mcs.is_idle
                and not mcs.is_recharging
                and not mcs.is_broken
                and not mcs.is_energy_stranded
                and mcs.current_target is not None
            ):
                mcs.advance_charging()

        for fcs in self.FCSs:
            if not fcs.is_idle:
                fcs.advance_charging()

        for ev in self.EVs:
            if ev.is_charged or ev.fail_charge or ev.is_normal:
                continue
            if (ev.is_charged and ev.charge_pos is None) or ev.is_quasi:
                ev.last_pos = list(ev.pos)
                if ev.track_index < len(ev.track) - 1:
                    ev.track_index += 1
                    ev.pos = list(ev.track[ev.track_index])
                    dist_km = euclidean_distance(
                        ev.last_pos[0], ev.last_pos[1], ev.pos[0], ev.pos[1]
                    ) / 1000.0
                    ev.remain -= dist_km * POWER_UNIT
                    ev.set_charge()
                else:
                    ev.arrived = True
                    ev.need_charge = False

        events = {}
        for mcs in self.MCSs:
            previous = previous_mcs_state[mcs.id]
            if (
                previous['is_idle']
                and not previous['is_broken']
                and not previous['is_recharging']
            ):
                mcs.total_idle_time_min += STEP_DURATION_MIN
            involved_in_task = previous['is_task'] or mcs.is_task
            involved_in_recharge = (
                previous['is_recharging'] or mcs.is_recharging
            )
            default_mode = 'TaskProgress' if involved_in_task else (
                'RechargeProgress' if involved_in_recharge else 'Inactive'
            )
            action = action_by_mcs_id.get(mcs.id, {})
            requested_mode = action.get('requested_mode', default_mode)
            high_action_mask = np.asarray(
                action.get('high_action_mask', [False, False]),
                dtype=bool,
            ).reshape(-1)
            serve_available = bool(
                high_action_mask[0] if high_action_mask.size > 0 else False
            )
            recharge_available = bool(
                high_action_mask[1] if high_action_mask.size > 1 else False
            )
            low_stay_selected = bool(
                action.get('low_stay_selected', False)
            )
            low_forced_stay = bool(action.get('low_forced_stay', False))
            # High 已删除 Wait。Low 选择固定“当前位置”候选承载等待语义：
            # 有其他 quasi 时是主动等待；只有当前位置时是被动等待。
            # requested_mode == Wait 仅保留给旧调用方兼容。
            legacy_selected_wait = requested_mode == 'Wait'
            forced_wait = bool(
                (low_stay_selected and low_forced_stay)
                or (
                    legacy_selected_wait
                    and not serve_available
                    and not recharge_available
                )
            )
            voluntary_wait = bool(
                (low_stay_selected and not low_forced_stay)
                or (
                    legacy_selected_wait
                    and (serve_available or recharge_available)
                )
            )
            if voluntary_wait:
                mcs.consecutive_voluntary_wait_steps += 1
            else:
                # 移动 Serve、Recharge、被动等待或任务推进会中断主动等待串。
                mcs.consecutive_voluntary_wait_steps = 0

            moved_distance_km = euclidean_distance(
                previous['pos'][0], previous['pos'][1], mcs.pos[0], mcs.pos[1]
            ) / 1000.0
            movement_energy = moved_distance_km * POWER_UNIT
            battery_delta = float(mcs.remain) - previous['remain']
            service_kwh = max(-battery_delta - movement_energy, 0.0) if involved_in_task else 0.0
            recharged_kwh = max(battery_delta + movement_energy, 0.0) if involved_in_recharge else 0.0
            newly_energy_stranded = bool(
                self._would_mcs_be_energy_stranded(mcs)
                and not previous['is_energy_stranded']
            )
            if newly_energy_stranded:
                # 与 broken 一样成为永久终止状态：不再参与匹配、邻居构建
                # 或后续策略决策，但保留独立状态和审计事件。
                mcs.is_energy_stranded = True
                mcs.is_idle = False
            events[mcs.id] = {
                'mode': action.get('mode', default_mode),
                'requested_mode': requested_mode,
                'recharge_matched': bool(action.get('recharge_matched', False)),
                'waited': action.get('mode') == 'Wait',
                'forced_wait': forced_wait,
                'voluntary_wait': voluntary_wait,
                'serve_available': serve_available,
                'recharge_available': recharge_available,
                'has_quasi_candidate': bool(
                    action.get('has_quasi_candidate', False)
                ),
                'low_candidate_priority_reward': float(
                    action.get('low_candidate_priority_reward', 0.0)
                ),
                'low_stay_selected': low_stay_selected,
                'low_forced_stay': low_forced_stay,
                'wait_duration_steps': (
                    1 if (low_stay_selected or legacy_selected_wait) else 0
                ),
                'consecutive_voluntary_wait_steps': int(
                    mcs.consecutive_voluntary_wait_steps
                ),
                'best_available_serve_potential': float(
                    previous['best_serve_potential']
                ),
                'movement_distance_km': float(moved_distance_km),
                'movement_energy_kwh': float(movement_energy),
                'service_kwh': float(service_kwh),
                'recharged_kwh': float(recharged_kwh),
                'battery_delta_kwh': float(battery_delta),
                'previous_remain_kwh': float(previous['remain']),
                'current_remain_kwh': float(mcs.remain),
                'profit_delta': float(
                    mcs.total_profit - previous['total_profit']
                ),
                'cost_delta': float(
                    mcs.total_cost - previous['total_cost']
                ),
                'previous_total_profit': float(previous['total_profit']),
                'previous_total_cost': float(previous['total_cost']),
                'previous_attraction': float(previous['spatial_attraction']),
                'previous_competition': float(previous['spatial_competition']),
                'previous_spatial_potential': float(previous['spatial_potential']),
                'previous_low_spatial_attraction': float(
                    previous['low_spatial_attraction']
                ),
                'previous_low_spatial_mcs_competition': float(
                    previous['low_spatial_mcs_competition']
                ),
                'previous_low_spatial_fcs_competition': float(
                    previous['low_spatial_fcs_competition']
                ),
                'previous_low_spatial_desirability': float(
                    previous['low_spatial_desirability']
                ),
                'newly_broken': bool(mcs.is_broken and not previous['is_broken']),
                'newly_energy_stranded': newly_energy_stranded,
                'became_idle': bool(mcs.is_idle and not previous['is_idle']),
                'became_task': bool(mcs.is_task and not previous['is_task']),
                'became_recharging': bool(mcs.is_recharging and not previous['is_recharging']),
                'low_decision_id': int(mcs.active_low_decision_id),
                'serve_option_id': int(mcs.active_serve_option_id),
                'high_option_id': int(mcs.active_high_option_id),
                'high_option_mode': str(mcs.active_high_mode),
            }
        self.mcs_step_events = events
        # 失败责任必须使用匹配前状态。匹配后 idle MCS 可能已接到其他任务，
        # 再回溯会把本应可控的失败错误归为外生失败。
        self.pending_failure_responsibilities = {}
        self.pending_failure_low_decisions = {}
        self.pending_failure_causal_records = {}
        for ev in self.EVs:
            previous_charged, previous_failed = self.ev_outcomes_before_step.get(
                ev.id, (bool(ev.is_charged), bool(ev.fail_charge))
            )
            if ev.fail_charge and not previous_failed and not previous_charged:
                history = self.failure_responsibility_history.get(ev.id)
                history_age = (
                    int(self.current_step) - int(history['step'])
                    if history is not None else 10 ** 9
                )
                if (
                    history is not None
                    and history_age <= FAILURE_RESPONSIBILITY_MAX_AGE_STEPS
                ):
                    weights = dict(history['weights'])
                    low_decisions = dict(history['low_decisions'])
                    causal_records = {
                        int(mcs_id): dict(record)
                        for mcs_id, record in history.get(
                            'causal_records', {}
                        ).items()
                    }
                    for record in causal_records.values():
                        record['attribution_age_steps'] = int(history_age)
                else:
                    weights = self.reward_builder.compute_failure_responsibility_weights(
                        ev, self.MCSs
                    )
                    low_decisions = {}
                    causal_records = {}
                    for mcs_id, weight in weights.items():
                        mcs = mcs_by_id.get(int(mcs_id))
                        low_decision_id = int(getattr(
                            mcs, 'active_low_decision_id', -1
                        ))
                        high_option_id = int(getattr(
                            mcs, 'active_high_option_id', -1
                        ))
                        low_decisions[int(mcs_id)] = low_decision_id
                        if low_decision_id >= 0:
                            high_share, low_share = 0.0, 1.0
                            cause_type = 'low_current_deferral'
                        elif high_option_id >= 0:
                            high_share, low_share = 1.0, 0.0
                            cause_type = 'high_current_deferral'
                        else:
                            high_share, low_share = 0.0, 0.0
                            cause_type = 'orphan_current_opportunity'
                        causal_records[int(mcs_id)] = {
                            'responsibility_weight': float(weight),
                            'high_option_id': high_option_id,
                            'low_decision_id': low_decision_id,
                            'requested_mode': str(getattr(
                                mcs, 'active_high_mode', ''
                            )),
                            'cause_type': cause_type,
                            'high_share': high_share,
                            'low_share': low_share,
                            'attribution_age_steps': 0,
                        }
                self.pending_failure_responsibilities[ev.id] = weights
                self.pending_failure_low_decisions[ev.id] = low_decisions
                self.pending_failure_causal_records[ev.id] = causal_records
        self.current_step += 1

    # ============================================================
    # step_finish
    # ============================================================

    def step_finish(self):
        """last_agents = agents; 清空 agents; 清空每步邻居列表"""
        self.last_agents = list(self.agents)
        self.agents.clear()

        for ev in self.EVs:
            ev.step_finish()
        for mcs in self.MCSs:
            mcs.step_finish()

    # ============================================================
    # ③ match_and_get_neibor() — 匹配 + 构建邻居 + 产生新 agents
    # ============================================================

    def match_and_get_neibor(self):
        # ── 即时匹配: IEV ↔ Idle MCS / Available FCS ──
        results = self.immediate_matcher.match_all(self.EVs, self.MCSs, self.FCSs)
        self.last_immediate_results = list(results)
        ev_by_id = {ev.id: ev for ev in self.EVs}
        mcs_by_id = {mcs.id: mcs for mcs in self.MCSs}
        for result in results:
            if not result.get('success'):
                continue
            ev = ev_by_id.get(result.get('ev_id'))
            if ev is not None:
                feasible_mcs_count = max(
                    int(result.get('feasible_mcs_count', 0)), 0
                )
                feasible_fcs_slots = max(
                    int(result.get('feasible_fcs_slot_count', 0)), 0
                )
                ev.feasible_mcs_count = feasible_mcs_count
                ev.feasible_fcs_slot_count = feasible_fcs_slots
                rescue_probability = float(np.clip(
                    result.get('counterfactual_failure_probability', 0.0),
                    0.0,
                    1.0,
                ))
                # High/Low 的业务成功信用均使用同一个反事实救援价值；
                # 层级差异由稳定 option/decision 路由负责，不再由两套替代
                # 资源计数代理制造目标偏差。
                ev.service_marginal_weight = rescue_probability
                ev.low_service_marginal_weight = rescue_probability
                ev.service_counterfactual_failure_probability = (
                    rescue_probability
                )
                ev.service_success_credit = (
                    self.reward_builder.compute_success_credit(
                        rescue_probability
                    )
                )
                ev.service_rescue_diagnostics = dict(
                    result.get('rescue_diagnostics', {})
                )
            if result.get('provider_type') != 'MCS':
                if ev is not None:
                    ev.service_low_decision_id = -1
                    ev.service_serve_option_id = -1
                continue
            provider = mcs_by_id.get(result.get('provider_id'))
            low_decision_id = int(
                getattr(provider, 'active_low_decision_id', -1)
            )
            serve_option_id = int(
                getattr(provider, 'active_serve_option_id', -1)
            )
            if ev is not None:
                ev.service_low_decision_id = low_decision_id
                ev.service_serve_option_id = serve_option_id
            provider.active_serve_has_matched = True
            event = self.mcs_step_events.get(result.get('provider_id'))
            if event is not None:
                event['became_task'] = True
                event['matched_iev_id'] = result.get('ev_id', -1)
                event['low_decision_id'] = low_decision_id
                event['serve_option_id'] = serve_option_id
        if self.verbose:
            print(f'step{self.current_step}充电匹配阶段内：\n')
            for rst in results:
                if rst['success']:
                    print(
                        f"{rst['ev_id']}号IEV被分配至{rst['provider_id']}号{rst['provider_type']}, 充电量{rst['charge_power']}Kwh，时长{rst['charge_time']}min。")
                else:
                    print(f"{rst['ev_id']}号IEV暂未匹配成功，加入Agent队列等待调度")
            print('\n')

        # 匹配之后的iev, quasi列表
        iev_list = [ev for ev in self.EVs if ev.is_iev]
        quasi_list = [ev for ev in self.EVs if ev.is_quasi]
        ev_list = iev_list + quasi_list

        iev_id_list = [ev.id for ev in iev_list]
        if self.verbose:
            print(f"匹配之后的iev列表{iev_id_list}, 用以检查")

        # EV-MCS + EV-FCS 邻居
        for ev in ev_list:
            # EV-MCS 邻居
            for mcs in self.MCSs:
                dist_km = euclidean_distance(mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1]) / 1000.0
                if dist_km > COMM_RANGE:
                    continue
                if (
                    not mcs.is_broken
                    and not mcs.is_energy_stranded
                    and not mcs.is_recharging
                    and mcs.is_idle
                ):
                    # idle MCS
                    ev.near_idle_mcs.append(mcs)
                    if ev.is_quasi:
                        mcs.near_quasi.append(ev)
                    if ev.is_iev:
                        mcs.near_iev.append(ev)
                elif (
                    not mcs.is_broken
                    and not mcs.is_energy_stranded
                    and not mcs.is_recharging
                    and not mcs.is_idle
                ):
                    # task MCS
                    ev.near_task_mcs.append(mcs)
                    if ev.is_iev:
                        mcs.near_iev.append(ev)
            # EV-FCS 邻居
            for fcs in self.FCSs:
                dist_km = euclidean_distance(fcs.pos[0], fcs.pos[1], ev.pos[0], ev.pos[1]) / 1000.0
                if dist_km > COMM_RANGE:
                    continue
                if fcs.has_available_slot():
                    ev.near_available_fcs.append(fcs)
                elif fcs.is_busy:
                    ev.near_busy_fcs.append(fcs)
                if ev.is_quasi:
                    fcs.near_quasi.append(ev)
                if ev.is_iev:
                    fcs.near_iev.append(ev)

        # IEV-IEV or IEV-quasi 邻居
        for i in range(len(iev_list)):
            for j in range(len(ev_list)):
                if iev_list[i] is ev_list[j]:
                    continue
                dist_km = euclidean_distance(
                    iev_list[i].pos[0], iev_list[i].pos[1],
                    ev_list[j].pos[0], ev_list[j].pos[1]
                ) / 1000.0
                if dist_km > COMM_RANGE:
                    continue
                if ev_list[j].is_iev:
                    iev_list[i].near_iev.append(ev_list[j])
                    ev_list[j].near_iev.append(iev_list[i])
                if ev_list[j].is_quasi:
                    iev_list[i].near_quasi.append(ev_list[j])
                    ev_list[j].near_iev.append(iev_list[i])

        # quasi-quasi 邻居
        for i in range(len(quasi_list)):
            for j in range(i + 1, len(quasi_list)):
                dist_km = euclidean_distance(
                    quasi_list[i].pos[0], quasi_list[i].pos[1],
                    quasi_list[j].pos[0], quasi_list[j].pos[1]) / 1000.0
                if dist_km > COMM_RANGE:
                    continue
                else:
                    quasi_list[i].near_quasi.append(quasi_list[j])
                    quasi_list[j].near_quasi.append(quasi_list[i])
        # MCS-MCS 邻居
        for i in range(len(self.MCSs)):
            for j in range(i + 1, len(self.MCSs)):
                if (
                    self.MCSs[i].is_broken
                    or self.MCSs[i].is_energy_stranded
                    or self.MCSs[i].is_recharging
                ):
                    continue
                dist_km = euclidean_distance(
                    self.MCSs[i].pos[0], self.MCSs[i].pos[1],
                    self.MCSs[j].pos[0], self.MCSs[j].pos[1]) / 1000.0
                if dist_km > COMM_RANGE:
                    continue
                if (
                    not self.MCSs[j].is_broken
                    and not self.MCSs[j].is_energy_stranded
                    and not self.MCSs[j].is_recharging
                    and self.MCSs[j].is_idle
                ):
                    self.MCSs[i].near_idle_mcs.append(self.MCSs[j])
                elif (
                    not self.MCSs[j].is_broken
                    and not self.MCSs[j].is_energy_stranded
                    and not self.MCSs[j].is_recharging
                    and not self.MCSs[j].is_idle
                ):
                    self.MCSs[i].near_task_mcs.append(self.MCSs[j])
                if self.MCSs[i].is_idle and not self.MCSs[i].is_energy_stranded:
                    self.MCSs[j].near_idle_mcs.append(self.MCSs[i])
                elif (
                    not self.MCSs[i].is_idle
                    and not self.MCSs[i].is_energy_stranded
                ):
                    self.MCSs[j].near_task_mcs.append(self.MCSs[i])

        # MCS-FCS 邻居
        for i in range(len(self.MCSs)):
            for j in range(len(self.FCSs)):
                if (
                    self.MCSs[i].is_broken
                    or self.MCSs[i].is_energy_stranded
                    or self.MCSs[i].is_recharging
                ):
                    continue
                dist_km = euclidean_distance(
                    self.MCSs[i].pos[0], self.MCSs[i].pos[1],
                    self.FCSs[j].pos[0], self.FCSs[j].pos[1]) / 1000.0
                if dist_km > COMM_RANGE:
                    continue
                if self.MCSs[i].is_idle:
                    self.FCSs[j].near_idle_mcs.append(self.MCSs[i])
                else:
                    self.FCSs[j].near_task_mcs.append(self.MCSs[i])
                if self.FCSs[j].has_available_slot():
                    self.MCSs[i].near_available_fcs.append(self.FCSs[j])
                else:
                    self.MCSs[i].near_busy_fcs.append(self.FCSs[j])

        # ── 生成新一轮 agents ──
        for ev in iev_list:
            self.agents.append(ev)
        for mcs in self.MCSs:
            if (
                not mcs.is_broken
                and not mcs.is_energy_stranded
                and not mcs.is_recharging
                and mcs.is_idle
            ):
                self.agents.append(mcs)

    # ============================================================
    # ④ get_obs_n() — 为 new/old agents 构建观测
    # ============================================================

    def get_obs_n(self):
        """返回 (new_obs_n, old_obs_n, done_n)"""
        new_obs_n = []  # 新一轮agent在当前step的观测(obs_t)
        old_obs_n = []  # 上一轮agent在当前step的观测(obs_t+1)
        done_n = []  # 上一轮agent在当前step的完成标志

        for agent in self.agents:
            if isinstance(agent, MCS):
                new_obs_n.append(self.obs_builder.obs_mcs(agent, self.FCSs))
            else:
                new_obs_n.append(self.obs_builder.obs_iev(agent))

        for agent in self.last_agents:
            if agent in self.agents:
                # 仍在活跃列表中, 复用观测
                done_n.append(False)
                idx = self.agents.index(agent)
                old_obs_n.append(new_obs_n[idx])
            else:
                # 已不再活跃, 返回 done=True 的占位观测
                done_n.append(True)
                # 特征置空
                if isinstance(agent, MCS):
                    low_candidates = np.zeros(
                        (TOP_K_MCS_CANDIDATES, MCS_FEAT_DIM_tgt), dtype=np.float32
                    )
                    low_candidate_mask = np.zeros(TOP_K_MCS_CANDIDATES, dtype=bool)
                    low_self_state = np.zeros(MCS_FEAT_DIM_self, dtype=np.float32)
                    old_obs_n.append({
                        'high_state': np.zeros(MCS_HIGH_FEAT_DIM, dtype=np.float32),
                        'high_action_mask': np.asarray([False, False], dtype=bool),
                        'low_self_state': low_self_state,
                        'low_candidates': low_candidates,
                        'low_candidate_mask': low_candidate_mask,
                        'candidate_ids': np.full(TOP_K_MCS_CANDIDATES, -1, dtype=np.int64),
                        'candidate_is_stay': np.zeros(
                            TOP_K_MCS_CANDIDATES, dtype=bool
                        ),
                        'quasi_candidate_count': 0,
                        'stay_candidate_index': 0,
                        'obs_self': low_self_state,
                        'obs_tgt': low_candidates,
                        'mask': np.logical_not(low_candidate_mask),
                        'done': True,
                    })
                else:
                    old_obs_n.append({
                        'obs_self': np.zeros(EV_FEAT_DIM_self, dtype=np.float32),
                        'obs_tgt': np.zeros((TOP_K_MCS_CANDIDATES, EV_FEAT_DIM_tgt), dtype=np.float32),
                        'mask': np.zeros(TOP_K_MCS_CANDIDATES, dtype=bool),
                        'done': True,
                    })

        return new_obs_n, old_obs_n, done_n

    # ============================================================
    # ⑤ mix_get_reward_n() — 为 last_agents 计算奖励
    # ============================================================

    def mix_get_reward_n(self):
        """Return rewards aligned with last_agents and retain all MCS details."""
        new_success_evs = []
        new_failure_evs = []
        for ev in self.EVs:
            previous_charged, previous_failed = (
                self.ev_outcomes_before_step.get(
                    ev.id,
                    (bool(ev.is_charged), bool(ev.fail_charge)),
                )
            )
            if ev.is_charged and not previous_charged:
                new_success_evs.append(ev)
            if ev.fail_charge and not previous_failed:
                new_failure_evs.append(ev)

        mcs_success_count_by_id: Dict[int, int] = {
            mcs.id: 0 for mcs in self.MCSs
        }
        mcs_success_weight_by_id: Dict[int, float] = {
            mcs.id: 0.0 for mcs in self.MCSs
        }
        low_success_weight_by_decision: Dict[int, float] = {}
        high_success_weight_by_option: Dict[int, float] = {}
        low_failure_weight_by_decision: Dict[int, float] = {}
        self.last_low_event_records = []
        new_mcs_success_count = 0
        new_fcs_success_count = 0
        unattributed_mcs_success_count = 0
        rescue_success_count = 0
        replacement_success_count = 0
        immediate_counterfactual_replacement_count = 0
        future_rescue_evaluated_success_count = 0
        success_base_credit_sum = 0.0
        success_rescue_bonus_sum = 0.0
        success_credit_sum = 0.0
        counterfactual_failure_probability_sum = 0.0
        for ev in new_success_evs:
            if ev.charge_provider_type == 'MCS':
                new_mcs_success_count += 1
                if ev.charge_provider_id in mcs_success_count_by_id:
                    mcs_success_count_by_id[ev.charge_provider_id] += 1
                    marginal_weight = float(np.clip(
                        getattr(ev, 'service_marginal_weight', 1.0),
                        0.0,
                        1.0,
                    ))
                    mcs_success_weight_by_id[
                        ev.charge_provider_id
                    ] += marginal_weight
                    low_decision_id = int(getattr(
                        ev, 'service_low_decision_id', -1
                    ))
                    success_credit = self.reward_builder.compute_success_credit(
                        marginal_weight
                    )
                    success_base_credit_sum += MCS_SUCCESS_BASE_CREDIT
                    rescue_bonus = MCS_SUCCESS_RESCUE_CREDIT * marginal_weight
                    success_rescue_bonus_sum += rescue_bonus
                    success_credit_sum += success_credit
                    counterfactual_failure_probability_sum += marginal_weight
                    rescue_diagnostics = dict(getattr(
                        ev, 'service_rescue_diagnostics', {}
                    ))
                    if bool(rescue_diagnostics.get(
                        'immediate_counterfactual_replacement', 0.0
                    )):
                        immediate_counterfactual_replacement_count += 1
                    else:
                        future_rescue_evaluated_success_count += 1
                    if marginal_weight >= RESCUE_SUCCESS_THRESHOLD:
                        rescue_success_count += 1
                    else:
                        replacement_success_count += 1
                    serve_option_id = int(getattr(
                        ev, 'service_serve_option_id', -1
                    ))
                    if serve_option_id >= 0:
                        high_success_weight_by_option[serve_option_id] = (
                            high_success_weight_by_option.get(
                                serve_option_id, 0.0
                            ) + marginal_weight
                        )
                    if low_decision_id >= 0:
                        low_marginal_weight = float(np.clip(
                            getattr(
                                ev, 'low_service_marginal_weight',
                                marginal_weight,
                            ),
                            0.0,
                            1.0,
                        ))
                        low_success_weight_by_decision[low_decision_id] = (
                            low_success_weight_by_decision.get(
                                low_decision_id, 0.0
                            ) + low_marginal_weight
                        )
                    self.last_low_event_records.append({
                        'ev_id': int(ev.id),
                        'kind': 'mcs_success',
                        'low_decision_id': low_decision_id,
                        'serve_option_id': serve_option_id,
                        'responsibility_weight': (
                            low_marginal_weight
                            if low_decision_id >= 0 else 0.0
                        ),
                        'high_responsibility_weight': marginal_weight,
                        'counterfactual_failure_probability': marginal_weight,
                        'success_base_credit': MCS_SUCCESS_BASE_CREDIT,
                        'success_rescue_bonus': rescue_bonus,
                        'success_credit': success_credit,
                        'immediate_counterfactual_replacement': float(
                            rescue_diagnostics.get(
                                'immediate_counterfactual_replacement', 0.0
                            )
                        ),
                        'feasible_mcs_count': int(getattr(
                            ev, 'feasible_mcs_count', 0
                        )),
                        'feasible_fcs_slot_count': int(getattr(
                            ev, 'feasible_fcs_slot_count', 0
                        )),
                    })
                else:
                    unattributed_mcs_success_count += 1
            elif ev.charge_provider_type == 'FCS':
                new_fcs_success_count += 1
                self.last_low_event_records.append({
                    'ev_id': int(ev.id),
                    'kind': 'fcs_success_kpi',
                    'low_decision_id': -1,
                    'responsibility_weight': 0.0,
                })

        failure_weight_by_mcs_id: Dict[int, float] = {
            mcs.id: 0.0 for mcs in self.MCSs
        }
        high_failure_weight_by_option: Dict[int, float] = {}
        controllable_failure_count = 0
        uncontrollable_failure_count = 0
        historically_attributed_failure_count = 0
        high_failure_route_count = 0
        low_failure_route_count = 0
        orphan_failure_route_count = 0
        high_failure_weight_sum = 0.0
        low_failure_weight_sum = 0.0
        orphan_failure_weight_sum = 0.0
        failure_attribution_age_weighted_sum = 0.0
        failure_cause_counts: Dict[str, int] = {}
        for ev in new_failure_evs:
            weights = self.pending_failure_responsibilities.get(ev.id, {})
            if weights:
                controllable_failure_count += 1
                current_feasible = (
                    self.reward_builder.compute_failure_responsibility_weights(
                        ev, self.MCSs
                    )
                )
                if not current_feasible:
                    historically_attributed_failure_count += 1
                low_event_weight_sum = 0.0
                high_event_weight_sum = 0.0
                orphan_event_weight_sum = 0.0
                causal_records = self.pending_failure_causal_records.get(
                    ev.id, {}
                )
                for mcs_id, weight in weights.items():
                    if mcs_id in failure_weight_by_mcs_id:
                        record = causal_records.get(int(mcs_id), {})
                        age_steps = max(int(record.get(
                            'attribution_age_steps', 0
                        )), 0)
                        decayed_weight = float(weight) * (
                            FAILURE_RESPONSIBILITY_DECAY ** age_steps
                        )
                        failure_weight_by_mcs_id[mcs_id] += decayed_weight
                        failure_attribution_age_weighted_sum += (
                            decayed_weight * age_steps
                        )
                        cause_type = str(record.get(
                            'cause_type', 'unknown_opportunity'
                        ))
                        failure_cause_counts[cause_type] = (
                            failure_cause_counts.get(cause_type, 0) + 1
                        )
                        high_option_id = int(record.get(
                            'high_option_id', -1
                        ))
                        low_decision_id = int(record.get(
                            'low_decision_id', -1
                        ))
                        high_share = float(np.clip(
                            record.get('high_share', 0.0), 0.0, 1.0
                        ))
                        low_share = float(np.clip(
                            record.get('low_share', 0.0), 0.0, 1.0
                        ))
                        high_weight = (
                            decayed_weight * high_share
                            if high_option_id >= 0 else 0.0
                        )
                        low_weight = (
                            decayed_weight * low_share
                            if low_decision_id >= 0 else 0.0
                        )
                        if high_weight > 0.0:
                            high_failure_weight_by_option[high_option_id] = (
                                high_failure_weight_by_option.get(
                                    high_option_id, 0.0
                                ) + high_weight
                            )
                            high_event_weight_sum += high_weight
                            high_failure_route_count += 1
                        if low_weight > 0.0:
                            low_failure_weight_by_decision[low_decision_id] = (
                                low_failure_weight_by_decision.get(
                                    low_decision_id, 0.0
                                ) + low_weight
                            )
                            low_event_weight_sum += low_weight
                            low_failure_route_count += 1
                        orphan_weight = max(
                            decayed_weight - high_weight - low_weight, 0.0
                        )
                        orphan_event_weight_sum += orphan_weight
                        if orphan_weight > 1e-12:
                            orphan_failure_route_count += 1
                self.last_low_event_records.append({
                    'ev_id': int(ev.id),
                    'kind': 'controllable_failure',
                    'low_decision_id': -1,
                    'responsibility_weight': float(low_event_weight_sum),
                    'high_responsibility_weight': float(
                        high_event_weight_sum
                    ),
                    'orphan_responsibility_weight': float(
                        orphan_event_weight_sum
                    ),
                })
                high_failure_weight_sum += high_event_weight_sum
                low_failure_weight_sum += low_event_weight_sum
                orphan_failure_weight_sum += orphan_event_weight_sum
            else:
                uncontrollable_failure_count += 1

        attributed_success_count = int(sum(mcs_success_count_by_id.values()))
        success_weight_sum = float(sum(mcs_success_weight_by_id.values()))
        failure_weight_sum = float(sum(failure_weight_by_mcs_id.values()))
        low_success_weight_sum = float(sum(
            low_success_weight_by_decision.values()
        ))
        low_failure_weight_sum = float(sum(
            low_failure_weight_by_decision.values()
        ))
        forced_wait_count = sum(
            bool(event.get('forced_wait', False))
            for event in self.mcs_step_events.values()
        )
        voluntary_wait_count = sum(
            bool(event.get('voluntary_wait', False))
            for event in self.mcs_step_events.values()
        )
        energy_stranded_count = sum(
            bool(event.get('newly_energy_stranded', False))
            for event in self.mcs_step_events.values()
        )
        low_active_wait_count = sum(
            bool(event.get('low_stay_selected', False))
            and not bool(event.get('low_forced_stay', False))
            for event in self.mcs_step_events.values()
        )
        low_passive_wait_count = sum(
            bool(event.get('low_stay_selected', False))
            and bool(event.get('low_forced_stay', False))
            for event in self.mcs_step_events.values()
        )
        self.last_system_reward_event = {
            'new_success_count': len(new_success_evs),
            'new_failure_count': len(new_failure_evs),
            'new_mcs_success_count': new_mcs_success_count,
            'new_fcs_success_count': new_fcs_success_count,
            'fcs_success_kpi_count': new_fcs_success_count,
            'unattributed_mcs_success_count': unattributed_mcs_success_count,
            'attributed_mcs_success_count': attributed_success_count,
            'attributed_mcs_success_weight_sum': success_weight_sum,
            'low_attributed_mcs_success_weight_sum': low_success_weight_sum,
            'rescue_success_count': rescue_success_count,
            'replacement_success_count': replacement_success_count,
            'immediate_counterfactual_replacement_count': (
                immediate_counterfactual_replacement_count
            ),
            'future_rescue_evaluated_success_count': (
                future_rescue_evaluated_success_count
            ),
            'mcs_success_base_credit_sum': success_base_credit_sum,
            'mcs_success_rescue_bonus_sum': success_rescue_bonus_sum,
            'mcs_success_credit_sum': success_credit_sum,
            'counterfactual_failure_probability_sum': (
                counterfactual_failure_probability_sum
            ),
            'controllable_failure_count': controllable_failure_count,
            'historically_attributed_failure_count': (
                historically_attributed_failure_count
            ),
            'uncontrollable_failure_count': uncontrollable_failure_count,
            'unattributed_iev_failure_count': uncontrollable_failure_count,
            'controllable_failure_weight_sum': failure_weight_sum,
            'high_controllable_failure_weight_sum': (
                high_failure_weight_sum
            ),
            'low_controllable_failure_weight_sum': low_failure_weight_sum,
            'orphan_controllable_failure_weight_sum': (
                orphan_failure_weight_sum
            ),
            'high_failure_route_count': high_failure_route_count,
            'low_failure_route_count': low_failure_route_count,
            'orphan_failure_route_count': orphan_failure_route_count,
            'failure_attribution_age_weighted_sum': (
                failure_attribution_age_weighted_sum
            ),
            'failure_cause_high_recharge_count': failure_cause_counts.get(
                'high_recharge_deferral', 0
            ),
            'failure_cause_low_wait_count': failure_cause_counts.get(
                'low_wait_deferral', 0
            ),
            'failure_cause_low_reposition_count': failure_cause_counts.get(
                'low_reposition_deferral', 0
            ),
            'forced_wait_count': forced_wait_count,
            'voluntary_wait_count': voluntary_wait_count,
            'energy_stranded_count': energy_stranded_count,
            'low_active_wait_count': low_active_wait_count,
            'low_passive_wait_count': low_passive_wait_count,
        }

        self.last_mcs_reward_components = {}
        # 失败必须直接按稳定 decision/option ID 进入 rollout；不能先映射回
        # 失败发生时的当前 MCS，否则关闭后的历史决策会再次丢失。
        self.last_low_reward_by_decision = {
            int(decision_id): self.reward_builder.compute_low_failure_reward(
                weight
            )
            for decision_id, weight in low_failure_weight_by_decision.items()
        }
        self.last_high_reward_by_option = {
            int(option_id): (
                self.reward_builder.compute_high_success_reward(weight)
            )
            for option_id, weight in high_success_weight_by_option.items()
        }
        for option_id, weight in high_failure_weight_by_option.items():
            self.last_high_reward_by_option[int(option_id)] = float(
                self.last_high_reward_by_option.get(int(option_id), 0.0)
                + self.reward_builder.compute_high_failure_reward(weight)
            )
        self.last_system_reward_event[
            'high_routed_failure_reward_total'
        ] = float(sum(
            self.reward_builder.compute_high_failure_reward(weight)
            for weight in high_failure_weight_by_option.values()
        ))
        self.last_system_reward_event[
            'low_routed_failure_reward_total'
        ] = float(sum(
            self.reward_builder.compute_low_failure_reward(weight)
            for weight in low_failure_weight_by_decision.values()
        ))
        for mcs in self.MCSs:
            event = self.mcs_step_events.get(mcs.id, {})
            # ImmediateMatcher 在 update() 之后、reward 之前确认服务与补电，
            # 因此经济增量必须在此处用动作前快照重新计算，不能只看 update。
            event['profit_delta'] = float(
                mcs.total_profit
                - float(event.get('previous_total_profit', mcs.total_profit))
            )
            event['cost_delta'] = float(
                mcs.total_cost
                - float(event.get('previous_total_cost', mcs.total_cost))
            )
            attributed_success_count = mcs_success_count_by_id.get(mcs.id, 0)
            event['attributed_mcs_success_count'] = attributed_success_count
            event['attributed_mcs_success_weight'] = float(
                mcs_success_weight_by_id.get(mcs.id, 0.0)
            )
            # 仅保留按 MCS 的责任权重用于审计；训练失败奖励已经按原始
            # option/decision ID 直接路由，当前 option 分量必须为 0。
            event['routed_controllable_failure_weight'] = float(
                failure_weight_by_mcs_id.get(mcs.id, 0.0)
            )
            event['controllable_failure_weight'] = 0.0
            event['controllable_failure_count'] = (
                controllable_failure_count
            )
            event['uncontrollable_failure_count'] = (
                uncontrollable_failure_count
            )
            event['unattributed_iev_failure_count'] = (
                uncontrollable_failure_count
            )
            event['mcs_team_size'] = max(len(self.MCSs), 1)
            event['fcs_success_kpi_count'] = new_fcs_success_count
            low_decision_id = int(event.get('low_decision_id', -1))
            event['low_attributed_mcs_success_weight'] = float(
                low_success_weight_by_decision.get(low_decision_id, 0.0)
                if low_decision_id >= 0 else 0.0
            )
            event['low_controllable_failure_weight'] = 0.0
            post_spatial = self.reward_builder.compute_mcs_spatial_features(
                mcs, self.EVs, self.MCSs, self.FCSs
            )
            post_low_spatial = (
                self.reward_builder.compute_low_spatial_features(
                    mcs, self.EVs, self.FCSs
                )
            )
            event['post_attraction'] = post_spatial['attraction']
            event['post_competition'] = post_spatial['competition']
            event['post_spatial_potential'] = post_spatial['potential']
            event['post_immediate_iev_demand'] = post_spatial[
                'immediate_iev_demand'
            ]
            event['post_mcs_competition'] = post_spatial[
                'mcs_competition'
            ]
            event['post_fcs_competition'] = post_spatial[
                'fcs_competition'
            ]
            event['post_low_attraction'] = post_low_spatial['attraction']
            event['post_low_immediate_iev_attraction'] = post_low_spatial[
                'immediate_iev_attraction'
            ]
            event['post_low_mcs_competition'] = post_low_spatial[
                'mcs_competition'
            ]
            event['post_low_fcs_competition'] = post_low_spatial[
                'fcs_competition'
            ]
            event['post_low_spatial_desirability'] = post_low_spatial[
                'desirability'
            ]
            components = self.reward_builder.compute_mcs_reward(mcs, event)
            self.last_mcs_reward_components[mcs.id] = components
            mcs.total_reward += components['total']
            if low_decision_id >= 0 and components['low_total'] != 0.0:
                self.last_low_reward_by_decision[low_decision_id] = float(
                    self.last_low_reward_by_decision.get(
                        low_decision_id, 0.0
                    ) + components['low_total']
                )

        reward_n = []
        for agent in self.last_agents:
            if isinstance(agent, MCS):
                reward = self.last_mcs_reward_components[agent.id]['total']
            else:
                reward = self.reward_builder.compute_iev_reward(agent, {})
                agent.total_reward += reward
            reward_n.append(float(reward))
        return reward_n

    # ============================================================
    # 查询接口
    # ============================================================

    def get_global_state(self):
        return self.obs_builder.build_global_state(self.EVs, self.MCSs, self.FCSs)

    def get_done(self) -> bool:
        return self.current_step >= MAX_STEPS_PER_EPISODE

    def build_info(self) -> Dict:
        # 统计EV
        n_iev, n_quasi, n_charging, n_success, n_fail = 0, 0, 0, 0, 0
        for e in self.EVs:
            if e.is_iev:
                n_iev += 1
            elif e.is_quasi:
                n_quasi += 1
            elif e.is_charged and e.charge_pos is not None:
                # 正在充电
                n_charging += 1
            elif e.is_charged and e.charge_pos is None:
                n_success += 1
            elif e.is_fail:
                n_fail += 1
        # 统计MCS
        n_idle, n_task, n_recharging, n_broken, n_energy_stranded = 0, 0, 0, 0, 0
        for m in self.MCSs:
            if m.is_energy_stranded:
                n_energy_stranded += 1
            elif m.is_idle and not m.is_broken and not m.is_recharging:
                n_idle += 1
            elif m.is_task:
                n_task += 1
            elif m.is_recharging and not m.is_broken:
                n_recharging += 1
            elif m.is_broken:
                n_broken += 1
        # 统计FCS
        avail_slots, occ_slots = 0, 0
        for f in self.FCSs:
            avail_slots += f.available_slots
            occ_slots += f.occupied_slots

        return {
            'step': self.current_step,
            'num_iev': n_iev, 'num_quasi': n_quasi, 'num_charging': n_charging,
            'num_success': n_success, 'num_fail': n_fail,
            'num_idle_mcs': n_idle, 'num_task_mcs': n_task,
            'num_recharge_mcs': n_recharging, 'num_broken': n_broken,
            'num_energy_stranded': n_energy_stranded,
            'avail_slots': avail_slots, 'occ_slots': occ_slots,
            'num_agents': len(self.agents),
        }
