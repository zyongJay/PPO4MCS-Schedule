import time
import math

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from env.config import *
from env.core import Vehicle, MCS, TrafficNet
from env.utils import *


class GraphState:
    def __init__(self, hetero_data, node_mapping, candidate_edge_index_dict, candidate_dst_by_src, mcs_pos_dict,
                 quasi_pos_dict):
        self.hetero_data = hetero_data
        self.node_mapping = node_mapping
        self.candidate_edge_index_dict = candidate_edge_index_dict
        self.candidate_dst_by_src = candidate_dst_by_src
        self.mcs_pos_dict = mcs_pos_dict
        self.quasi_pos_dict = quasi_pos_dict


class MCSGraphState(GraphState):
    pass


class IEVGraphState(GraphState):
    pass


class World(object):
    def __init__(self, conf, gnn, device):
        self.conf = conf
        self.EVs = []
        self.MCSs = []
        self.agents = []
        self.last_agents = []
        self.quasi_iev = []
        self.pending_iev = []
        self.tnet = TrafficNet()
        self.gnn = gnn
        self.MCS_init_pos = [np.random.uniform(self.tnet.min_lat, self.tnet.max_lat, self.conf['NUM_MCS']),
                             np.random.uniform(self.tnet.min_lon, self.tnet.max_lon, self.conf['NUM_MCS'])]
        self.ev_init_data = None
        self.device = device
        self.global_graph = None
        self.current_step = 0
        self.set_data()
        self.init_world()

    def set_data(self, date=3):
        date = 5
        self.ev_init_data = pd.read_csv(PATH_INIT_EV + str(date) + '.csv')

    def init_world(self):
        for i in range(self.conf['NUM_MCS']):
            pos = [float(self.MCS_init_pos[1][i]), float(self.MCS_init_pos[0][i])]
            mcs = MCS(i, pos, self.conf)
            self.MCSs.append(mcs)

        remain = np.maximum(0, np.random.normal(loc=self.conf['MEAN'], scale=self.conf['STAND'], size=self.conf['NUM_EV']))
        for index, row in self.ev_init_data.iterrows():
            if index >= self.conf['NUM_EV']:
                break
            ev = Vehicle(row['id'], remain[index], row['distance'], [float(row['lng']), float(row['lat'])], self.conf)
            tracks = row['track'].split(',')
            ev.track = [track.split(' ') for track in tracks]
            ev.track_index = 0
            ev.destination = ev.track[-1]
            ev.set_charge()
            self.EVs.append(ev)

        self.ev_init_data = None

    def step_finish(self):
        self.last_agents = self.agents
        self.agents = []
        self.quasi_iev = []
        self.global_graph = None
        for mcs in self.MCSs:
            mcs.step_finish()
        for ev in self.EVs:
            ev.step_finish()

    def reset_mcs(self):
        for i in range(len(self.MCSs)):
            pos = [self.MCS_init_pos[1][i], self.MCS_init_pos[0][i]]
            self.MCSs[i].reset(pos)

    def reset_vehicle(self):
        self.set_data()
        remain = np.maximum(0,
                            np.random.normal(loc=self.conf['MEAN'], scale=self.conf['STAND'], size=self.conf['NUM_EV']))
        for index, row in self.ev_init_data.iterrows():
            if index >= self.conf['NUM_EV']:
                break
            self.EVs[index].reset(remain[index], row['distance'], [row['lng'], row['lat']])
            self.EVs[index].track = [track.split() for track in row['track'].split(',')]
            self.EVs[index].track_index = 0
        self.ev_init_data = None

    def reset_world(self):
        self.agents = []
        self.last_agents = []
        self.quasi_iev = []
        self.global_graph = None
        self.current_step = 0
        self.reset_mcs()
        self.reset_vehicle()

    # 执行action
    def update(self, action_n):
        if len(action_n) > 0:
            for i in range(len(self.agents)):
                agent = self.agents[i]
                agent.last_pos = agent.pos
                dist = haversine(action_n[i][0], action_n[i][1], action_n[i][0], agent.pos[1]) + \
                       haversine(action_n[i][0], agent.pos[1], agent.pos[0], agent.pos[1])
                agent.last_dist = dist

                # 对agent更新状态
                if isinstance(agent, MCS):
                    agent.pos = action_n[i]
                    agent.remain -= dist * POWER_UNIT
                    if agent.remain < 0:
                        agent.remain += 200
                    agent.total_idle_time += 5
                    agent.cost += dist * POWER_UNIT * MCS_CHARGE_PRICE
                else:
                    agent.pos = action_n[i]
                    agent.remain -= dist * POWER_UNIT
                    agent.set_charge()      # 更新充电状态
                    agent.total_wait_time += 5
                    agent.total_extra_dist += dist

        # 对task_mcs更新状态
        for mcs in self.MCSs:
            if mcs.is_idle is False and mcs.is_arrive is False:
                ability = 300 * MCS_SPEED / 1000  # 一个step能走的最大距离（km）
                dist = haversine(mcs.pos[0], mcs.pos[1], mcs.charge_pos[0], mcs.charge_pos[1])  # Km
                if ability < dist:
                    mcs.pos[0] += (ability / dist) * (mcs.charge_pos[0] - mcs.pos[0])
                    mcs.pos[1] += (ability / dist) * (mcs.charge_pos[1] - mcs.pos[1])
                else:
                    mcs.pos = mcs.charge_pos
                    mcs.is_arrive = True
                    charge_time = max(0, (300 - (dist * 1000) / MCS_SPEED) / 60)  # 抵达充电位置后，剩余可用于充电的时间（min）
                    if charge_time > mcs.charge_time:
                        mcs.finish_charge()
                    else:
                        mcs.charge_time -= charge_time

            elif mcs.is_idle is False and mcs.is_arrive is True:
                if mcs.charge_time < 5:
                    mcs.finish_charge()
                else:
                    mcs.charge_time -= 5

        # 对quasi更新状态
        for ev in self.EVs:
            if ev.need_charge is False and ev.need_power > 0 and ev.is_charged is False:
                last_pos = ev.pos
                # 沿着轨迹移动
                ev.track_index += 1
                ev.pos = ev.track[ev.track_index]
                dist = haversine(ev.pos[0], ev.pos[1], last_pos[0], last_pos[1])
                ev.remain -= dist * POWER_UNIT
                ev.set_charge()
            elif ev.is_charged is True or ev.fail_charge:
                pass

        self.current_step += 1

    def is_in_area(self, pos):
        if 104.1631 > float(pos[0]) > 103.9787 and 30.7309 > float(pos[1]) > 30.5965:
            return True
        else:
            return False

    def mix_get_reward_n(self):
        reward_n = []

        num_agents = len(self.last_agents)
        if num_agents == 0:
            return reward_n

        # ==================== 物理极值与超参数定义 ====================
        MAX_CHARGE_PER_SESSION = self.conf['CHARGE_SPEED'] / 3.0  # 充电过程中的最大充电量（单位：Kwh）
        max_steps = max(1.0, float(self.conf.get('MAX_STEPS_PER_EPISODE', 100)))
        step_progress = min(1.0, self.current_step / max_steps)
        move_penalty_weight = 1.5
        mcs_local_weight = 0.8 - 0.3 * step_progress
        mcs_global_weight = 1.0 - mcs_local_weight

        # ==================== 全局统计指标收集器 ====================
        step_success_count = 0.0  # 本 Step 成功充电数
        step_fail_count = 0.0  # 本 Step 充电失败数

        local_rewards = [0.0] * num_agents
        mcs_indices = []
        iev_indices = []

        # ==================== 1. 局部奖励计算 (APF 势场法) ====================
        for idx, agent in enumerate(self.last_agents):
            r_local = 0.0

            if isinstance(agent, MCS):
                mcs_indices.append(idx)
                # 遍历agent附件的quasi：agent--quasi
                for quasi_iev in agent.near_quasi_IEV:
                    dist = haversine(quasi_iev.pos[0], quasi_iev.pos[1], agent.pos[0], agent.pos[1])
                    if min(quasi_iev.need_power, self.conf['CHARGE_SPEED'] / 3) <= agent.remain:
                        # 以“紧急度”主导吸引项：优先提升充电成功率，而非按需求电量大小拉开过大差距。
                        # 同时保留一个较弱的需求电量修正项，并整体做尺度约束，避免该项过大压制其它奖励。
                        urgent = 1.0 + max(0.0, 1.0 - quasi_iev.remain / 30.0)
                        urgent = float(np.clip(urgent, 1.0, 2.0))
                        need_power_ratio = min(quasi_iev.need_power, self.conf['CHARGE_SPEED'] / 3) / MAX_CHARGE_PER_SESSION
                        attract = urgent * (1.0 + 0.2 * need_power_ratio)
                        r_local += attract / (dist + 1)
                        # quasi--idle（竞争）
                        for idle_mcs in quasi_iev.near_idle_MCS:
                            if idle_mcs is not agent:
                                dist_i = haversine(agent.pos[0], agent.pos[1], idle_mcs.pos[0], idle_mcs.pos[1])
                                if idle_mcs.remain >= min(quasi_iev.need_power, self.conf['CHARGE_SPEED'] / 3):
                                    r_local -= 0.5 * (idle_mcs.remain / 300.0) / (dist_i + 1)
                        # quasi--task（竞争）
                        for task_mcs in quasi_iev.near_task_MCS:
                            dist_t = haversine(agent.pos[0], agent.pos[1], task_mcs.pos[0], task_mcs.pos[1])
                            if task_mcs.remain >= min(quasi_iev.need_power,
                                                      self.conf['CHARGE_SPEED'] / 3) and task_mcs.charge_time <= 5:
                                remain_power = task_mcs.remain - task_mcs.charge_power
                                r_local -= 0.5 * (remain_power / 300.0) / (dist_t + 1)
                move_energy = agent.last_dist * POWER_UNIT
                r_local -= move_penalty_weight * (move_energy / MAX_CHARGE_PER_SESSION)

                # 若当前无quasi可调度，前期鼓励合理巡航而不是长期原地等待
                if len(agent.near_quasi_IEV) == 0:
                    wider_quasi_count = 0
                    nearest_quasi_dist = float('inf')
                    for q in self.quasi_iev:
                        d = haversine(q.pos[0], q.pos[1], agent.pos[0], agent.pos[1])
                        if d < nearest_quasi_dist:
                            nearest_quasi_dist = d
                        if d < COMM_REGION_R * 2.0:
                            wider_quasi_count += 1
                    if wider_quasi_count > 0:
                        r_local += 0.15 * wider_quasi_count / (nearest_quasi_dist / COMM_REGION_R + 1)
                    else:
                        r_local -= 0.05

                # 对邻域内已失败IEV施加直接惩罚，强化“提升充电率”目标
                nearby_fail_count = 0
                for ev in self.EVs:
                    if ev.fail_charge and self.is_in_area(ev.pos):
                        dist_fail = haversine(agent.pos[0], agent.pos[1], ev.pos[0], ev.pos[1])
                        if dist_fail < COMM_REGION_R:
                            nearby_fail_count += 1
                if nearby_fail_count > 0:
                    r_local -= 0.5 * nearby_fail_count

                # ── Credit Assignment: MCS 成功被匹配到充电任务 ──
                # 若 MCS 上一步是 idle，本步 match_and_get_neibor() 后被分配了任务，
                # 说明其站位决策正确，应获得直接奖励
                if not agent.is_idle and agent.charge_ev is not None:
                    charge_ratio = min(agent.charge_power, MAX_CHARGE_PER_SESSION) / MAX_CHARGE_PER_SESSION
                    r_local += 1.0 * charge_ratio  # 成功接单奖励

            else:
                iev_indices.append(idx)
                # 遍历agent附件的task：agent--task
                for task_mcs in agent.near_task_MCS:
                    dist_m = haversine(task_mcs.pos[0], task_mcs.pos[1], task_mcs.charge_pos[0], task_mcs.charge_pos[1])
                    remain_power = task_mcs.remain - task_mcs.charge_power - dist_m * POWER_UNIT
                    if remain_power > min(agent.need_power, self.conf['CHARGE_SPEED'] / 3) \
                            and agent.total_wait_time + task_mcs.charge_time <= self.conf['MAX_WAIT_TIME'] * 5:

                        dist = haversine(task_mcs.charge_pos[0], task_mcs.charge_pos[1], agent.pos[0], agent.pos[1])
                        r_local += (remain_power / 300.0) / (dist + 1)
                        # task--iev（竞争）
                        for other_iev in task_mcs.near_IEV:
                            if other_iev is not agent:
                                dist_o = haversine(other_iev.pos[0], other_iev.pos[1], task_mcs.pos[0], task_mcs.pos[1])
                                r_local -= 0.5 * (other_iev.need_power / MAX_CHARGE_PER_SESSION) / (dist_o + 1)
                # ── Credit Assignment: IEV 充电成功/失败 ──
                if agent.is_charged:
                    step_success_count += 1
                    charge_ratio = min(agent.need_power, MAX_CHARGE_PER_SESSION) / MAX_CHARGE_PER_SESSION
                    r_local += 1.0 * charge_ratio
                if agent.fail_charge:
                    step_fail_count += 1
                    r_local -= 0.5

            local_rewards[idx] = r_local

        # 使用 tanh 做软截断替代 min-max 归一化，避免跨 MCS 零和博弈
        if mcs_indices:
            for i in mcs_indices:
                local_rewards[i] = np.tanh(local_rewards[i] * 0.3)
        if iev_indices:
            for i in iev_indices:
                local_rewards[i] = np.tanh(local_rewards[i] * 0.6)

        # 混合奖励
        for i, agent in enumerate(self.last_agents):
            r_global = 0.0
            if isinstance(agent, MCS):
                if not agent.is_idle:
                    r_global += step_success_count / (step_success_count + step_fail_count + 1)
                    r_global += min(agent.charge_power, MAX_CHARGE_PER_SESSION) / MAX_CHARGE_PER_SESSION
                agent.reward = 0.3 * r_global + 0.7 * local_rewards[i]
                agent.total_reward += agent.reward
                reward_n.append(agent.reward)
            else:
                if agent.is_charged:
                    r_global += step_success_count / (step_success_count + step_fail_count + 1)
                    r_global += min(agent.need_power, MAX_CHARGE_PER_SESSION) / MAX_CHARGE_PER_SESSION
                if agent.fail_charge:
                    r_global -= min(agent.need_power, MAX_CHARGE_PER_SESSION) / MAX_CHARGE_PER_SESSION
                agent.reward = 0.3 * r_global + 0.7 * local_rewards[i]
                agent.total_reward += agent.reward
                reward_n.append(agent.reward)

        return reward_n

    def get_reward_n(self):
        reward_n = []
        mcs_indices = []
        iev_indices = []

        for idx, agent in enumerate(self.last_agents):
            reward = 0
            # last_dist = haversine(agent.pos[0], agent.pos[1], agent.last_pos[0], agent.last_pos[1])
            if isinstance(agent, MCS):
                mcs_indices.append(idx)

                for quasi_iev in agent.near_quasi_IEV:
                    dist = haversine(quasi_iev.pos[0], quasi_iev.pos[1], agent.pos[0], agent.pos[1])
                    if min(quasi_iev.need_power, self.conf['CHARGE_SPEED'] / 3) <= agent.remain:
                        reward += min(quasi_iev.need_power, self.conf['CHARGE_SPEED'] / 3) / (dist + 1)

                        for idle_mcs in quasi_iev.near_idle_MCS:
                            if idle_mcs is not agent:
                                dist_i = haversine(idle_mcs.pos[0], idle_mcs.pos[1], quasi_iev.pos[0], quasi_iev.pos[1])
                                if idle_mcs.remain >= min(quasi_iev.need_power, self.conf['CHARGE_SPEED'] / 3):
                                    reward -= 0.1 * idle_mcs.remain / (dist_i + 1)

                        for task_mcs in quasi_iev.near_task_MCS:
                            dist_t = haversine(task_mcs.pos[0], task_mcs.pos[1], quasi_iev.pos[0], quasi_iev.pos[1])
                            if task_mcs.remain >= min(quasi_iev.need_power,
                                                      self.conf['CHARGE_SPEED'] / 3) and task_mcs.charge_time <= 5:
                                remain_power = task_mcs.remain - task_mcs.charge_power
                                reward -= 0.1 * remain_power / (dist_t + 1)

            else:
                iev_indices.append(idx)
                for task_mcs in agent.near_task_MCS:
                    dist_m = haversine(task_mcs.pos[0], task_mcs.pos[1], task_mcs.charge_pos[0], task_mcs.charge_pos[1])
                    remain_power = task_mcs.remain - task_mcs.charge_power - dist_m * POWER_UNIT
                    if remain_power > min(agent.need_power, self.conf['CHARGE_SPEED'] / 3) \
                            and agent.total_wait_time + task_mcs.charge_time <= self.conf['MAX_WAIT_TIME'] * 5:

                        dist = haversine(task_mcs.charge_pos[0], task_mcs.charge_pos[1], agent.pos[0], agent.pos[1])
                        reward += remain_power / (dist + 1)

                        for other_iev in task_mcs.near_IEV:
                            if other_iev is not agent:
                                dist_o = haversine(other_iev.pos[0], other_iev.pos[1], task_mcs.pos[0], task_mcs.pos[1])
                                reward -= 0.5 * other_iev.need_power / (dist_o + 1)

            # reward -= POWER_UNIT*last_dist
            reward_n.append(reward)

        # MCS 与 IEV 分开归一化，避免两类 agent 的奖励尺度互相干扰
        if mcs_indices:
            mcs_rewards = [reward_n[i] for i in mcs_indices]
            mcs_rewards_norm = min_max_norm(mcs_rewards)
            for i, r in zip(mcs_indices, mcs_rewards_norm):
                reward_n[i] = r

        if iev_indices:
            iev_rewards = [reward_n[i] for i in iev_indices]
            iev_rewards_norm = min_max_norm(iev_rewards)
            for i, r in zip(iev_indices, iev_rewards_norm):
                reward_n[i] = r

        for index, agent in enumerate(self.last_agents):
            agent.total_reward += reward_n[index]

        return reward_n

    def match(self):
        """调度前IEV与idle MCS进行充电匹配，剩余的IEV和idle MCS加入agent队列，产生新一轮agent"""
        self.agents = []
        self.quasi_iev = []

        # 充电匹配
        for ev in self.EVs:
            if self.is_in_area(ev.pos) and ev.need_charge is False and ev.need_power > 0 and ev.is_charged is False:
                self.quasi_iev.append(ev)
            if self.is_in_area(ev.pos) and ev.need_charge and ev.is_charged is False:
                for mcs in self.MCSs:
                    dist = haversine(mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1])
                    if dist < COMM_REGION_R:
                        if mcs.is_idle and mcs.remain > min(ev.need_power, self.conf['CHARGE_SPEED'] / 3):
                            ev.is_charged = True  # 该IEV一定能充上电
                            ev.expense = min(ev.need_power, self.conf['CHARGE_SPEED'] / 3) * CHARGE_PRICE
                            if dist < ev.nearest_mcs_dist:
                                ev.nearest_mcs_dist = dist
                                ev.set_new_mcs(mcs)
                                mcs.set_new_iev(ev)

        # 充电结束后，构建新一轮agent队列
        for i in range(len(self.EVs)):
            if self.EVs[i].need_charge and self.EVs[i].is_charged is False and self.is_in_area(self.EVs[i].pos):
                self.agents.append(self.EVs[i])  # 将未匹配的 IEV 添加进agents中

        for i in range(len(self.MCSs)):
            if self.MCSs[i].is_idle:
                self.agents.append(self.MCSs[i])  # 将未匹配的 idle MCS 添加进agents中

    def match_and_get_neibor(self):
        for ev in self.EVs:
            if self.is_in_area(ev.pos):
                if ev.need_charge is False and ev.need_power > 0 and ev.is_charged is False:
                    self.quasi_iev.append(ev)

                # IEV+quasi_IEV与MCS相互添加邻居信息并匹配
                if (ev.need_charge and ev.is_charged is False) or (ev.need_power > 0 and ev.is_charged is False) \
                        or (ev in self.last_agents):
                    for mcs in self.MCSs:
                        # 周围的task_MCS
                        dist = haversine(mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1])
                        if dist < COMM_REGION_R:
                            if mcs.is_idle is False:  # task_mcs
                                ev.near_task_MCS.append(mcs)

                                if ev.need_charge and ev.is_charged is False:   # 是发出充电请求的IEV
                                    mcs.near_IEV.append(ev)
                            else:   # 有空闲MCS
                                if ev.need_charge and mcs.remain > min(ev.need_power, self.conf['CHARGE_SPEED'] / 3) and ev.is_charged is False:  # 是发出充电请求的IEV
                                    ev.is_charged = True   # 此IEV一定有MCS服务
                                    ev.expense = min(ev.need_power, self.conf['CHARGE_SPEED']/3) * CHARGE_PRICE

                                    if dist < ev.nearest_mcs_dist:  # 此idle_MCS距离最近
                                        ev.nearest_mcs_dist = dist
                                        ev.set_new_mcs(mcs)
                                        mcs.set_new_iev(ev)

                                if ev.need_charge is False and ev.need_power > 0:  # 没有发出充电请求的EV
                                    mcs.near_quasi_IEV.append(ev)
                                    ev.near_idle_MCS.append(mcs)

        # 获取EVs之间的邻居关系
        for i in range(len(self.EVs)):
            # 匹配后，剩余的IEV与IEV相互添加邻居信息
            if self.EVs[i].need_charge and self.EVs[i].is_charged is False and self.is_in_area(self.EVs[i].pos):
                self.agents.append(self.EVs[i])
                for j in range(i + 1, len(self.EVs)):
                    if self.is_in_area(self.EVs[j].pos):
                        if self.EVs[j].need_charge and self.EVs[j].is_charged is False and \
                                haversine(self.EVs[j].pos[0], self.EVs[j].pos[1], self.EVs[i].pos[0], self.EVs[i].pos[1]) < COMM_REGION_R:
                            self.EVs[i].near_IEV.append(self.EVs[j])
                            self.EVs[j].near_IEV.append(self.EVs[i])
                        elif self.EVs[j].need_charge is False and self.EVs[j].need_power > 0 and \
                                haversine(self.EVs[j].pos[0], self.EVs[j].pos[1], self.EVs[i].pos[0], self.EVs[i].pos[1]) < COMM_REGION_R:
                            self.EVs[i].near_quasi_IEV.append(self.EVs[j])
                            self.EVs[j].near_IEV.append(self.EVs[i])

            # 上一批IEV添加邻居信息
            elif self.EVs[i] in self.last_agents and self.is_in_area(self.EVs[i].pos):      # 是上一轮调度后匹配成功 or 充电失败的IEV
                for j in range(i + 1, len(self.EVs)):
                    if self.is_in_area(self.EVs[j].pos):
                        dist = haversine(self.EVs[j].pos[0], self.EVs[j].pos[1], self.EVs[i].pos[0], self.EVs[i].pos[1])
                        if self.EVs[j].need_charge and self.EVs[j].is_charged is False and dist < COMM_REGION_R:
                            self.EVs[i].near_IEV.append(self.EVs[j])
                            self.EVs[j].near_IEV.append(self.EVs[i])
                        elif self.EVs[j].need_charge is False and self.EVs[j].need_power > 0 and dist < COMM_REGION_R:
                            self.EVs[i].near_quasi_IEV.append(self.EVs[j])
                            self.EVs[j].near_IEV.append(self.EVs[i])

        # 获取MCS相互的邻居关系
        for i in range(len(self.MCSs)):
            if self.MCSs[i].is_idle:
                self.agents.append(self.MCSs[i])
            for j in range(i + 1, len(self.MCSs)):
                if haversine(self.MCSs[i].pos[0], self.MCSs[i].pos[1], self.MCSs[j].pos[0], self.MCSs[j].pos[1]) < COMM_REGION_R:
                    if self.MCSs[j].is_idle:
                        self.MCSs[i].near_idle_MCS.append(self.MCSs[j])
                    else:
                        self.MCSs[i].near_task_MCS.append(self.MCSs[j])
                    if self.MCSs[i].is_idle:
                        self.MCSs[j].near_idle_MCS.append(self.MCSs[i])
                    else:
                        self.MCSs[j].near_task_MCS.append(self.MCSs[i])

    def get_obs_n(self):
        graph_data_dict = self.get_global_state()
        mcs_graph = graph_data_dict['mcs']
        iev_graph = graph_data_dict['iev']

        mcs_hetero = mcs_graph.hetero_data.to(self.device)
        iev_hetero = iev_graph.hetero_data.to(self.device)

        with torch.no_grad():
            mcs_h_dict = self.gnn['mcs'](mcs_hetero) if isinstance(self.gnn, dict) else self.gnn(mcs_hetero)
            iev_h_dict = self.gnn['iev'](iev_hetero) if isinstance(self.gnn, dict) else self.gnn(iev_hetero)

        # 构建原始特征字典，供 Q 网络显式建模源-目标交互
        raw_feat_dicts = {
            'mcs': {mcs.ID: torch.tensor(extract_mcs_features(mcs), dtype=torch.float32, device=self.device)
                    for mcs in self.MCSs},
            'ev': {ev.ID: torch.tensor(extract_vehicle_features(ev), dtype=torch.float32, device=self.device)
                   for ev in self.EVs},
        }

        new_obs_n = []
        old_obs_n = []

        for agent in self.agents:
            if isinstance(agent, MCS):
                new_obs_n.append(self.get_agent_obs(agent, mcs_h_dict, mcs_graph, raw_feat_dicts))
            else:
                new_obs_n.append(self.get_agent_obs(agent, iev_h_dict, iev_graph, raw_feat_dicts))

        for agent in self.last_agents:
            if agent in self.agents:
                if isinstance(agent, MCS):
                    old_obs_n.append(self.get_agent_obs(agent, mcs_h_dict, mcs_graph, raw_feat_dicts))
                else:
                    old_obs_n.append(self.get_agent_obs(agent, iev_h_dict, iev_graph, raw_feat_dicts))
            else:
                hidden_dim = HIDDEN_DIM
                if isinstance(agent, MCS):
                    obs = {
                        'feat': {
                            'self_h': torch.zeros(hidden_dim, device=self.device),
                            'target_h': torch.zeros(0, hidden_dim, device=self.device),
                            'self_raw': torch.zeros(MCS_FEAT_DIM, device=self.device),
                            'target_raw': torch.zeros(0, VEHICLE_FEAT_DIM, device=self.device),
                        },
                        'pos': torch.zeros(0, 2, device=self.device),
                        'type': 'MCS',
                        'id': agent.ID,
                        'done': True
                    }
                else:
                    obs = {
                        'feat': {
                            'self_h': torch.zeros(hidden_dim, device=self.device),
                            'target_h': torch.zeros(0, hidden_dim, device=self.device),
                            'self_raw': torch.zeros(VEHICLE_FEAT_DIM, device=self.device),
                            'target_raw': torch.zeros(0, MCS_FEAT_DIM, device=self.device),
                        },
                        'pos': torch.zeros(0, 2, device=self.device),
                        'type': 'IEV',
                        'id': agent.ID,
                        'done': True
                    }
                old_obs_n.append(obs)

        return new_obs_n, old_obs_n

    def get_agent_obs(self, agent, h_dict, graph_data, raw_feat_dicts=None):
        """
        obs = {
            'feat': {
                'self_h':      self_h,        # 自身 GNN 聚合嵌入
                'target_h':    target_h,       # 每个候选目标 GNN 聚合嵌入
                'self_raw':    self_raw,       # 自身原始特征 (用于显式交互建模)
                'target_raw':  target_raw,     # 每个候选目标原始特征
            },
            'pos': target_pos,                 # 每个候选动作的位置
            'type': 'MCS' if isinstance(agent, MCS) else 'IEV',
            'id': agent.ID,
            'done': False
        }
        """
        device = next(iter(h_dict.values())).device if h_dict else torch.device('cpu')
        hidden_dim = HIDDEN_DIM

        is_mcs_agent = isinstance(agent, MCS)
        if is_mcs_agent:
            node_type = 'idle'
            src_raw_dim = MCS_FEAT_DIM
            dst_raw_dim = VEHICLE_FEAT_DIM
        else:
            node_type = 'iev'
            src_raw_dim = VEHICLE_FEAT_DIM
            dst_raw_dim = MCS_FEAT_DIM

        node_mapping = graph_data.node_mapping

        # ── 自节点 GNN 嵌入 ──
        node_idx = node_mapping[node_type].get(agent.ID, -1)
        if 0 <= node_idx < h_dict[node_type].size(0):
            self_h = h_dict[node_type][node_idx]
        else:
            self_h = torch.zeros(hidden_dim, device=device)

        # ── 自节点原始特征 ──
        if raw_feat_dicts is not None:
            if is_mcs_agent:
                self_raw = raw_feat_dicts['mcs'].get(agent.ID,
                    torch.zeros(src_raw_dim, device=device))
            else:
                self_raw = raw_feat_dicts['ev'].get(agent.ID,
                    torch.zeros(src_raw_dim, device=device))
        else:
            self_raw = torch.zeros(src_raw_dim, device=device)

        # ── 候选目标: GNN 嵌入 + 原始特征 + 位置 ──
        target_h_list = []
        target_raw_list = []
        target_pos_list = []
        target_ids_list = []
        candidate_dst_by_src = graph_data.candidate_dst_by_src
        mcs_pos_dict = graph_data.mcs_pos_dict
        quasi_pos_dict = graph_data.quasi_pos_dict

        if is_mcs_agent:
            edge_types = [('idle', 'dispatch', 'quasi')]
        else:
            edge_types = [('iev', 'dispatch', 'task')]

        # 目标原始特征的备选形状 (task/idle→MCS, quasi/iev→EV)
        task_types = {'task', 'idle'}
        quasi_types = {'quasi', 'iev'}

        for edge_type in edge_types:
            if edge_type not in candidate_dst_by_src:
                continue
            dst_type = edge_type[2]
            dst_ids = candidate_dst_by_src[edge_type].get(agent.ID, [])

            for dst_id in dst_ids:
                # GNN 嵌入
                t_idx = node_mapping[dst_type].get(dst_id, -1)
                if 0 <= t_idx < h_dict[dst_type].size(0):
                    th = h_dict[dst_type][t_idx]
                else:
                    th = torch.zeros(hidden_dim, device=device)
                target_h_list.append(th)

                # 原始特征
                if raw_feat_dicts is not None:
                    if dst_type in task_types:
                        raw = raw_feat_dicts['mcs'].get(dst_id,
                            torch.zeros(dst_raw_dim, device=device))
                    else:
                        raw = raw_feat_dicts['ev'].get(dst_id,
                            torch.zeros(dst_raw_dim, device=device))
                else:
                    raw = torch.zeros(dst_raw_dim, device=device)
                target_raw_list.append(raw)

                # 位置
                if dst_type in task_types:
                    pos = mcs_pos_dict.get(dst_id)
                elif dst_type in quasi_types:
                    pos = quasi_pos_dict.get(dst_id)
                else:
                    pos = None

                if pos is not None:
                    target_pos_list.append(pos)
                    target_ids_list.append(dst_id)

        # ── Top-K 候选过滤：按距离只保留最近的 K 个候选 ──
        top_k = self.conf.get('TOP_K_MCS_CANDIDATES' if is_mcs_agent else 'TOP_K_IEV_CANDIDATES', None)
        if top_k and len(target_ids_list) > top_k:
            agent_pos = agent.pos  # [lon, lat]
            dists = [haversine(agent_pos[0], agent_pos[1], p[0], p[1])
                     for p in target_pos_list]
            sorted_idx = sorted(range(len(dists)), key=lambda i: dists[i])[:top_k]
            target_h_list = [target_h_list[i] for i in sorted_idx]
            target_raw_list = [target_raw_list[i] for i in sorted_idx]
            target_pos_list = [target_pos_list[i] for i in sorted_idx]
            target_ids_list = [target_ids_list[i] for i in sorted_idx]

        # ── 堆叠 ──
        if target_h_list:
            target_h = torch.stack(target_h_list)
            target_raw = torch.stack(target_raw_list)
        else:
            target_h = torch.zeros(0, hidden_dim, device=device)
            target_raw = torch.zeros(0, dst_raw_dim, device=device)

        target_pos = torch.tensor(target_pos_list, dtype=torch.float32,
                                  device=device) if target_pos_list else torch.zeros(0, 2, device=device)

        obs = {
            'feat': {
                'self_h': self_h,
                'target_h': target_h,
                'self_raw': self_raw,
                'target_raw': target_raw,
            },
            'pos': target_pos,
            'target_ids': target_ids_list,
            'type': 'MCS' if is_mcs_agent else 'IEV',
            'id': agent.ID,
            'done': False
        }
        return obs

    def get_global_state(self):
        if not self.global_graph:
            self.global_graph = {}
            # 分别构造MCS图和IEV图
            mcs_graph = self.build_mcs_graph()
            iev_graph = self.build_iev_graph()
            self.global_graph['mcs'] = mcs_graph
            self.global_graph['iev'] = iev_graph

        return self.global_graph

    def build_mcs_graph(self):
        """
        构建idleMCS类agent的异构图
        节点类型：idle, quasi, task
        信息边：idle<->idle, idle<->quasi, quasi<->task
        调度边：idle->quasi
        """
        hetero_data = HeteroData()
        node_dict = {'idle': {}, 'quasi': {}, 'task': {}}

        idle_list = [mcs for mcs in self.MCSs if mcs.is_idle]
        task_list = [mcs for mcs in self.MCSs if not mcs.is_idle]
        
        quasi_list = []
        for ev in self.quasi_iev:
            for idle_mcs in idle_list:
                dist = haversine(idle_mcs.pos[0], idle_mcs.pos[1], ev.pos[0], ev.pos[1])
                if dist < COMM_REGION_R:
                    quasi_list.append(ev)
                    break

        for idle in idle_list:
            node_dict['idle'][idle.ID] = extract_mcs_features(idle)

        for quasi in quasi_list:
            node_dict['quasi'][quasi.ID] = extract_vehicle_features(quasi)

        for task in task_list:
            node_dict['task'][task.ID] = extract_mcs_features(task)

        node_mapping = {'idle': {}, 'quasi': {}, 'task': {}}

        for node_type, nodes in node_dict.items():
            if node_type == 'quasi':
                feat_dim = VEHICLE_FEAT_DIM
            else:
                feat_dim = MCS_FEAT_DIM

            if len(nodes) == 0:
                hetero_data[node_type].x = torch.empty((0, feat_dim))
                continue
            else:
                feat_list = []
                for idx, (node_id, feat) in enumerate(nodes.items()):
                    feat_list.append(feat)
                    node_mapping[node_type][node_id] = idx

                hetero_data[node_type].x = torch.tensor(np.array(feat_list), dtype=torch.float32)

        # 信息边
        info_edges = {
            ('idle', 'info', 'idle'): {'src': [], 'dst': []},
            ('idle', 'info', 'quasi'): {'src': [], 'dst': []},
            ('quasi', 'info', 'idle'): {'src': [], 'dst': []},
            ('quasi', 'info', 'task'): {'src': [], 'dst': []},
            ('task', 'info', 'quasi'): {'src': [], 'dst': []},
        }

        for i in range(len(idle_list)):
            for j in range(i + 1, len(idle_list)):
                mcs_i = idle_list[i]
                mcs_j = idle_list[j]
                dist = haversine(mcs_i.pos[0], mcs_i.pos[1], mcs_j.pos[0], mcs_j.pos[1])
                if dist < COMM_REGION_R:
                    info_edges[('idle', 'info', 'idle')]['src'].append(node_mapping['idle'][mcs_i.ID])
                    info_edges[('idle', 'info', 'idle')]['dst'].append(node_mapping['idle'][mcs_j.ID])
                    info_edges[('idle', 'info', 'idle')]['src'].append(node_mapping['idle'][mcs_j.ID])
                    info_edges[('idle', 'info', 'idle')]['dst'].append(node_mapping['idle'][mcs_i.ID])

        for idle in idle_list:
            for quasi in quasi_list:
                dist = haversine(idle.pos[0], idle.pos[1], quasi.pos[0], quasi.pos[1])
                if dist < COMM_REGION_R:
                    info_edges[('idle', 'info', 'quasi')]['src'].append(node_mapping['idle'][idle.ID])
                    info_edges[('idle', 'info', 'quasi')]['dst'].append(node_mapping['quasi'][quasi.ID])
                    info_edges[('quasi', 'info', 'idle')]['src'].append(node_mapping['quasi'][quasi.ID])
                    info_edges[('quasi', 'info', 'idle')]['dst'].append(node_mapping['idle'][idle.ID])

        for quasi in quasi_list:
            for task in task_list:
                dist = haversine(quasi.pos[0], quasi.pos[1], task.pos[0], task.pos[1])
                if dist < COMM_REGION_R:
                    info_edges[('quasi', 'info', 'task')]['src'].append(node_mapping['quasi'][quasi.ID])
                    info_edges[('quasi', 'info', 'task')]['dst'].append(node_mapping['task'][task.ID])
                    info_edges[('task', 'info', 'quasi')]['src'].append(node_mapping['task'][task.ID])
                    info_edges[('task', 'info', 'quasi')]['dst'].append(node_mapping['quasi'][quasi.ID])

        for edge_type, indices in info_edges.items():
            src_type, rel_type, dst_type = edge_type
            if len(indices['src']) > 0:
                edge_index = torch.tensor([indices['src'], indices['dst']], dtype=torch.long)
                hetero_data[src_type, rel_type, dst_type].edge_index = edge_index
            else:
                hetero_data[src_type, rel_type, dst_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 调度边：仅保留 is_in_area 范围内的 quasi 作为合法候选
        candidate_edges = {
            ('idle', 'dispatch', 'quasi'): {'src': [], 'dst': []},
        }

        candidate_quasi_list = [quasi for quasi in quasi_list if self.is_in_area(quasi.pos)]

        for idle in idle_list:
            if idle.ID in node_mapping['idle']:
                for quasi in candidate_quasi_list:
                    if quasi.ID in node_mapping['quasi']:
                        dist = haversine(idle.pos[0], idle.pos[1], quasi.pos[0], quasi.pos[1])
                        if dist < COMM_REGION_R:
                            candidate_edges[('idle', 'dispatch', 'quasi')]['src'].append(node_mapping['idle'][idle.ID])
                            candidate_edges[('idle', 'dispatch', 'quasi')]['dst'].append(node_mapping['quasi'][quasi.ID])

        candidate_edge_index_dict = {}
        candidate_dst_by_src = {}

        idx_to_node_id = {
            ntype: {idx: nid for nid, idx in mapping.items()}
            for ntype, mapping in node_mapping.items()
        }

        for edge_type, indices in candidate_edges.items():
            src_type, rel_type, dst_type = edge_type
            candidate_dst_by_src[edge_type] = {}

            if len(indices['src']) > 0:
                candidate_edge_index_dict[edge_type] = torch.tensor([indices['src'], indices['dst']], dtype=torch.long)

                for i in range(len(indices['src'])):
                    src_idx = indices['src'][i]
                    dst_idx = indices['dst'][i]
                    src_id = idx_to_node_id[src_type][src_idx]
                    dst_id = idx_to_node_id[dst_type][dst_idx]
                    if src_id not in candidate_dst_by_src[edge_type]:
                        candidate_dst_by_src[edge_type][src_id] = []
                    candidate_dst_by_src[edge_type][src_id].append(dst_id)
            else:
                candidate_edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)

        mcs_pos_dict = {
            mcs.ID: [float(mcs.pos[0]), float(mcs.pos[1])]
            for mcs in self.MCSs
        }
        quasi_pos_dict = {
            ev.ID: [float(ev.pos[0]), float(ev.pos[1])]
            for ev in self.quasi_iev
        }

        return MCSGraphState(
            hetero_data, node_mapping, candidate_edge_index_dict, candidate_dst_by_src, mcs_pos_dict, quasi_pos_dict)

    def build_iev_graph(self):
        """
        构建IEV类agent的异构图
        节点类型：iev, task
        信息边：iev<->iev, iev<->task
        调度边：iev->task
        """
        hetero_data = HeteroData()
        node_dict = {'iev': {}, 'task': {}}

        iev_list = [agent for agent in self.agents if isinstance(agent, Vehicle)]
        all_task_list = [mcs for mcs in self.MCSs if not mcs.is_idle]
        task_list = []
        for task in all_task_list:
            for iev in iev_list:
                dist = haversine(iev.pos[0], iev.pos[1], task.pos[0], task.pos[1])
                if dist < COMM_REGION_R:
                    task_list.append(task)
                    break

        for iev in iev_list:
            node_dict['iev'][iev.ID] = extract_vehicle_features(iev)

        for task in task_list:
            node_dict['task'][task.ID] = extract_mcs_features(task)

        node_mapping = {'iev': {}, 'task': {}}

        for node_type, nodes in node_dict.items():
            if node_type == 'iev':
                feat_dim = VEHICLE_FEAT_DIM
            else:
                feat_dim = MCS_FEAT_DIM

            if len(nodes) == 0:
                hetero_data[node_type].x = torch.empty((0, feat_dim))
                continue
            else:
                feat_list = []
                for idx, (node_id, feat) in enumerate(nodes.items()):
                    feat_list.append(feat)
                    node_mapping[node_type][node_id] = idx

                hetero_data[node_type].x = torch.tensor(np.array(feat_list), dtype=torch.float32)

        info_edges = {
            ('iev', 'info', 'iev'): {'src': [], 'dst': []},
            ('iev', 'info', 'task'): {'src': [], 'dst': []},
            ('task', 'info', 'iev'): {'src': [], 'dst': []},
        }

        for i in range(len(iev_list)):
            for j in range(i + 1, len(iev_list)):
                ev_i = iev_list[i]
                ev_j = iev_list[j]
                dist = haversine(ev_i.pos[0], ev_i.pos[1], ev_j.pos[0], ev_j.pos[1])
                if dist < COMM_REGION_R:
                    info_edges[('iev', 'info', 'iev')]['src'].append(node_mapping['iev'][ev_i.ID])
                    info_edges[('iev', 'info', 'iev')]['dst'].append(node_mapping['iev'][ev_j.ID])
                    info_edges[('iev', 'info', 'iev')]['src'].append(node_mapping['iev'][ev_j.ID])
                    info_edges[('iev', 'info', 'iev')]['dst'].append(node_mapping['iev'][ev_i.ID])

        for iev in iev_list:
            for task in task_list:
                dist = haversine(iev.pos[0], iev.pos[1], task.pos[0], task.pos[1])
                if dist < COMM_REGION_R:
                    info_edges[('iev', 'info', 'task')]['src'].append(node_mapping['iev'][iev.ID])
                    info_edges[('iev', 'info', 'task')]['dst'].append(node_mapping['task'][task.ID])
                    info_edges[('task', 'info', 'iev')]['src'].append(node_mapping['task'][task.ID])
                    info_edges[('task', 'info', 'iev')]['dst'].append(node_mapping['iev'][iev.ID])

        for edge_type, indices in info_edges.items():
            src_type, rel_type, dst_type = edge_type
            if len(indices['src']) > 0:
                edge_index = torch.tensor([indices['src'], indices['dst']], dtype=torch.long)
                hetero_data[src_type, rel_type, dst_type].edge_index = edge_index
            else:
                hetero_data[src_type, rel_type, dst_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        candidate_edges = {
            ('iev', 'dispatch', 'task'): {'src': [], 'dst': []},
        }

        for iev in iev_list:
            if iev.ID in node_mapping['iev']:
                for task in task_list:
                    if task.ID in node_mapping['task']:
                        dist = haversine(iev.pos[0], iev.pos[1], task.pos[0], task.pos[1])
                        if dist < COMM_REGION_R:
                            candidate_edges[('iev', 'dispatch', 'task')]['src'].append(node_mapping['iev'][iev.ID])
                            candidate_edges[('iev', 'dispatch', 'task')]['dst'].append(node_mapping['task'][task.ID])

        candidate_edge_index_dict = {}
        candidate_dst_by_src = {}

        idx_to_node_id = {
            ntype: {idx: nid for nid, idx in mapping.items()}
            for ntype, mapping in node_mapping.items()
        }

        for edge_type, indices in candidate_edges.items():
            src_type, rel_type, dst_type = edge_type
            candidate_dst_by_src[edge_type] = {}

            if len(indices['src']) > 0:
                candidate_edge_index_dict[edge_type] = torch.tensor([indices['src'], indices['dst']], dtype=torch.long)

                for i in range(len(indices['src'])):
                    src_idx = indices['src'][i]
                    dst_idx = indices['dst'][i]
                    src_id = idx_to_node_id[src_type][src_idx]
                    dst_id = idx_to_node_id[dst_type][dst_idx]
                    if src_id not in candidate_dst_by_src[edge_type]:
                        candidate_dst_by_src[edge_type][src_id] = []
                    candidate_dst_by_src[edge_type][src_id].append(dst_id)
            else:
                candidate_edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)

        mcs_pos_dict = {
            mcs.ID: [float(mcs.pos[0]), float(mcs.pos[1])]
            for mcs in self.MCSs
        }
        quasi_pos_dict = {
            ev.ID: [float(ev.pos[0]), float(ev.pos[1])]
            for ev in self.quasi_iev
        }

        return IEVGraphState(
            hetero_data, node_mapping, candidate_edge_index_dict, candidate_dst_by_src, mcs_pos_dict, quasi_pos_dict)

    def step_level_graph(self):
        """
            每个 Step 仅调用一次！构建包含全场所有agent+相关task+相关quasi的单张大图。
        """
        hetero_data = HeteroData()
        node_dict = {'iev': {}, 'quasi': {}, 'idle': {}, 'task': {}}

        iev_list = [agent for agent in self.agents if isinstance(agent, Vehicle)]
        idle_list = [mcs for mcs in self.MCSs if mcs.is_idle]
        task_list = [mcs for mcs in self.MCSs if not mcs.is_idle]

        quasi_list = []         # 可能会用到的quasi
        for ev in self.quasi_iev:
            for agent in self.agents:
                dist = haversine(agent.pos[0], agent.pos[1], ev.pos[0], ev.pos[1])
                if dist < COMM_REGION_R:
                    quasi_list.append(ev)
                    break

        for iev in iev_list:
            node_dict['iev'][iev.ID] = extract_vehicle_features(iev)

        for quasi in quasi_list:
            node_dict['quasi'][quasi.ID] = extract_vehicle_features(quasi)

        for idle in idle_list:
            node_dict['idle'][idle.ID] = extract_mcs_features(idle)

        for task in task_list:
            node_dict['task'][task.ID] = extract_mcs_features(task)

        # 节点映射表 node_id->node_idx
        node_mapping = {'iev': {}, 'quasi': {}, 'idle': {}, 'task': {}}

        for node_type, nodes in node_dict.items():
            if node_type == 'iev' or node_type == 'quasi':
                feat_dim = VEHICLE_FEAT_DIM
            else:
                feat_dim = MCS_FEAT_DIM

            if len(nodes) == 0:
                hetero_data[node_type].x = torch.empty((0, feat_dim))
                continue
            else:
                feat_list = []
                for idx, (node_id, feat) in enumerate(nodes.items()):
                    feat_list.append(feat)
                    node_mapping[node_type][node_id] = idx  # 记录：ID "M_0" 在第 0 行

                hetero_data[node_type].x = torch.tensor(np.array(feat_list), dtype=torch.float32)

        # 构建全局信息边
        info_edges = {('quasi', 'info', 'idle'): {'src': [], 'dst': []},
                      ('idle', 'info', 'quasi'): {'src': [], 'dst': []},
                      ('task', 'info', 'iev'): {'src': [], 'dst': []},
                      ('iev', 'info', 'task'): {'src': [], 'dst': []},
                      ('idle', 'peer', 'idle'): {'src': [], 'dst': []},
                      ('iev', 'peer', 'iev'): {'src': [], 'dst': []},
                      }

        # 1. quasi ↔ idle 信息边
        for mcs in idle_list:
            for ev in quasi_list:
                dist = haversine(mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1])
                if dist < COMM_REGION_R and self.is_in_area(ev.pos) and mcs.is_idle:
                    info_edges[('quasi', 'info', 'idle')]['src'].append(node_mapping['quasi'][ev.ID])
                    info_edges[('quasi', 'info', 'idle')]['dst'].append(node_mapping['idle'][mcs.ID])
                    info_edges[('idle', 'info', 'quasi')]['src'].append(node_mapping['idle'][mcs.ID])
                    info_edges[('idle', 'info', 'quasi')]['dst'].append(node_mapping['quasi'][ev.ID])
        # 2. iev ↔ task 信息边
        for ev in iev_list:
            for mcs in task_list:
                dist = haversine(ev.pos[0], ev.pos[1], mcs.pos[0], mcs.pos[1])
                if dist < COMM_REGION_R and self.is_in_area(ev.pos):
                    info_edges[('iev', 'info', 'task')]['src'].append(node_mapping['iev'][ev.ID])
                    info_edges[('iev', 'info', 'task')]['dst'].append(node_mapping['task'][mcs.ID])
                    info_edges[('task', 'info', 'iev')]['src'].append(node_mapping['task'][mcs.ID])
                    info_edges[('task', 'info', 'iev')]['dst'].append(node_mapping['iev'][ev.ID])

        # 3. idle ↔ idle peer边
        for i in range(len(idle_list)):
            for j in range(i + 1, len(idle_list)):
                mcs_i = idle_list[i]
                mcs_j = idle_list[j]
                dist = haversine(mcs_i.pos[0], mcs_i.pos[1], mcs_j.pos[0], mcs_j.pos[1])
                if dist < COMM_REGION_R:
                    info_edges[('idle', 'peer', 'idle')]['src'].append(node_mapping['idle'][mcs_i.ID])
                    info_edges[('idle', 'peer', 'idle')]['dst'].append(node_mapping['idle'][mcs_j.ID])
                    info_edges[('idle', 'peer', 'idle')]['src'].append(node_mapping['idle'][mcs_j.ID])
                    info_edges[('idle', 'peer', 'idle')]['dst'].append(node_mapping['idle'][mcs_i.ID])

        # 4. iev ↔ iev peer边
        for i in range(len(iev_list)):
            for j in range(i + 1, len(iev_list)):
                ev_i = iev_list[i]
                ev_j = iev_list[j]
                dist = haversine(ev_i.pos[0], ev_i.pos[1], ev_j.pos[0], ev_j.pos[1])
                if dist < COMM_REGION_R:
                    info_edges[('iev', 'peer', 'iev')]['src'].append(node_mapping['iev'][ev_i.ID])
                    info_edges[('iev', 'peer', 'iev')]['dst'].append(node_mapping['iev'][ev_j.ID])
                    info_edges[('iev', 'peer', 'iev')]['src'].append(node_mapping['iev'][ev_j.ID])
                    info_edges[('iev', 'peer', 'iev')]['dst'].append(node_mapping['iev'][ev_i.ID])

        # 将信息边添加到hetero_data
        for edge_type, indices in info_edges.items():
            src_type, rel_type, dst_type = edge_type
            if len(indices['src']) > 0:
                edge_index = torch.tensor([indices['src'], indices['dst']], dtype=torch.long)
                hetero_data[src_type, rel_type, dst_type].edge_index = edge_index
            else:
                hetero_data[src_type, rel_type, dst_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 构建候选调度边
        candidate_edges = {
            ('idle', 'dispatch', 'quasi'): {'src': [], 'dst': []},
            ('iev', 'dispatch', 'task'): {'src': [], 'dst': []},
        }

        # 1. idle → dispatch → quasi
        for mcs in idle_list:
            if mcs.ID in node_mapping['idle']:
                for ev in quasi_list:
                    if ev.ID in node_mapping['quasi']:
                        dist = haversine(mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1])
                        if dist < COMM_REGION_R:
                            candidate_edges[('idle', 'dispatch', 'quasi')]['src'].append(node_mapping['idle'][mcs.ID])
                            candidate_edges[('idle', 'dispatch', 'quasi')]['dst'].append(node_mapping['quasi'][ev.ID])

        # 3. iev → dispatch → task
        for ev in iev_list:
            if ev.ID in node_mapping['iev']:
                for mcs in task_list:
                    if mcs.ID in node_mapping['task']:
                        dist = haversine(ev.pos[0], ev.pos[1], mcs.pos[0], mcs.pos[1])
                        if dist < COMM_REGION_R:
                            candidate_edges[('iev', 'dispatch', 'task')]['src'].append(node_mapping['iev'][ev.ID])
                            candidate_edges[('iev', 'dispatch', 'task')]['dst'].append(node_mapping['task'][mcs.ID])

        # 保存候选边信息（用于后续动作选择）
        candidate_edge_index_dict = {}
        candidate_edge_src_ids = {}
        candidate_edge_dst_ids = {}

        # index->node_id
        idx_to_node_id = {
            ntype: {idx: nid for nid, idx in mapping.items()}
            for ntype, mapping in node_mapping.items()
        }

        for edge_type, indices in candidate_edges.items():
            src_type, rel_type, dst_type = edge_type
            candidate_edge_src_ids[edge_type] = []
            candidate_edge_dst_ids[edge_type] = []

            if len(indices['src']) > 0:
                candidate_edge_index_dict[edge_type] = torch.tensor([indices['src'], indices['dst']], dtype=torch.long)

                # 记录边对应的原始节点ID（已优化为 O(1) 极速查找）
                for i in range(len(indices['src'])):
                    src_idx = indices['src'][i]  # 源节点索引
                    dst_idx = indices['dst'][i]  # 目标节点索引

                    # 直接通过反向字典拿 ID
                    candidate_edge_src_ids[edge_type].append(idx_to_node_id[src_type][src_idx])
                    candidate_edge_dst_ids[edge_type].append(idx_to_node_id[dst_type][dst_idx])

            else:
                candidate_edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)

        # 构建的原始特征字典（供后续使用）
        raw_x_dict = {node_type: hetero_data[node_type].x for node_type in node_dict.keys()}

        return hetero_data, node_mapping, raw_x_dict, candidate_edge_index_dict, candidate_edge_src_ids, candidate_edge_dst_ids
