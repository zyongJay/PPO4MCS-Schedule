from typing import List
from world import World


class MultiAgentEnv:

    def __init__(self, seed: int = 42):
        self.world = World(seed)

    def step(self, action_n: List[dict]):
        # 1.根据action移动, 更新环境
        # action_n是环境跟新动作，与Actor层输出的动作有所区别

        self.world.update(action_n)                                 # agent移动、task_mcs更新状态、quasi_iev更新状态
        self.world.step_finish()                                    # 此处实现 old_agents =  agents
        # 2.进行充电匹配, 产生新一轮agents
        self.world.match_and_get_neibor()                           # 产生新一轮 agent
        # 3.为 new/old agents 分别获取局部obs
        new_obs_n, old_obs_n, done_n = self.world.get_obs_n()       # 构建obs
        # 4.为 last_agents 分配reward
        reward_n = self.world.mix_get_reward_n()                    # 计算每个agent获取的奖励

        return new_obs_n, old_obs_n, reward_n, done_n

    def reset(self):
        """重置环境, 返回初始观测"""
        self.world.reset_world()
        self.world.match_and_get_neibor()                   # 产生第一轮的agents
        new_obs_n, _old_obs_n, _done_n = self.world.get_obs_n()      # 初始局部obs
        return new_obs_n
