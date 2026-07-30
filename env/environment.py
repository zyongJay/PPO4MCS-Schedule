class MultiAgentEnv:
    def __init__(self, world):
        self.world = world

    def step(self, action_n):
        # 1.根据action移动，更新环境
        self.world.update(action_n)                     # agent移动、task_mcs更新状态、quasi_iev更新状态
        self.world.step_finish()                        # 此处实现 old_agents =  agents
        # 2.进行充电匹配
        self.world.match_and_get_neibor()               # 产生新一轮agent
        # 3.为new/old_agents分别获取局部obs
        new_obs_n, old_obs_n = self.world.get_obs_n()   # 构建全局异构图、聚合、提取对应obs
        # 4.为last_agents分配reward
        reward_n = self.world.mix_get_reward_n()        # 计算每个agent获取的奖励
        return new_obs_n, old_obs_n, reward_n

    def reset(self):
        self.world.reset_world()
        self.world.match_and_get_neibor()               # 产生第一轮的agents
        new_obs_n, old_obs_n = self.world.get_obs_n()   # 初始局部obs
        return new_obs_n
