from env.config import *
from env.utils import haversine


# 电车
class Vehicle(object):
    def __init__(self, id, remain, distance, pos, conf):
        self.conf = conf
        self.ID = id  # 车辆编号
        self.remain = remain  # 车辆剩余电量，单位kwh
        self.pos = pos  # 车辆当前位置
        self.total_distance = distance  # 从起点到终点的总距离
        self.track = None  # 车辆轨迹，从历史数据中获取
        self.track_index = 0  # 车辆位于轨迹的哪一步
        self.destination = []  # 车辆终点经纬度
        self.wait_time = 0  # 充电等待时间step
        self.last_dist = 0  # action前后的距离变化
        self.last_pos = []  # action之前的位置

        self.need_charge = False  # 是否发出充电请求
        self.need_power = max(distance * POWER_UNIT - remain, 0)  # 电车到达终点的缺电量，对应论文中的电量缺口St
        self.set_charge()

        self.is_charged = False  # 是否有充电桩服务
        self.nearest_mcs_dist = 99  # 记录距离最近的mcs距离 用于选择最佳的MCS
        self.charge_pos = None  # 充电位置
        self.charge_mcs = None  # 为其服务的mcs

        self.near_task_MCS = []  # 附近的busyMCS列表
        self.near_IEV = []  # 附近的IEV列表
        self.near_quasi_IEV = []  # 附近的EV列表
        self.near_point = []  # 附近的调度点列表
        self.near_idle_MCS = []  # 附近的idle mcs列表
        self.reward = 0  # 奖励值

        self.total_wait_time = 0  # 总等待时间min
        self.total_reward = 0  # 总奖励值
        self.total_extra_dist = 0  # 总绕行距离

        self.action_next = None  # 下一步动作
        self.fail_charge = False  # 是否充电失败

        self.expense = 0  # 充电花费
        self.last_target_task_id = None  # 上次 action 选择的目标 task MCS ID

    def set_charge(self):
        # 每个车辆在每个step调用一次
        if self.remain < self.conf['LOW_POWER']:
            if self.need_charge:  # 如果电车原来就在需要充电的状态
                self.wait_time += 1  # 充电等待时间+1
                if self.wait_time > self.conf['MAX_WAIT_TIME'] or self.remain < LOWEST_POWER:      # 充电失败-如果充电等待时间超过最大等待时间或者剩余电量低于最低阈值
                    self.remain = self.conf['MEAN']  # 假设超过最大等待时间后以其他方式充上了电
                    self.need_charge = False         # 后续不参与调度
                    self.need_power = 0
                    self.fail_charge = True
            else:  # 如果电车刚转入需要充电的状态
                self.need_charge = True
                self.wait_time = 0
        else:   # 电量尚充足
            self.need_charge = False
            self.wait_time = 0

    def set_new_mcs(self, mcs):  # 为IEV设置充电的MCS
        # 取二者中点为充电地点
        self.charge_pos = [(float(self.pos[0]) + float(mcs.pos[0])) / 2,
                           (float(self.pos[1]) + float(mcs.pos[1])) / 2]
        # 假设走的路径是曼哈顿距离
        charge_dist = haversine(self.charge_pos[0], self.charge_pos[1], self.charge_pos[0], self.pos[1]) + haversine(self.charge_pos[0], self.pos[1], self.pos[0], self.pos[1])

        if self.charge_mcs is not None:     # 由于在选择最优MCS时，会多次调用该函数，因此需要将之前绑定的MCS充电任务取消
            self.charge_mcs.is_idle = True
            self.charge_mcs.is_arrive = False
            self.charge_mcs.charge_ev = None
            self.charge_mcs.charge_pos = None  # 若出现更优的MCS，则将原先次优的MCS取消绑定
            self.charge_mcs.charge_time = 0
            self.charge_mcs.charge_power = 0
            self.charge_mcs.reward = 0
            self.charge_mcs.cost -= self.charge_mcs.charge_dist * POWER_UNIT * MCS_CHARGE_PRICE
            self.charge_mcs.charge_dist = 0

        # 绑定该MCS
        mcs.cost += charge_dist * POWER_UNIT * MCS_CHARGE_PRICE
        mcs.charge_dist = charge_dist
        self.charge_mcs = mcs
        # self.reward = 1
        self.action_next = self.charge_pos      # 仅标注，后续无实际调度(充过电的EV就不参与后续调度了)

    # EV重置，格式化
    def reset(self, remain, distance, pos):
        self.remain = remain
        self.pos = pos
        self.total_distance = distance
        self.need_power = max(distance * POWER_UNIT - remain, 0)
        self.track_index = 0
        self.wait_time = 0        # 充电等待时间
        self.need_charge = False  # 是否处于缺电状态
        self.is_charged = False   # 是否充过电
        self.last_dist = 0
        self.last_pos = []
        self.fail_charge = False

        self.near_task_MCS = []
        self.near_IEV = []
        self.near_quasi_IEV = []
        self.near_idle_MCS = []
        self.near_point = []
        self.nearest_mcs_dist = 99
        self.set_charge()

        self.track = None  # 车辆轨迹，从历史数据中获取
        self.track_index = 0  # 车辆位于轨迹的哪一步

        self.charge_pos = None
        self.charge_mcs = None

        self.near_task_MCS = []
        self.near_IEV = []
        self.near_point = []
        self.reward = 0

        self.total_wait_time = 0
        self.total_reward = 0
        self.total_extra_dist = 0

        self.action_next = None
        self.expense = 0
        self.last_target_task_id = None

    def step_finish(self):
        self.near_task_MCS = []
        self.near_IEV = []
        self.near_point = []
        self.near_quasi_IEV = []
        self.near_idle_MCS = []
        self.nearest_mcs_dist = 99
        # self.reward = 0
        # self.last_dist = 0


# 移动充电桩，假设为一直电量充足的状态
class MCS(object):
    def __init__(self, id, pos, conf):
        self.conf = conf
        self.ID = id
        self.is_idle = True
        self.pos = pos  # 当前位置
        self.remain = 300

        self.charge_ev = None
        self.charge_pos = None  # 若MCS有任务在身，则设置充电地点和充电时间
        self.is_arrive = False
        self.charge_time = 0  # 时间单位为分钟
        self.charge_power = 0
        self.last_dist = 0
        self.last_pos = []

        self.total_profit = 0  # 一天总的充电收益
        self.total_reward = 0
        self.total_idle_time = 0  # 一天总的空闲巡航时间
        # self.cur_profit = 0  # 本次充电收益
        # self.cur_dist = 0 # 本次移动的距离

        self.near_task_MCS = []
        self.near_idle_MCS = []
        self.near_quasi_IEV = []
        self.near_IEV = []
        self.near_point = []        # 记录附近的调度点

        self.reward = 0
        self.cost = 0
        self.charge_dist = 0
        self.last_target_quasi_id = None  # 上次 action 选择的目标 quasi EV ID

    def reset(self, pos):
        self.is_idle = True
        self.pos = pos
        self.remain = 300

        self.charge_ev = None
        self.charge_pos = None  # 若MCS有任务在身，则设置充电地点和充电时间
        self.is_arrive = False
        self.charge_time = 0  # 时间单位为分钟
        self.charge_power = 0

        self.total_profit = 0  # 一天总的充电收益
        self.total_reward = 0
        self.total_idle_time = 0  # 一天总的空闲巡航时间

        self.near_task_MCS = []
        self.near_idle_MCS = []
        self.near_quasi_IEV = []
        self.near_IEV = []
        self.near_point = []

        self.reward = 0
        self.last_dist = 0
        self.last_pos = []
        self.cost = 0
        self.charge_dist = 0
        self.last_target_quasi_id = None

    def set_new_iev(self, iev):
        self.is_idle = False
        self.charge_ev = iev
        self.charge_pos = iev.charge_pos

        time = (iev.need_power / self.conf['CHARGE_SPEED']) * 60  # 充电多少分钟
        if time > 20:
            self.charge_time = 20   # 最多4个step(4 * 5 min)
        else:
            self.charge_time = time

        self.charge_power = (self.charge_time / 60) * self.conf['CHARGE_SPEED']     # 实际充电量

    def finish_charge(self):
        self.is_arrive = False
        self.is_idle = True
        self.charge_time = 0
        self.charge_pos = None
        self.charge_ev = None

        self.total_profit += self.charge_power * CHARGE_PRICE
        self.remain -= self.charge_power
        if self.remain < 10:    # mcs电量不足时，默认直接充满电（补能简化）
            self.remain = 300

        self.charge_power = 0
        self.charge_dist = 0

    def step_finish(self):
        self.near_task_MCS = []
        self.near_idle_MCS = []
        self.near_quasi_IEV = []
        self.near_IEV = []
        self.near_point = []


# 调度点的类定义
class TrafficNode(object):
    def __init__(self, id, pos):
        self.ID = id
        self.pos = pos


# 路网(调度点的集合)
class TrafficNet(object):
    def __init__(self):
        self.n = NUM_TRAFFIC_NODES
        self.nodes = []
        # 成都市区经纬度范围
        self.min_lon = 103.9787
        self.max_lon = 104.1631
        self.min_lat = 30.5965
        self.max_lat = 30.7309
