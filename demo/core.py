import enum
import math
from typing import List, Tuple, Optional, Union

from config import *


class SlotState(enum.Enum):
    """FCS的一个充电位状态"""
    AVAILABLE = "available"
    OCCUPIED = "occupied"


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    # # 计算两点欧式距离，单位：m
    # return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # 或者切换为地球球面距离，单位：m
    # x1: lon1, y1: lat1, x2: lon2, y2: lat2
    lon1 = float(x1)
    lon2 = float(x2)
    lat1 = float(y1)
    lat2 = float(y2)
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
         math.cos(lat1) * math.cos(lat2))
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c * 1000.0


def is_in_area(x: float, y: float) -> bool:
    return AREA_X_MIN <= x <= AREA_X_MAX and AREA_Y_MIN <= y <= AREA_Y_MAX


def clip_position(x: float, y: float) -> Tuple[float, float]:
    x = max(AREA_X_MIN, min(AREA_X_MAX, x))
    y = max(AREA_Y_MIN, min(AREA_Y_MAX, y))
    return x, y


def move_toward_target(obj, pos):
    """obj向着pos移动一步（1个step内）
       返回：本step移动后剩余充电时间
    """
    target = list(pos)
    if target is None or obj.is_arrive:
        return STEP_DURATION_MIN

    dist_m = euclidean_distance(obj.pos[0], obj.pos[1], target[0], target[1])
    if dist_m < 1e-6:
        obj.is_arrive = True
        return STEP_DURATION_MIN

    max_dist_m = MOVE_SPEED * 60 * STEP_DURATION_MIN  # step内最大移动距离
    if max_dist_m >= dist_m:  # 1个step内能抵达
        obj.pos = target
        obj.is_arrive = True
        actual_dist_m = dist_m
    else:
        ratio = max_dist_m / dist_m
        obj.pos[0] += ratio * (target[0] - obj.pos[0])
        obj.pos[1] += ratio * (target[1] - obj.pos[1])
        actual_dist_m = max_dist_m

    # 本step内实际移动计算
    actual_dist_km = actual_dist_m / 1000.0
    energy_kwh = actual_dist_km * POWER_UNIT
    obj.remain -= energy_kwh
    charge_time_min = max(0, STEP_DURATION_MIN - actual_dist_m / MOVE_SPEED / 60)  # 本step剩余充电时间

    return charge_time_min


class EV:
    def __init__(self, ev_id, pos, remain_kwh, total_distance_km):
        self.id = ev_id
        self.pos = list(pos)
        self.last_pos = list(pos)
        self.remain = remain_kwh
        self.total_distance = total_distance_km  # 轨迹总长度
        self.detour_dist_km = 0.0

        # ── 轨迹 ──
        self.destination = []  # 终点坐标
        self.track = []
        self.track_index = 0
        self.arrived = False  # 是否已到达轨迹终点

        # ── 状态 ──
        self.state = None
        self.need_charge = False
        self.need_power = max(total_distance_km * POWER_UNIT - remain_kwh, 0.0)
        self.is_normal = False
        self.is_charged = False  # 是否已被匹配充电
        self.fail_charge = False  # 充电失败
        self.is_arrive = False  # 是否达到充电位置
        self.set_charge()

        # ── 统计 ──
        self.reward = 0  # 上一轮action后环境给的奖励
        self.total_reward = 0.0
        self.total_extra_dist_km = 0.0  # 调度移动(agent) + detour移动
        self.total_wait_time_min = 0.0  # 充电总延迟: 调度等待(agent) + 等待 + 绕行 + 充电 耗时
        self.expense = 0.0  # 充电费用: 充电量 * 充电价格
        # 要统计的：
        #1.绕行距离: detour_dist = dist(pi, pc) + dist(pc, ui) - dist(pi, ui)
        #2.充电总延迟(Charging delay):
        # 2.1 等待时间 d(pi, pc) / v or max(0, d(pi_ev, pc) / v - d(pi_mcs, pc) / v)
        # 2.2 绕行时间 detour_dist / v
        # 2.3 充电耗时 (st + c * detour_dist) / vc

        # ── 充电服务信息 ──
        self.charge_pos = None
        self.charge_provider = None
        self.charge_provider_id = -1
        self.charge_provider_type = ""  # "MCS" 或 "FCS"
        # 固化产生本次 MCS 服务的 Low Serve 决策，供延迟事件精确归因。
        self.service_low_decision_id = -1
        self.service_serve_option_id = -1
        # 匹配阶段基于同一物理约束矩阵计算的 FCS 可替代性代理。
        self.service_marginal_weight = 0.0
        self.low_service_marginal_weight = 0.0
        # 若本次 MCS 不接单，IEV 最终失败的反事实概率。成功奖励
        # 由较小基础信用与该救援概率共同决定。
        self.service_counterfactual_failure_probability = 0.0
        self.service_success_credit = 0.0
        self.service_rescue_diagnostics = {}
        self.feasible_mcs_count = 0
        self.feasible_fcs_slot_count = 0
        self.charge_power_kwh = 0.0  # 计划充电量
        self.charge_time_remain_min = 0.0  # 剩余充电时间

        # ── 邻居列表 ──
        self.near_quasi: List[EV] = []
        self.near_iev: List[EV] = []
        self.near_idle_mcs: List[MCS] = []
        self.near_task_mcs: List[MCS] = []
        self.near_available_fcs: List[FCS] = []
        self.near_busy_fcs: List[FCS] = []

        # ── 等待信息 ──
        self.wait_time_steps = 0  # 累计等待step数
        self.waiting_target_pos = []
        self.waiting_target = None
        self.waiting_target_id = -1
        self.waiting_target_type = ""  # "MCS" 或 "FCS"

    # 状态更新
    def set_charge(self):
        # 每个车辆在每个step调用一次
        if self.is_charged or self.fail_charge:
            return
        if self.need_power == 0:
            self.is_normal = True
            return
        if self.remain < EV_LOW_POWER_THRESHOLD:
            if self.need_charge:  # 如果电车原来就在需要充电的状态
                self.wait_time_steps += 1
                self.total_wait_time_min += STEP_DURATION_MIN
                if self.wait_time_steps > MAX_WAIT_TIME_STEPS or self.remain < EV_LOWEST_POWER:
                    self.fail_charge = True  # 超时失败
                    self.need_charge = False

            else:  # 刚变成IEV状态
                self.need_charge = True
                self.wait_time_steps = 0
        else:  # quasi状态
            self.need_charge = False
            self.wait_time_steps = 0

    # 绑定充电对象
    def set_target(self, obj, provider_id: int, provider_type: str,
                   charge_pos: List[float], charge_power_kwh: float,
                   charge_time_min: float, detour_dist_km: float):
        """绑定充电关系，注入充电信息"""
        self.charge_provider = obj  # 充电方实体
        self.charge_provider_type = provider_type  # 充电方类型 MCS/FCS
        self.charge_provider_id = provider_id  # 充电方ID
        self.charge_pos = list(charge_pos)  # 充电位置
        self.charge_power_kwh = charge_power_kwh  # 充电电量
        self.charge_time_remain_min = charge_time_min  # 充电剩余时间
        self.detour_dist_km = detour_dist_km

        # 清除 waiting target
        self.is_charged = True
        self.waiting_target_id = -1
        self.waiting_target_type = ""
        self.waiting_target_pos = None

        # 统计指标更新
        # 计算绕行距离
        self.total_extra_dist_km += detour_dist_km

        # 计算充电总延迟
        self_move_dist = euclidean_distance(self.pos[0], self.pos[1], charge_pos[0], charge_pos[1])
        obj_move_dist = euclidean_distance(obj.pos[0], obj.pos[1], charge_pos[0], charge_pos[1])
        if provider_type == "FCS":
            wait_min = self_move_dist / MOVE_SPEED / 60.0
        else:
            wait_iev = self_move_dist / MOVE_SPEED / 60.0
            wait_mcs = obj_move_dist / MOVE_SPEED / 60.0
            wait_min = max(wait_iev, wait_mcs)
        self.total_wait_time_min += wait_min + detour_dist_km * 1000.0 / MOVE_SPEED / 60.0 + 60.0 * charge_power_kwh / CHARGE_SPEED

        # 计算充电费用
        self.expense += CHARGE_PRICE * charge_power_kwh

    # 充电任务推进    交给充电方MCS / FCS 具体实现
    def advance_charging(self):
        """推进充电任务，交给充电方MCS / FCS 具体实现"""
        pass

    # 充电任务结束后重置    交给充电方MCS / FCS 具体实现
    def finish_charging(self):
        """充电结束 → 转入 SUCCESS 状态, 后续沿轨迹移动至终点。"""
        # self.charge_provider = None
        # self.charge_provider_type = ""
        # self.charge_provider_id = -1
        # self.charge_pos = None
        self.waiting_target_id = -1
        self.waiting_target_type = ""
        self.waiting_target_pos = None

        self.charge_power_kwh = 0.0  # 计划充电量
        self.charge_time_remain_min = 0.0  # 剩余充电时间
        # 下一个step从充电位置移动到下一个轨迹点

    def reset(self, pos: List[float], remain_kwh: float):
        self.pos = list(pos)
        self.last_pos = list(pos)
        self.remain = remain_kwh
        # self.total_distance = total_distance_km

        # ── 轨迹 ──
        # self.destination = []  # 终点坐标
        # self.track = []
        self.track_index = 0
        self.arrived = False  # 是否已到达轨迹终点

        # ── 状态 ──
        self.state = None
        self.need_charge = False
        self.need_power = max(self.total_distance * POWER_UNIT - remain_kwh, 0.0)
        self.is_normal = False
        self.is_charged = False  # 是否已被匹配充电
        self.fail_charge = False  # 充电失败
        self.set_charge()

        # ── 统计 ──
        self.reward = 0  # 上一轮action后环境给的奖励
        self.total_reward = 0.0
        self.total_extra_dist_km = 0.0
        self.expense = 0.0

        # ── 充电服务信息 ──
        self.charge_pos = None
        self.charge_provider = None
        self.charge_provider_id = -1
        self.charge_provider_type = ""  # "MCS" 或 "FCS"
        self.service_low_decision_id = -1
        self.service_serve_option_id = -1
        self.service_marginal_weight = 0.0
        self.low_service_marginal_weight = 0.0
        self.service_counterfactual_failure_probability = 0.0
        self.service_success_credit = 0.0
        self.service_rescue_diagnostics = {}
        self.feasible_mcs_count = 0
        self.feasible_fcs_slot_count = 0
        self.charge_power_kwh = 0.0  # 计划充电量
        self.charge_time_remain_min = 0.0  # 剩余充电时间

        # ── 邻居列表 ──
        self.near_quasi = []
        self.near_iev = []
        self.near_idle_mcs = []
        self.near_task_mcs = []
        self.near_available_fcs = []
        self.near_busy_fcs = []

        # ── 等待信息 ──
        self.wait_time_steps = 0  # 累计等待step数
        self.total_wait_time_min = 0.0
        self.waiting_target_pos = []
        self.waiting_target = None
        self.waiting_target_id = -1
        self.waiting_target_type = ""  # "MCS" 或 "FCS"

    def step_finish(self):
        self.last_pos = self.pos
        self.near_quasi = []
        self.near_iev = []
        self.near_idle_mcs = []
        self.near_task_mcs = []
        self.near_available_fcs = []
        self.near_busy_fcs = []
        self.reward = 0

    # 属性
    @property
    def is_iev(self) -> bool:
        return not self.is_normal and not self.is_charged and not self.fail_charge and self.need_charge

    @property
    def is_quasi(self) -> bool:
        return not self.is_normal and not self.is_charged and not self.fail_charge and not self.need_charge

    @property
    def is_success(self) -> bool:
        return self.is_charged

    @property
    def is_fail(self) -> bool:
        return self.fail_charge

    @property
    def is_active(self) -> bool:
        """是否需要参与调度系统 (QUASI/IEV/充电中)"""
        return not self.is_normal and not self.is_charged and not self.fail_charge


class MCS:
    def __init__(self, mcs_id: int, pos: List[float], remain_kwh: float = MCS_BATTERY_CAPACITY):
        self.id = mcs_id
        self.pos = list(pos)
        self.remain = remain_kwh
        self.is_idle = True
        self.is_recharging = False
        self.is_broken = False
        # energy-stranded 与 broken 同属永久失去后续服务能力的终止状态。
        # 前者表示车辆未物理损坏，但剩余电量已无法到达任何 FCS。
        self.is_energy_stranded = False
        self.last_pos = list(pos)
        self.reward = 0

        # ── 当前任务对象信息 ──
        self.current_target = None
        self.current_target_type = ""  # "IEV" / "FCS" / "QUASI"
        self.current_target_id = -1
        self.current_target_pos = None

        # ── 充电任务信息 ──
        self.charge_power_kwh = 0.0
        self.charge_time_remain_min = 0.0
        self.is_arrive = False  # 是否抵达充电位置

        # ── 邻居列表 ──
        self.near_quasi = []
        self.near_iev = []
        self.near_idle_mcs = []
        self.near_task_mcs = []
        self.near_available_fcs = []
        self.near_busy_fcs = []

        # ── 统计 ──
        self.total_energy_consumed = 0.0  # 累计耗于移动的电量
        self.total_charged_kwh = 0.0  # 累计充给IEV的电量
        self.total_cost = 0.0  # 移动开销 + 充电成本 + 补电成本
        self.total_profit = 0.0  # 给IEV充电的净利润
        self.total_reward = 0.0
        self.total_idle_time_min = 0.0
        # 连续主动 Wait 的环境 step 数；forced Wait 不累计。
        self.consecutive_voluntary_wait_steps = 0
        # 当前仍在生效的 High Serve / Low 空间决策。一个 Serve Option
        # 最多允许一次 MCS-IEV 匹配，避免任务完成的同一 step 自动接取
        # 第二个订单并继续沿用旧 Option ID。
        self.active_low_decision_id = -1
        self.active_serve_option_id = -1
        # High option_id 对 Serve/Recharge 均稳定存在；不能再借用仅 Serve
        # 有效的 active_serve_option_id 做历史失败回写。
        self.active_high_option_id = -1
        self.active_high_mode = ""
        self.active_serve_has_matched = False
        self.active_low_candidate_id = -1
        self.active_low_started_step = -1

    # 注入任务信息    MCS-IEV充电 / MCS-FCS补电共用
    def set_target(self, obj, target_type: str, target_id: int,
                   target_pos: List[float],
                   charge_power: float = 0.0,
                   charge_time: float = 0.0):
        self.current_target = obj
        self.current_target_id = target_id
        self.current_target_type = target_type
        self.current_target_pos = list(target_pos)
        self.charge_power_kwh = charge_power
        self.charge_time_remain_min = charge_time

        dist_km = euclidean_distance(self.pos[0], self.pos[1], target_pos[0], target_pos[1]) / 1000.0
        energy_consumed = POWER_UNIT * dist_km

        if target_type == "IEV":
            self.is_idle = False
            self.is_arrive = False

            # 统计指标更新
            self.total_charged_kwh += charge_power
            self.total_energy_consumed += energy_consumed
            self.total_profit += charge_power * CHARGE_PRICE - (energy_consumed + charge_power) * RC_PRICE
            self.total_cost += (energy_consumed + charge_power) * RC_PRICE

        elif target_type == "FCS":
            self.is_idle = False  # 补电期间不参与充电匹配
            self.is_recharging = True
            self.is_arrive = False

            # 统计指标更新
            self.total_energy_consumed += energy_consumed
            self.total_cost += (energy_consumed + charge_power) * RC_PRICE

        return -1

    # 推进充电任务 IEV-MCS
    def advance_charging(self):
        if not self.is_arrive or not self.current_target.is_arrive:
            # IEV-MCS未到齐，则先推进移动前往充电点
            rest_time_mcs = move_toward_target(self, self.current_target_pos)
            rest_time_iev = move_toward_target(self.current_target, self.current_target_pos)
            rest_time = min(rest_time_mcs, rest_time_iev)  # 抵达充电位置后剩余充电时间（min）
            # 本step移动后，若双方均到达且仍有时间，则剩余时间用于充电
            if self.is_arrive and self.current_target.is_arrive and rest_time > 0:
                if self.charge_time_remain_min > rest_time:
                    # 充电指标更新
                    self.charge_time_remain_min -= rest_time
                    self.charge_power_kwh -= rest_time * CHARGE_SPEED_PER_MIN
                    self.current_target.charge_time_remain_min -= rest_time
                    self.current_target.charge_power_kwh -= rest_time * CHARGE_SPEED_PER_MIN
                    # 实际电量转移
                    self.remain -= rest_time * CHARGE_SPEED_PER_MIN
                    self.current_target.remain += rest_time * CHARGE_SPEED_PER_MIN
                else:
                    # 实际电量转移
                    self.remain -= self.charge_power_kwh
                    self.current_target.remain += self.charge_power_kwh
                    self.current_target.finish_charging()
                    self.finish_charging()
        else:  # 二者均已抵达充电位置
            # 任务状态变化
            charge_min = min(STEP_DURATION_MIN, self.charge_time_remain_min)
            charged = CHARGE_SPEED_PER_MIN * charge_min
            charged = min(charged, self.charge_power_kwh)
            self.charge_time_remain_min -= charge_min
            self.charge_power_kwh -= charged
            self.current_target.charge_time_remain_min -= charge_min
            self.current_target.charge_power_kwh -= charged

            # MCS-IEV 电量变化 (实时检测电量变化)
            # IEV MCS 各项统计指标在匹配成功时已经计算
            self.remain -= charged
            self.current_target.remain += charged

            if self.charge_time_remain_min <= 0 or self.charge_power_kwh <= 0:
                self.current_target.finish_charging()
                self.finish_charging()

    # 充电任务结束后重置
    def finish_charging(self):
        """充电任务完成，更新MCS状态"""
        self.is_idle = True
        self.is_arrive = False
        self.is_recharging = False

        self.current_target = None
        self.current_target_id = -1
        self.current_target_type = ""
        self.current_target_pos = None
        self.charge_power_kwh = 0
        self.charge_time_remain_min = 0

    def reset(self, pos: List[float], remain_kwh: float = MCS_BATTERY_CAPACITY):
        self.pos = list(pos)
        self.remain = remain_kwh
        self.is_idle = True
        self.is_recharging = False
        self.is_broken = False
        self.is_energy_stranded = False
        self.last_pos = list(pos)
        self.reward = 0

        # ── 当前任务对象信息 ──
        self.current_target = None
        self.current_target_type = ""  # "IEV" / "FCS" / "QUASI"
        self.current_target_id = -1
        self.current_target_pos = None

        # ── 充电任务信息 ──
        self.charge_power_kwh = 0.0
        self.charge_time_remain_min = 0.0
        self.is_arrive = False  # 是否抵达充电位置

        # ── 邻居列表 ──
        self.near_quasi = []
        self.near_iev = []
        self.near_idle_mcs = []
        self.near_task_mcs = []
        self.near_available_fcs = []
        self.near_busy_fcs = []

        # ── 统计 ──
        self.total_energy_consumed = 0.0  # 累计耗于移动的电量
        self.total_charged_kwh = 0.0  # 累计充给IEV的电量
        self.total_cost = 0.0  # 移动开销 + 充电成本 + 补电成本
        self.total_profit = 0.0  # 给IEV充电的毛利润
        self.total_reward = 0.0
        self.total_idle_time_min = 0.0
        self.consecutive_voluntary_wait_steps = 0
        self.active_low_decision_id = -1
        self.active_serve_option_id = -1
        self.active_high_option_id = -1
        self.active_high_mode = ""
        self.active_serve_has_matched = False
        self.active_low_candidate_id = -1
        self.active_low_started_step = -1

    def step_finish(self):
        self.last_pos = self.pos
        self.near_quasi = []
        self.near_iev = []
        self.near_idle_mcs = []
        self.near_task_mcs = []
        self.near_available_fcs = []
        self.near_busy_fcs = []

    # 属性
    @property
    def is_task(self) -> bool:
        return (
            not self.is_idle
            and not self.is_recharging
            and not self.is_broken
            and not self.is_energy_stranded
        )


class FCS:
    def __init__(self, fcs_id: int, pos: List[float], num_slots: int = FCS_SLOTS_PER_STATION):
        self.id = fcs_id
        self.pos = list(pos)
        self.num_slots = num_slots

        # 每个slot的信息
        self.slot_states: List[SlotState] = [SlotState.AVAILABLE] * num_slots
        self.slot_target: List[Optional[Union[EV, MCS]]] = [None] * num_slots
        self.slot_iev_id: List[int] = [-1] * num_slots  # 正在充电的EV ID
        self.slot_mcs_id: List[int] = [-1] * num_slots  # 正在补电的MCS ID
        self.slot_charge_remain_kwh: List[float] = [0.0] * num_slots
        self.slot_charge_remain_min: List[float] = [0.0] * num_slots

        # 统计指标
        self.total_idle_time_min = 0.0
        self.total_charged_kwh = 0.0  # 为IEV充电 和 为MCS补电的总电量
        self.total_profit = 0.0  # 为IEV充电 和 为MCS补电 的净利润
        self.total_cost = 0.0  # 充电成本

        # 邻居信息
        self.near_quasi = []
        self.near_iev = []
        self.near_idle_mcs = []
        self.near_task_mcs = []
        self.near_available_fcs = []
        self.near_busy_fcs = []

    @property
    def capacity(self) -> int:
        return self.num_slots

    @property
    def occupied_slots(self) -> int:
        return sum(1 for s in self.slot_states if s == SlotState.OCCUPIED)

    @property
    def available_slots(self) -> int:
        return self.capacity - self.occupied_slots

    @property
    def is_idle(self) -> bool:
        return self.available_slots == self.capacity

    @property
    def is_busy(self) -> bool:
        return self.available_slots == 0

    def has_available_slot(self) -> bool:
        return self.available_slots > 0

    def apply_slot(self) -> int:
        """返回第一个可用充电位索引, 无则返回-1"""
        for i, s in enumerate(self.slot_states):
            if s == SlotState.AVAILABLE:
                return i
        return -1

    def release_slot(self, slot_idx: int):
        """释放充电位"""
        if 0 <= slot_idx < self.num_slots:
            self.slot_target[slot_idx] = None
            self.slot_states[slot_idx] = SlotState.AVAILABLE
            self.slot_iev_id[slot_idx] = -1
            self.slot_mcs_id[slot_idx] = -1
            self.slot_charge_remain_kwh[slot_idx] = 0.0
            self.slot_charge_remain_min[slot_idx] = 0.0

    def set_target(self, target, target_type, target_id, charge_power_kwh, charge_time_min):
        idx = self.apply_slot()  # 分配一个空闲slot
        if idx < 0:
            return -1
        if target_type == "IEV":
            self.slot_target[idx] = target
            self.slot_iev_id[idx] = target_id
            self.slot_states[idx] = SlotState.OCCUPIED
            self.slot_charge_remain_kwh[idx] = charge_power_kwh
            self.slot_charge_remain_min[idx] = charge_time_min

            # 统计指标
            self.total_charged_kwh += charge_power_kwh
            self.total_profit += charge_power_kwh * (CHARGE_PRICE - PG_PRICE)
            self.total_cost += charge_power_kwh * PG_PRICE

        if target_type == "MCS":
            self.slot_target[idx] = target
            self.slot_mcs_id[idx] = target_id
            self.slot_states[idx] = SlotState.OCCUPIED
            self.slot_charge_remain_kwh[idx] = charge_power_kwh
            self.slot_charge_remain_min[idx] = charge_time_min

            # 统计指标
            self.total_charged_kwh += charge_power_kwh
            self.total_profit += charge_power_kwh * (RC_PRICE - PG_PRICE)
            self.total_cost += charge_power_kwh * PG_PRICE
        return idx

    def advance_charging(self):
        """推进所有占用位的充电进度"""
        for idx in range(self.num_slots):
            if self.slot_states[idx] != SlotState.OCCUPIED or self.slot_target[idx] is None:
                continue
            charge_speed = CHARGE_SPEED_PER_MIN if isinstance(self.slot_target[idx], EV) else RECHARGE_SPEED_PER_MIN
            if not self.slot_target[idx].is_arrive:
                rest_time = move_toward_target(self.slot_target[idx], self.pos)
                if self.slot_target[idx].is_arrive and rest_time > 0:
                    if self.slot_charge_remain_min[idx] > rest_time:
                        # 充电信息更新
                        self.slot_charge_remain_min[idx] -= rest_time
                        self.slot_charge_remain_kwh[idx] -= rest_time * charge_speed
                        self.slot_target[idx].charge_time_remain_min -= rest_time
                        self.slot_target[idx].charge_power_kwh -= rest_time * charge_speed
                        # 实际电量转移
                        self.slot_target[idx].remain += rest_time * charge_speed
                    else:
                        self.slot_target[idx].remain += self.slot_charge_remain_kwh[idx]
                        self.slot_target[idx].finish_charging()
                        self.release_slot(idx)
            else:
                charge_min = min(STEP_DURATION_MIN, self.slot_charge_remain_min[idx])
                charged = charge_speed * charge_min
                self.slot_charge_remain_min[idx] -= charge_min
                self.slot_charge_remain_kwh[idx] -= charged
                self.slot_target[idx].charge_time_remain_min -= charge_min
                self.slot_target[idx].charge_power_kwh -= charged

                self.slot_target[idx].remain += charged

                if self.slot_charge_remain_min[idx] <= 0 or self.slot_charge_remain_kwh[idx] <= 0:
                    self.slot_target[idx].finish_charging()
                    self.release_slot(idx)

    def reset(self):
        for i in range(self.num_slots):
            self.slot_states[i] = SlotState.AVAILABLE
            self.slot_target[i] = None
            self.slot_iev_id[i] = -1
            self.slot_mcs_id[i] = -1
            self.slot_charge_remain_kwh[i] = 0.0
            self.slot_charge_remain_min[i] = 0.0

        self.total_idle_time_min = 0.0
        self.total_charged_kwh = 0.0
        self.total_profit = 0.0

        self.near_quasi = []
        self.near_iev = []
        self.near_idle_mcs = []
        self.near_task_mcs = []
        self.near_available_fcs = []
        self.near_busy_fcs = []
