"""
demo/config.py — 小规模演示场景配置 (简化版)

坐标系: 二维欧氏平面 (米)
时间单位: 分钟
"""

# ============================
# 1. 空间范围 (矩形区域, 单位: 米)
# ============================
# 坐标格式
AREA_WIDTH = 500.0
AREA_HEIGHT = 500.0
AREA_X_MIN = 0.0
AREA_X_MAX = AREA_WIDTH
AREA_Y_MIN = 0.0
AREA_Y_MAX = AREA_HEIGHT

#经纬度格式
AREA_LON_MIN = 103.9787
AREA_LON_MAX = 104.1631
AREA_LAT_MIN = 30.5965
AREA_LAT_MAX = 30.7309

# ============================
# 2. 实体数量
# ============================
NUM_EV = 300
NUM_MCS = 10
NUM_FCS = 5
FCS_SLOTS_PER_STATION = 3

# ============================
# 3. 物理参数
# ============================
MOVE_SPEED = 11  # 移动速度（m/s）
CHARGE_SPEED = 120.0  # 充电功率 (kWh/h)
POWER_UNIT = 0.3  # 单位距离能耗 (kWh/km)

CHARGE_SPEED_PER_MIN = CHARGE_SPEED / 60.0  # 2 kWh/min

# ============================
# 4. 电池参数 (简化)
# ============================
EV_BATTERY_CAPACITY = 60.0
EV_LOW_POWER_THRESHOLD = 10.0  # 低于此值 → IEV
EV_LOWEST_POWER = 1.0  # 最低电量，低于此值判定充电失败

MCS_BATTERY_CAPACITY = 300.0  # kwh
MCS_RECHARGE_THRESHOLD = 40  # MCS低于此值必须补电

# ============================
# 5. 时间参数
# ============================
STEP_DURATION_MIN = 5
MAX_WAIT_TIME_STEPS = 4  # IEV最大等待步数
MAX_STEPS_PER_EPISODE = 200

# ============================
# 6. 充电参数
# ============================
MAX_CHARGE_PER_SESSION_KWH = CHARGE_SPEED_PER_MIN * STEP_DURATION_MIN * MAX_WAIT_TIME_STEPS
MAX_RECHARGE_PER_SESSION_KWH = MAX_CHARGE_PER_SESSION_KWH  # kwh
MAX_CHARGE_TIME_MIN = 20.0

# ============================
# 7. 匹配与通信参数
# ============================
COMM_RANGE = 3.0  # 通信范围 单位Km

# ============================
# 8. 价格参数
# ============================
CHARGE_PRICE = 1.6
RC_PRICE = 0.8
PG_PRICE = 0.5

# ============================
# 9. 候选数量
# ============================
TOP_K_MCS_CANDIDATES = 5
TOP_K_IEV_CANDIDATES = 5

# ============================
# 10. 奖励权重 (预留)
# ============================
REWARD_SCALE = 0.01
W_CHARGE = 3.0
W_MOVE = 1.0

MCS_FEAT_DIM_tgt = 5
MCS_FEAT_DIM_self = 3
MCS_HIGH_FEAT_DIM = 7
MCS_GLOBAL_STATE_DIM = 20
MCS_CRITIC_STATE_DIM = MCS_GLOBAL_STATE_DIM + MCS_HIGH_FEAT_DIM
EV_FEAT_DIM_tgt = 6
EV_FEAT_DIM_self = 6
FCS_FEAT_DIM = 7
HIDDEN_DIM = 32

EV_POWER_MEAN = 40
EV_POWER_STD = 16

DATA = 3
TRACK_DATA_PATH = '../data/track/2014080' + str(DATA) + '.csv'

MAX_MOVE_PER_STEP = MOVE_SPEED * 60 * STEP_DURATION_MIN  # m
