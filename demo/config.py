"""
demo/config.py — 小规模演示场景配置 (简化版)

坐标系: 二维欧氏平面 (米)
时间单位: 分钟
"""

# ============================
# 1. 空间范围 (矩形区域, 单位: 米)
# ============================
AREA_WIDTH = 500.0
AREA_HEIGHT = 500.0
AREA_X_MIN = 0.0
AREA_X_MAX = AREA_WIDTH
AREA_Y_MIN = 0.0
AREA_Y_MAX = AREA_HEIGHT

# ============================
# 2. 实体数量
# ============================
NUM_EV = 5
NUM_MCS = 3
NUM_FCS = 2
FCS_SLOTS_PER_STATION = [2, 2]

# ============================
# 3. 物理参数
# ============================
MOVE_SPEED = 3              # 移动速度（m/s）
CHARGE_SPEED = 120.0        # 充电功率 (kWh/h)
POWER_UNIT = 0.3            # 单位距离能耗 (kWh/km)

CHARGE_SPEED_PER_MIN = CHARGE_SPEED / 60.0  # 2 kWh/min

# ============================
# 4. 电池参数 (简化)
# ============================
EV_BATTERY_CAPACITY = 60.0
EV_LOW_POWER_THRESHOLD = 10.0    # 低于此值 → IEV
EV_LOWEST_POWER = 1.0            # 最低电量，低于此值判定充电失败

MCS_BATTERY_CAPACITY = 300.0
MCS_RECHARGE_THRESHOLD = 40.0    # MCS低于此值必须补电

# ============================
# 5. 时间参数
# ============================
STEP_DURATION_MIN = 1.0
MAX_STEPS_PER_EPISODE = 200
MAX_WAIT_TIME_STEPS = 4         # IEV最大等待步数

# ============================
# 6. 充电参数
# ============================
MAX_CHARGE_PER_SESSION_KWH = CHARGE_SPEED_PER_MIN * STEP_DURATION_MIN * MAX_WAIT_TIME_STEPS
MAX_CHARGE_TIME_MIN = 20.0

# ============================
# 7. 匹配与通信参数
# ============================
COMM_RANGE = 2.0              # 通信范围 单位Km

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
EV_FEAT_DIM = 6
FCS_FEAT_DIM = 7  # IEV 候选目标统一为 7 维
HIDDEN_DIM = 32
