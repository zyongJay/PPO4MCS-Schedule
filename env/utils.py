import math

import numpy as np


def _clip01(x):
    # 数据截断在0-1范围内
    return float(np.clip(x, 0.0, 1.0))


def haversine(lon1, lat1, lon2, lat2):
    # distance between latitudes
    # and longitudes
    lon1 = float(lon1)
    lon2 = float(lon2)
    lat1 = float(lat1)
    lat2 = float(lat2)
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
    return rad * c


def min_max_norm(data):
    """归一化数据到[0,1]范围"""
    if not data:
        return data
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.0] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def normalize_pos(pos):
    """归一化位置坐标到[0,1]范围"""
    # 成都市区经纬度范围
    min_lon = 103.9787
    max_lon = 104.1631
    min_lat = 30.5965
    max_lat = 30.7309

    lon = (float(pos[0]) - min_lon) / (max_lon - min_lon)
    lat = (float(pos[1]) - min_lat) / (max_lat - min_lat)

    return [lon, lat]


def extract_vehicle_features(ev):
    """
    抽取车辆节点特征（IEV或quasi）。
    """
    need_power = _clip01(ev.need_power / (ev.conf["CHARGE_SPEED"] / 3))
    remain = _clip01(ev.remain / 100.0)
    norm_lon, norm_lat = normalize_pos(ev.pos)
    need_charge = 1.0 if ev.need_charge else 0.0
    # 等待时间按配置中的最大可等待step归一化。
    wait_time = _clip01(ev.wait_time / ev.conf.get('MAX_WAIT_TIME', 2))

    # 特征顺序：空间->能量->状态
    return np.array(
        [
            norm_lon,
            norm_lat,
            remain,
            need_power,
            need_charge,
            wait_time
        ],
        dtype=np.float32,
    )


def extract_mcs_features(mcs):
    """
    抽取MCS节点特征。
    """
    remain = _clip01(mcs.remain / 300.0)
    norm_lon, norm_lat = normalize_pos(mcs.pos)
    is_idle = 1.0 if mcs.is_idle else 0.0
    is_arrive = 1.0 if mcs.is_arrive else 0.0
    max_charge_power = max(1.0, float(mcs.conf.get('CHARGE_SPEED', 120)) / 3.0)
    charge_power = _clip01(mcs.charge_power / max_charge_power)
    charge_time = _clip01((mcs.charge_time or 0.0) / 20.0)

    # 特征顺序：空间->能量->状态
    return np.array(
        [
            norm_lon,
            norm_lat,
            remain,
            is_idle,
            is_arrive,
            charge_power,
            charge_time,
        ],
        dtype=np.float32,
    )
