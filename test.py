import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from GDQN.net import GDQNNet, MCSHeteroGNN, IEVHeteroGNN
from env.config import conf
from env.core import MCS
from env.environment import MultiAgentEnv
from env.world import World

# World 构造时固定种子，决定 MCS 初始位置（与 SEED_LIST 无关）。
WORLD_INIT_SEED = 2026
# 各轮 env.reset() 之前重设的种子，保证各轮 EV 初电与策略 RNG 可复现。

SEED_LIST = [5678, 9012, 3456, 7890, 1357, 2468, 8642, 9753, 2050, 2060, 7867, 89, 678, 8980, 239, 2344, 4899, 324, 4243, 4353, 435, 4325, 6452, 6843]

def _set_test_seed(seed):
    """
    固定测试阶段随机种子，保证环境初始化可复现（包括 MCS 初始位置）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _animate_episode_mcs_trajectory(episode_positions, save_path, title, tracked_mcs_ids=None):
    """
    将单个 episode 中指定 MCS 的位置随 step 变化做成 GIF 动画并保存。
    episode_positions: List[List[List[lon, lat]]], 形状约为 [steps, num_mcs, 2]
    """
    if not episode_positions:
        return

    pos_arr = np.array(episode_positions, dtype=float)
    if pos_arr.ndim != 3 or pos_arr.shape[-1] != 2:
        return

    steps, selected_mcs_count, _ = pos_arr.shape
    if tracked_mcs_ids is None:
        tracked_mcs_ids = list(range(selected_mcs_count))
    tracked_mcs_ids = tracked_mcs_ids[:selected_mcs_count]
    if not tracked_mcs_ids:
        return

    color_map = ['tab:blue', 'tab:orange', 'tab:green']

    # 固定坐标范围：与 is_in_area / TrafficNet 的成都市边界一致，便于跨策略对比
    min_lon, max_lon = 103.9787, 104.1631
    min_lat, max_lat = 30.5965, 30.7309

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.grid(alpha=0.25)

    lines = []
    points = []
    labels = []
    for i, mcs_id in enumerate(tracked_mcs_ids):
        color = color_map[i % len(color_map)]
        (line,) = ax.plot([], [], color=color, linewidth=2.0, alpha=0.8, label=f'MCS{mcs_id}')
        point = ax.scatter([], [], c=color, s=55, marker='o')
        text = ax.text(0.0, 0.0, '', fontsize=9, color=color)
        lines.append(line)
        points.append(point)
        labels.append(text)

    def _update(frame):
        for i, mcs_id in enumerate(tracked_mcs_ids):
            lons = pos_arr[:frame + 1, i, 0]
            lats = pos_arr[:frame + 1, i, 1]
            lines[i].set_data(lons, lats)
            points[i].set_offsets(np.array([[lons[-1], lats[-1]]]))
            labels[i].set_position((lons[-1], lats[-1]))
            labels[i].set_text(f'MCS{mcs_id}')
        ax.set_title(f'{title} | Step {frame + 1}/{steps}')
        return lines + points + labels

    anim = FuncAnimation(fig, _update, frames=steps, interval=160, blit=False, repeat=False)
    anim.save(save_path, writer=PillowWriter(fps=6))
    plt.close(fig)


def _load_dual_pretrained_gnn(device, hidden_dim):
    gnn = {
        'mcs': MCSHeteroGNN(hidden_channels=hidden_dim).to(device),
        'iev': IEVHeteroGNN(hidden_channels=hidden_dim).to(device),
    }
    mcs_ckpt = './GDQN/models/pretrained_mcs_gnn.pth'
    iev_ckpt = './GDQN/models/pretrained_iev_gnn.pth'
    gnn['mcs'].load_state_dict(torch.load(mcs_ckpt, map_location=device))
    gnn['iev'].load_state_dict(torch.load(iev_ckpt, map_location=device))
    for p in gnn['mcs'].parameters():
        p.requires_grad = False
    for p in gnn['iev'].parameters():
        p.requires_grad = False
    gnn['mcs'].eval()
    gnn['iev'].eval()
    return gnn


def test(conf, load_episode=100, test_episodes=10):
    """
    测试脚本
    :param conf: 配置文件
    :param load_episode: 需要加载的 DQN 模型的 Episode 轮数后缀
    :param test_episodes: 测试运行的轮数
    """
    conf = dict(conf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 用 WORLD_INIT_SEED 锁定 World 构造期间的随机流，
    # 这样 MCS 初始位置（World.__init__ 里的 np.random.uniform）对所有模型/所有轮都一致。
    _set_test_seed(WORLD_INIT_SEED)
    print(f"🔒 World 构造种子已固定: {WORLD_INIT_SEED}（决定 MCS 初始位置）")
    print(f"🔒 各轮使用的种子序列: {SEED_LIST[:test_episodes]}")

    # ==========================================
    # 1. 加载并冻结双塔预训练 GNN
    # ==========================================
    try:
        gnn = _load_dual_pretrained_gnn(device=device, hidden_dim=conf.get('HIDDEN_DIM', 32))
        print("✅ 成功加载双塔预训练 GNN 权重！")
    except FileNotFoundError:
        print("❌ 未找到双塔预训练权重（pretrained_mcs_gnn.pth / pretrained_iev_gnn.pth）！")
        return

    # ==========================================
    # 2. 初始化环境与网络
    # ==========================================
    world = World(conf, gnn=gnn, device=device)
    env = MultiAgentEnv(world)
    gdqn = GDQNNet(conf, gnn=gnn)

    # ==========================================
    # 3. 加载训练好的 DQN 适配器与解码器参数
    # ==========================================
    # 如果不是纯随机模式，则加载模型参数
    if not conf.get('MCS_RANDOM', False) or not conf.get('IEV_RANDOM', False):
        try:
            mcs_path = f"./models/mcs_net_ep{load_episode}.pkl"
            iev_path = f"./models/iev_net_ep{load_episode}.pkl"

            # 你的 GDQNNet 内部含有 mcs_eval_net 和 iev_eval_net
            gdqn.mcs_eval_net.load_state_dict(torch.load(mcs_path, map_location=device))
            gdqn.iev_eval_net.load_state_dict(torch.load(iev_path, map_location=device))

            # 开启 eval 模式 (锁定可能存在的 Dropout/BatchNorm)
            gdqn.mcs_eval_net.eval()
            gdqn.iev_eval_net.eval()
            print(f"✅ 成功加载 DQN 模型权重 (对应 Episode: {load_episode})！")
        except FileNotFoundError:
            print(f"❌ 找不到对应的 DQN 权重文件: {mcs_path} 或 {iev_path}！")
            return

    # ==========================================
    # 4. 设置经典 epsilon-greedy 测试策略
    # epsilon=0.0 代表 100% 选网络最大 Q 值；epsilon=1.0 代表纯随机
    # ==========================================
    mcs_epsilon = 1.0 if conf.get('MCS_RANDOM', False) else 0.0
    iev_epsilon = 1.0 if conf.get('IEV_RANDOM', False) else 0.0

    MAX_STEPS = conf.get('TEST_MAX_STEPS_PER_EPISODE', 100)

    # 全局指标记录器
    history_metrics = {
        'total_charged_prop': [],
        'avg_mcs_reward': [],
        'avg_iev_reward': [],
        'avg_wait_time': [],
        'avg_extra_dist': [],
        'total_profit': [],
        'avg_idle_time': [],
        'energy_efficiency': []
    }
    tracked_mcs_ids = [mcs_id for mcs_id in [0, 1, 2, 3, 4, 5] if mcs_id < int(conf.get('NUM_MCS', 0))]
    # 仅记录 mcs0/mcs1/mcs2 的坐标序列
    # 格式: [episode][step][tracked_mcs_idx] = [lon, lat]
    all_episode_mcs_positions = []

    print(f'\n🚀 开始模型测试 (模式: MCS epsilon {mcs_epsilon}, IEV epsilon {iev_epsilon})...')

    for i_episode in range(test_episodes):
        # 每轮在 env.reset() 之前重设固定种子，保证 EV 初始电量、动作 RNG 在跨模型对比时可复现。
        # 注意：MCS 初始位置已在 World 构造时一次性确定并缓存，此处重设种子不会改变它。
        episode_seed = int(SEED_LIST[i_episode % len(SEED_LIST)])
        _set_test_seed(episode_seed)
        print(f"\n🔒 Episode {i_episode + 1:02d} 种子已重设: {episode_seed}")

        obs_n = env.reset()
        decide_time = []
        episode_mcs_positions = []

        # 记录初始位置 (step 0 动作执行前)
        init_positions = []
        for mcs_id in tracked_mcs_ids:
            if mcs_id < len(env.world.MCSs):
                mcs = env.world.MCSs[mcs_id]
                init_positions.append([float(mcs.pos[0]), float(mcs.pos[1])])
        episode_mcs_positions.append(init_positions)

        for step in range(MAX_STEPS):
            start_time = time.perf_counter()
            action_n = []

            # 动作选择 (由于无需存经验，不再需要记录 action_idx_n)
            with torch.no_grad():  # 测试时彻底关闭所有梯度计算，极大加速
                for index, agent in enumerate(env.world.agents):
                    agent_obs = obs_n[index]
                    if isinstance(agent, MCS):
                        a_idx = gdqn.mcs_choose_action(agent_obs, mcs_epsilon)
                        pos = agent_obs['pos'][a_idx] if a_idx is not None else agent.pos
                        if a_idx is not None and 'target_ids' in agent_obs:
                            agent.last_target_quasi_id = agent_obs['target_ids'][a_idx]
                        action_n.append(pos)
                    else:
                        a_idx = gdqn.iev_choose_action(agent_obs, iev_epsilon)
                        if a_idx is not None:
                            pos = agent_obs['pos'][a_idx]
                            agent.last_target_task_id = agent_obs['target_ids'][a_idx]
                        else:
                            # 无可调度 task 时，IEV 采用兜底动作：沿轨迹移动；若已到轨迹末端则原地
                            if agent.track_index + 1 < len(agent.track):
                                next_pos = agent.track[agent.track_index + 1]
                                pos = [float(next_pos[0]), float(next_pos[1])]
                            else:
                                pos = agent.pos
                        action_n.append(pos)

            end_time = time.perf_counter()
            decide_time.append((end_time - start_time) * 1000)

            # 与环境交互
            new_obs_n, _, _ = env.step(action_n)
            obs_n = new_obs_n
            # 记录当前 step 结束后 mcs0/mcs1/mcs2 的坐标（长度与 steps 数一致）
            step_positions = []
            for mcs_id in tracked_mcs_ids:
                if mcs_id < len(env.world.MCSs):
                    mcs = env.world.MCSs[mcs_id]
                    step_positions.append([float(mcs.pos[0]), float(mcs.pos[1])])
            episode_mcs_positions.append(step_positions)

        all_episode_mcs_positions.append(episode_mcs_positions)

        # ==========================================
        # 测试指标统计 (与 train.py 完全一致)
        # ==========================================
        profit, mcs_reward_sum, idleTime, cost = 0, 0, 0, 0
        num, failNum, waitTime, iev_reward_sum, extraDist, expense = 0, 0, 0, 0, 0, 0

        for mcs in env.world.MCSs:
            profit += (mcs.total_profit - mcs.cost)
            mcs_reward_sum += mcs.total_reward
            idleTime += mcs.total_idle_time
            cost += mcs.cost

        raw_mcs_cost = cost

        profit += cost
        cost = ((profit + cost) / 1.6) * 0.5 + cost
        profit -= cost

        normal_num = 0
        for ev in env.world.EVs:
            if ev.need_power == 0 and not ev.fail_charge:
                normal_num += 1
                continue
            if ev.is_charged:
                num += 1
                iev_reward_sum += ev.total_reward
                waitTime += ev.total_wait_time
                extraDist += ev.total_extra_dist
                expense += ev.expense
            if ev.fail_charge:
                failNum += 1

        # 计算全局指标
        charged_prop = num / (num + failNum) if (num + failNum) > 0 else 0
        avg_wait_time = waitTime / num if num > 0 else 0
        avg_extra_dist = extraDist / num if num > 0 else 0
        avg_idle_time = idleTime / float(conf['NUM_MCS'])
        avg_mcs_reward = mcs_reward_sum / (idleTime / 5) if idleTime > 0 else 0
        avg_iev_reward = iev_reward_sum / (waitTime / 5) if waitTime > 0 else 0

        charge_price = conf.get('CHARGE_PRICE', 1.0)
        mcs_power_price = conf.get('MCS_CHARGE_PRICE', 1.0)

        total_charged_energy = expense / charge_price if charge_price > 0 else 0
        total_consumed_energy = raw_mcs_cost / mcs_power_price if mcs_power_price > 0 else 0

        energy_efficiency = total_charged_energy / (total_charged_energy + total_consumed_energy) if (
                                                                                                             total_charged_energy + total_consumed_energy) > 0 else 0

        # 保存至 history
        history_metrics['total_charged_prop'].append(charged_prop)
        history_metrics['avg_mcs_reward'].append(avg_mcs_reward)
        history_metrics['avg_iev_reward'].append(avg_iev_reward)
        history_metrics['avg_wait_time'].append(avg_wait_time)
        history_metrics['avg_extra_dist'].append(avg_extra_dist)
        history_metrics['total_profit'].append(profit)
        history_metrics['avg_idle_time'].append(avg_idle_time)
        history_metrics['energy_efficiency'].append(energy_efficiency)

        # 终端输出 (移除 Loss 打印)
        print(f"\n{'=' * 15} Test Episode {i_episode + 1:02d} 结束 {'=' * 15}")
        print(f"[测试监控] 决策耗时: {np.mean(decide_time):.2f} ms")
        print(f"  ├─ MCS Avg Reward: {avg_mcs_reward:.2f}")
        print(f"  └─ IEV Avg Reward: {avg_iev_reward:.2f}")

        print(f"[物理业务指标]")
        print(f"  ├─ [服务情况] 成功数: {num} | 失败数: {failNum} | 充电成功率: {charged_prop * 100:.2f}%")
        print(f"  ├─ [IEV 体验] 均等待时间: {avg_wait_time:.2f} min | 均绕行距离: {avg_extra_dist:.2f} km")
        print(f"  ├─ [MCS 运营] 总空闲时间: {avg_idle_time:.2f} min | 总移动成本: {cost:.2f} | 总利润: {profit:.2f}")
        print(
            f"  └─ [全局能效] 总充电量: {total_charged_energy:.2f} | 总耗电量: {total_consumed_energy:.2f} | 电能利用率: {energy_efficiency * 100:.2f}%")
        print(f"{'=' * 48}")

    # ==========================================
    # 测试结束：保存测试数据的 CSV 文件
    # ==========================================
    os.makedirs('./results', exist_ok=True)

    # 动态命名，区分是 RANDOM 策略还是 DQN 策略的测试结果
    num_ev = conf.get('NUM_EV', 0)
    # prefix = "RANDOM" if (conf.get('MCS_RANDOM', False) and conf.get('IEV_RANDOM', False)) else f"DQN_ep{load_episode}"
    
    if conf.get('MCS_RANDOM', False) and conf.get('IEV_RANDOM', False):
        prefix = "RANDOM"
    elif not conf.get('MCS_RANDOM', False) and conf.get('IEV_RANDOM', False):
        prefix = "MCS_ONLY"
    elif conf.get('MCS_RANDOM', False) and not conf.get('IEV_RANDOM', False):
        prefix = "IEV_ONLY"
    else:
        prefix = f"DQN_ep{load_episode}"
    
    csv_path = f'./results/test_metrics_{prefix}_EV{num_ev}.csv'
    # traj_json_path = f'./results/test_mcs_positions_{prefix}.json'

    df_metrics = pd.DataFrame(history_metrics)
    df_metrics.index.name = 'test_episode'

    # 增加一行计算各项指标的平均值 (测试通常需要看多次跑的平均结果)
    df_metrics.loc['mean'] = df_metrics.mean()

    df_metrics.to_csv(csv_path)
    print(f"\n✅ 测试完全结束！测试指标（含均值）已保存至：{csv_path}")

    # 保存轨迹原始数据（step 顺序）
    # traj_payload = [
    #     {'episode': ep_idx, 'tracked_mcs_ids': tracked_mcs_ids, 'mcs_positions': ep_positions}
    #     for ep_idx, ep_positions in enumerate(all_episode_mcs_positions)
    # ]
    # with open(traj_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(traj_payload, f, ensure_ascii=False, indent=2)
    # print(f"✅ MCS 逐 step 坐标序列已保存至：{traj_json_path}")

    # 为每个测试 episode 绘制轨迹动画
    for ep_idx, ep_positions in enumerate(all_episode_mcs_positions):
        fig_path = f'./results/mcs_trajectory_{prefix}_ep{ep_idx + 1:02d}.gif'
        _animate_episode_mcs_trajectory(
            episode_positions=ep_positions,
            save_path=fig_path,
            title=f'MCS Trajectory Animation - {prefix} - Episode {ep_idx + 1:02d}',
            tracked_mcs_ids=tracked_mcs_ids
        )
        print(f"✅ 轨迹动画已保存：{fig_path}")


if __name__ == '__main__':

    # conf['ABLATION_NO_GNN'] = False
    # conf['MCS_RANDOM'] = True
    # conf['IEV_RANDOM'] = True
    test(conf, load_episode=100, test_episodes=1)