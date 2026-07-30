import time
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from GDQN.net import GDQNNet, MCSHeteroGNN, IEVHeteroGNN
from env.config import conf
from env.core import MCS
from env.environment import MultiAgentEnv
from env.world import World


def _load_dual_pretrained_gnn(device, hidden_dim):
    gnn = {
        'mcs': MCSHeteroGNN(hidden_channels=hidden_dim).to(device),
        'iev': IEVHeteroGNN(hidden_channels=hidden_dim).to(device),
    }

    mcs_ckpt = './GDQN/models/pretrained_mcs_gnn.pth'
    iev_ckpt = './GDQN/models/pretrained_iev_gnn.pth'

    try:
        gnn['mcs'].load_state_dict(torch.load(mcs_ckpt, map_location=device))
        print(f"✅ 成功加载 MCS 预训练权重: {mcs_ckpt}")
    except FileNotFoundError:
        print(f"⚠️ 未找到 MCS 预训练权重: {mcs_ckpt}，将使用随机初始化。")

    try:
        gnn['iev'].load_state_dict(torch.load(iev_ckpt, map_location=device))
        print(f"✅ 成功加载 IEV 预训练权重: {iev_ckpt}")
    except FileNotFoundError:
        print(f"⚠️ 未找到 IEV 预训练权重: {iev_ckpt}，将使用随机初始化。")

    return gnn


def _set_partial_trainable(model, unfreeze_last_conv_layers=1):
    for p in model.parameters():
        p.requires_grad = False

    if unfreeze_last_conv_layers <= 0:
        model.eval()
        return 0

    start_idx = max(0, len(model.convs) - unfreeze_last_conv_layers)
    for i in range(start_idx, len(model.convs)):
        for p in model.convs[i].parameters():
            p.requires_grad = True
        for p in model.norms[i].parameters():
            p.requires_grad = True

    model.train()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 2. 训练主循环
# ==========================================
def train(conf):
    conf = dict(conf)
    conf.setdefault('EPS_START', 0.20)
    conf.setdefault('EPS_END', 0.01)
    conf.setdefault('EPS_DECAY_EPISODES', 40)

    # 初始化环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 加载双塔预训练 GNN，并完全冻结（只训练 DQN）
    # ==========================================
    hidden_dim = conf.get('HIDDEN_DIM', 32)
    gnn = _load_dual_pretrained_gnn(device=device, hidden_dim=hidden_dim)
    trainable_mcs = _set_partial_trainable(gnn['mcs'], unfreeze_last_conv_layers=0)
    trainable_iev = _set_partial_trainable(gnn['iev'], unfreeze_last_conv_layers=0)
    print(f"🔒 GNN 全冻结完成 | MCS trainable params: {trainable_mcs} | IEV trainable params: {trainable_iev}")

    # 初始化环境与网络
    world = World(conf, gnn=gnn, device=device)
    env = MultiAgentEnv(world)
    gdqn = GDQNNet(conf, gnn=gnn)

    # 训练超参数
    MAX_EPISODES = conf.get('NUM_EPISODES', 60)
    MAX_STEPS = conf.get('MAX_STEPS_PER_EPISODE', 100)  # 每轮仿真的时间步数

    # ==========================================
    # 全局指标记录器 (用于最终保存画图)
    # ==========================================
    history_metrics = {
        'total_charged_prop': [],
        'avg_mcs_reward': [],
        'avg_iev_reward': [],
        'avg_wait_time': [],
        'avg_extra_dist': [],
        'total_profit': [],
        'avg_idle_time': [],
        'energy_efficiency': [],  # 电能利用率
        'mcs_epsilon': [],
        'iev_epsilon': [],
        'avg_decide_ms': [],
        'mcs_learn_updates': [],
        'iev_learn_updates': [],
        'mcs_replay_size': [],
        'iev_replay_size': [],
        'avg_transitions_per_step': [],
        'avg_gnn_encode_ms_per_step': [],
        'avg_dqn_train_ms_per_step': [],
    }

    print('\n🚀 开启 冻结GNN特征 + DQN训练 (epsilon-greedy + replay batch learn) ...')

    for i_episode in range(MAX_EPISODES):
        if conf.get('MCS_RANDOM', False):
            mcs_epsilon = 1.0
        else:
            ratio = min(1.0, i_episode / max(1, conf['EPS_DECAY_EPISODES']))
            mcs_epsilon = conf['EPS_START'] + (conf['EPS_END'] - conf['EPS_START']) * ratio
        if conf.get('IEV_RANDOM', False):
            iev_epsilon = 1.0
        else:
            ratio = min(1.0, i_episode / max(1, conf['EPS_DECAY_EPISODES']))
            iev_epsilon = conf['EPS_START'] + (conf['EPS_END'] - conf['EPS_START']) * ratio

        obs_n = env.reset()  # 初始状态：S_0

        ep_mcs_loss = []
        ep_iev_loss = []
        decide_time = []
        ep_transition_count = 0
        ep_mcs_updates = 0
        ep_iev_updates = 0
        ep_gnn_encode_ms = 0.0
        ep_dqn_train_ms = 0.0

        for step in range(MAX_STEPS):
            start_time = time.perf_counter()

            action_n = []
            action_idx_n = []

            # 动作选择
            for index, agent in enumerate(env.world.agents):
                agent_obs = obs_n[index]  # 获取当前状态S_t
                if isinstance(agent, MCS):
                    a_idx = gdqn.mcs_choose_action(agent_obs, mcs_epsilon)  # action_list中的下标
                    pos = agent_obs['pos'][a_idx] if a_idx is not None else agent.pos  # 调度位置坐标
                    # 记录目标 quasi ID，供 credit assignment 使用
                    if a_idx is not None and 'target_ids' in agent_obs:
                        agent.last_target_quasi_id = agent_obs['target_ids'][a_idx]
                    action_n.append(pos)
                    action_idx_n.append(a_idx)
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
                    action_idx_n.append(a_idx)

            end_time = time.perf_counter()
            decide_time.append((end_time - start_time) * 1000)

            # 与环境交互（拆分计时：单独统计 GNN 编码耗时）
            env.world.update(action_n)
            env.world.step_finish()
            env.world.match_and_get_neibor()
            gnn_t0 = time.perf_counter()
            new_obs_n, old_obs_n = env.world.get_obs_n()
            gnn_t1 = time.perf_counter()
            reward_n = env.world.mix_get_reward_n()
            ep_gnn_encode_ms += (gnn_t1 - gnn_t0) * 1000.0

            # 经验回放：写入 transition
            for index, agent in enumerate(env.world.last_agents):
                if action_idx_n[index] is not None:
                    state = obs_n[index]  # s_t
                    action = action_idx_n[index]  # action
                    reward = reward_n[index]  # r_t
                    next_state = old_obs_n[index]  # s_t+1

                    # 将局部 done 和 全局 done 结合
                    is_ep_done = (step == MAX_STEPS - 1)  # 全局回合截断
                    agent_done = next_state.get('done', False) or is_ep_done

                    gdqn.push_transition(
                        is_mcs=isinstance(agent, MCS),
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=agent_done
                    )
                    ep_transition_count += 1

            # Batch 学习（按 warmup + batch_size + learn_every 触发）
            dqn_t0 = time.perf_counter()
            mcs_loss, iev_loss = gdqn.step_and_learn()
            dqn_t1 = time.perf_counter()
            ep_dqn_train_ms += (dqn_t1 - dqn_t0) * 1000.0
            if mcs_loss is not None and not conf.get('MCS_RANDOM', False):
                ep_mcs_loss.append(mcs_loss)
                ep_mcs_updates += 1
            if iev_loss is not None and not conf.get('IEV_RANDOM', False):
                ep_iev_loss.append(iev_loss)
                ep_iev_updates += 1

            # 更新状态，准备进入下一个step
            obs_n = new_obs_n

        # 统计数据
        profit, mcs_reward_sum, idleTime, cost = 0, 0, 0, 0
        num, failNum, waitTime, iev_reward_sum, extraDist, expense = 0, 0, 0, 0, 0, 0

        for mcs in env.world.MCSs:
            profit += (mcs.total_profit - mcs.cost)
            mcs_reward_sum += mcs.total_reward
            idleTime += mcs.total_idle_time
            cost += mcs.cost

        # ✨ 核心插入：在 cost 被魔改计算之前，截获真实的物理移动成本
        raw_mcs_cost = cost

        profit += cost
        cost = ((profit + cost) / 1.6) * 0.5 + cost
        profit -= cost

        normal_num = 0

        for ev in env.world.EVs:
            if ev.need_power == 0 and not ev.fail_charge:
                normal_num += 1
                continue
            iev_reward_sum += ev.total_reward
            if ev.is_charged:
                num += 1
                waitTime += ev.total_wait_time  # min
                extraDist += ev.total_extra_dist  # km
                expense += ev.expense  # 累计收取的充电费
            if ev.fail_charge:
                failNum += 1

        # ==========================================
        # 全局指标与电能利用率计算
        # ==========================================
        charged_prop = num / (num + failNum) if (num + failNum) > 0 else 0
        avg_wait_time = waitTime / num if num > 0 else 0
        avg_extra_dist = extraDist / num if num > 0 else 0
        avg_idle_time = idleTime / float(conf['NUM_MCS'])
        avg_mcs_reward = mcs_reward_sum / conf['NUM_MCS'] if conf['NUM_MCS'] > 0 else 0
        avg_iev_reward = iev_reward_sum / (conf['NUM_EV'] - normal_num) if conf['NUM_EV'] > normal_num else 0

        # 根据 world.py 的物理设定，通过费用和单价反推出真实的物理电量
        charge_price = conf.get('CHARGE_PRICE', 1.0)
        mcs_power_price = conf.get('MCS_CHARGE_PRICE', 1.0)

        total_charged_energy = expense / charge_price if charge_price > 0 else 0
        total_consumed_energy = raw_mcs_cost / mcs_power_price if mcs_power_price > 0 else 0

        # 计算核心指标：电能利用率 = 总充电量 / (总充电量 + 总耗电量)
        energy_efficiency = total_charged_energy / (total_charged_energy + total_consumed_energy) if (
                                                                                                                 total_charged_energy + total_consumed_energy) > 0 else 0

        # ==========================================
        # 将当前指标存入 History 字典
        # ==========================================
        history_metrics['total_charged_prop'].append(charged_prop)
        history_metrics['avg_mcs_reward'].append(avg_mcs_reward)
        history_metrics['avg_iev_reward'].append(avg_iev_reward)
        history_metrics['avg_wait_time'].append(avg_wait_time)
        history_metrics['avg_extra_dist'].append(avg_extra_dist)
        history_metrics['total_profit'].append(profit)
        history_metrics['avg_idle_time'].append(avg_idle_time)
        history_metrics['energy_efficiency'].append(energy_efficiency)
        history_metrics['mcs_epsilon'].append(mcs_epsilon)
        history_metrics['iev_epsilon'].append(iev_epsilon)
        history_metrics['avg_decide_ms'].append(float(np.mean(decide_time)) if decide_time else 0.0)
        history_metrics['mcs_learn_updates'].append(ep_mcs_updates)
        history_metrics['iev_learn_updates'].append(ep_iev_updates)
        history_metrics['mcs_replay_size'].append(len(gdqn.mcs_replay))
        history_metrics['iev_replay_size'].append(len(gdqn.iev_replay))
        history_metrics['avg_transitions_per_step'].append(ep_transition_count / max(1, MAX_STEPS))
        history_metrics['avg_gnn_encode_ms_per_step'].append(ep_gnn_encode_ms / max(1, MAX_STEPS))
        history_metrics['avg_dqn_train_ms_per_step'].append(ep_dqn_train_ms / max(1, MAX_STEPS))

        # 终端输出
        print(f"\n{'=' * 15} Episode {i_episode:03d} 结束 {'=' * 15}")
        print(
            f"[RL 训练监控] 决策耗时: {np.mean(decide_time):.2f} ms | 探索率epsilon (MCS/IEV): {mcs_epsilon:.2f}/{iev_epsilon:.2f}")
        print(
            f"[训练样本监控] 回放池(MCS/IEV): {len(gdqn.mcs_replay)}/{len(gdqn.iev_replay)} | "
            f"每步样本数: {ep_transition_count / max(1, MAX_STEPS):.2f} | "
            f"更新次数(MCS/IEV): {ep_mcs_updates}/{ep_iev_updates}"
        )
        print(
            f"[耗时监控] GNN编码总耗时: {ep_gnn_encode_ms:.2f} ms | "
            f"DQN训练总耗时: {ep_dqn_train_ms:.2f} ms | "
            f"每步均值(GNN/DQN): {ep_gnn_encode_ms / max(1, MAX_STEPS):.2f}/{ep_dqn_train_ms / max(1, MAX_STEPS):.2f} ms"
        )
        if ep_mcs_loss:
            print(f"  ├─ MCS Agent -> Avg Loss: {np.mean(ep_mcs_loss):.4f} | Avg Reward: {avg_mcs_reward:.2f}")
        if ep_iev_loss:
            print(f"  └─ IEV Agent -> Avg Loss: {np.mean(ep_iev_loss):.4f} | Avg Reward: {avg_iev_reward:.2f}")

        print(f"[物理业务指标]")
        print(f"  ├─ [服务情况] 成功数: {num} | 失败数: {failNum} | 充电成功率: {charged_prop * 100:.2f}%")
        print(f"  ├─ [IEV 体验] 均等待时间: {avg_wait_time:.2f} min | 均绕行距离: {avg_extra_dist:.2f} km")
        print(f"  ├─ [MCS 运营] 总空闲时间: {avg_idle_time:.2f} min | 总移动成本: {cost:.2f} | 总利润: {profit:.2f}")
        print(
            f"  └─ [全局能效] 总充电量: {total_charged_energy:.2f} | 总耗电量: {total_consumed_energy:.2f} | 电能利用率: {energy_efficiency * 100:.2f}%")
        print(f"{'=' * 48}")

        # 模型保存
        if i_episode % 10 == 0 and i_episode != 0:
            gdqn.save_model(i_episode, conf)

    # ==========================================
    # 训练结束：保存所有全局性指标到 CSV 文件
    # ==========================================
    os.makedirs('./results', exist_ok=True)
    csv_path = './results/training_global_metrics.csv'
    df_metrics = pd.DataFrame(history_metrics)
    df_metrics.index.name = 'episode'
    df_metrics.to_csv(csv_path)
    print(f"\n✅ 训练完全结束！全局核心指标已保存至：{csv_path}")

    # ==========================================
    # 训练结束：绘制 Reward / 充电率 / 能效 随 Episode 变化图
    # ==========================================
    episodes = range(len(history_metrics['total_charged_prop']))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) 充电成功率
    ax = axes[0, 0]
    ax.plot(episodes, history_metrics['total_charged_prop'], 'b-', linewidth=1.2)
    ax.set_title('Charge Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # 2) MCS / IEV 平均 Reward
    ax = axes[0, 1]
    ax.plot(episodes, history_metrics['avg_mcs_reward'], 'tab:orange', linewidth=1.2, label='MCS')
    ax.plot(episodes, history_metrics['avg_iev_reward'], 'tab:green', linewidth=1.2, label='IEV')
    ax.set_title('Avg Reward per Agent')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) 电能利用率
    ax = axes[1, 0]
    ax.plot(episodes, history_metrics['energy_efficiency'], 'tab:purple', linewidth=1.2)
    ax.set_title('Energy Efficiency')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Ratio')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # 4) 总利润
    ax = axes[1, 1]
    ax.plot(episodes, history_metrics['total_profit'], 'tab:red', linewidth=1.2)
    ax.set_title('Total Profit')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Profit')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = './results/training_curves.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"📈 训练曲线图已保存至：{fig_path}")


# 运行训练
if __name__ == '__main__':
    train(conf)
