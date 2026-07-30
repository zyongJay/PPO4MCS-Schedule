这些指标的定义直接来自代码中的 PPO 更新逻辑，下面逐一解释：

### 1. MCS High Actor Loss（高层策略损失）
对应代码：`train.py` 中 High Actor 更新部分

High Actor 是一个二分类策略，输入 MCS 自身状态 + 系统统计信息，输出 Serve / Recharge 两个模式的概率。

计算过程：

$$r_{\mathrm{mode}} = \frac{\pi_{\theta}^{\mathrm{new}}(\mathrm{mode} \mid \mathrm{obs})}{\pi_{\theta}^{\mathrm{old}}(\mathrm{mode} \mid \mathrm{obs})}$$

$$L_{\mathrm{high}} = -\mathbb{E}\left[\min\left(r_{\mathrm{mode}} \cdot A_t,\; \mathrm{clip}(r_{\mathrm{mode}}, 1-\varepsilon, 1+\varepsilon) \cdot A_t\right)\right]$$

这里的负号是因为我们要最大化 PPO clipped surrogate objective，而 PyTorch 的 optimizer 做的是最小化 loss。所以 $L_{\mathrm{high}} < 0$ 表示 PPO 目标函数 $> 0$，即当前 advantage 是正的，策略正在向更好的方向更新。

- $L_{\mathrm{high}} \approx 0$：ratio ≈ 1，策略没有明显变化
- $L_{\mathrm{high}} \ll 0$（负值很大）：advantage 为正且策略在提升
- $L_{\mathrm{high}} > 0$：advantage 为负，策略在该样本上被惩罚

### 2. MCS Low Actor Loss（低层目标选择损失）
对应代码：`train.py` 中 Low Actor 更新部分

Low Actor 仅在 mode=Serve 时生效，负责从候选 quasi EV 中选择跟踪目标。采用动态动作空间：每个候选 EV 经过共享 MLP 计算一个 scalar score，再对所有候选做 softmax。

计算过程：

$$r_{\mathrm{target}} = \frac{\pi_{\psi}^{\mathrm{new}}(\mathrm{target} \mid \mathrm{obs}, \mathrm{mode}=\mathrm{Serve})}{\pi_{\psi}^{\mathrm{old}}(\mathrm{target} \mid \mathrm{obs}, \mathrm{mode}=\mathrm{Serve})}$$

$$L_{\mathrm{low}} = -\mathbb{E}\left[\min\left(r_{\mathrm{target}} \cdot A_t,\; \mathrm{clip}(r_{\mathrm{target}}, 1-\varepsilon, 1+\varepsilon) \cdot A_t\right)\right]$$

核心思想和高层一样，都是 PPO-clip。候选数量是动态的（每个 MCS 附近 quasi EV 数量不同），但每个候选独立计算 score → softmax → categorical 采样。

### 3. IEV Actor Loss（IEV 策略损失）
对应代码：`train.py` 中 IEV Actor 更新部分

IEV Actor 为未匹配成功的 IEV 选择等待目标（task MCS 或 occupied FCS）。结构与 Low Actor 完全相同（动态动作空间 + 逐候选打分 + softmax）。

计算过程：

$$r_{\mathrm{iev}} = \frac{\pi_{\phi}^{\mathrm{new}}(\mathrm{target} \mid \mathrm{obs})}{\pi_{\phi}^{\mathrm{old}}(\mathrm{target} \mid \mathrm{obs})}$$

$$L_{\mathrm{iev}} = -\mathbb{E}\left[\min\left(r_{\mathrm{iev}} \cdot A_t,\; \mathrm{clip}(r_{\mathrm{iev}}, 1-\varepsilon, 1+\varepsilon) \cdot A_t\right)\right]$$

### 4. Value Loss（Critic 价值损失）
对应代码：`train.py` Critic 更新部分

Centralized Critic 输入全局系统状态 $s$（17维：各类型 EV/MCS/FCS 统计量），输出标量 $V(s)$，用于 GAE 计算 advantage。损失是标准的 MSE：

$$L_{\mathrm{value}} = \mathbb{E}\left[(V(s_t) - R_t)^2\right]$$

其中 $R_t = A_t^{\mathrm{GAE}} + V(s_t)$ 是 GAE 计算出的 return。

GAE 的计算（在 `train.py` TrajectoryBuffer.compute_gae() 中）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$A_t = \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

$$R_t = A_t + V(s_t)$$

训练中 $L_{\mathrm{value}}$ 偏高（2-10），说明 Critic 难以精确拟合 return。原因包括：

- 环境随机性强（每 episode 初始条件不同）
- global state 特征可能不够充分
- return 本身方差大（reward 范围 -0.5 ~ +0.5，累积后波动大）

### 5. Entropy（策略熵）
对应代码：`train.py` 各 Actor 更新中 `ent.mean()` 部分

Entropy 是所有三个 Actor（High、Low、IEV）输出分布的平均信息熵：

$$H = -\sum_{a} \pi(a \mid \math{obs}) \cdot \log \pi(a \mid \mathrm{obs})$$

对于 High Actor：$H_{\math{high}} = -[p_{\mathrm{Serve}}\log p_{\mathrm{Serve}} + p_{\mathrm{Recharge}}\log p_{\mathrm{Recharge}}]$

对于 Low/IEV Actor（$N$ 个候选）：$H = -\sum_{i=1}^{N} p_i \log p_i$

Entropy 被加入 total loss 作为探索奖励（代码中的 `-entropy_coef * ent`），即鼓励策略保持一定的随机性，防止过早收敛到次优策略。

训练中 entropy 从 ~1.1 衰减到 ~0.4，说明策略逐渐从随机探索走向确定性决策，这是正常现象。但如果降到过低（<0.1），说明策略过于贪婪，可能需要增大 `entropy_coef`。

### 总 Loss 公式

$$L_{\mathrm{total}} = -L_{\mathrm{high}} - \lambda_1 L_{\mathrm{low}} - \lambda_2 L_{\mathrm{iev}} + c \cdot L_{\mathrm{value}}$$

其中 $\lambda_1 = \lambda_2 = 0.5$，$c = 0.5$（`value_coef`），entropy bonus 已内嵌在各 $L$ 的 `-entropy_coef * H` 项中。