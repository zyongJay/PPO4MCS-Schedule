"""Neural networks for option-aware MCS-only MAPPO/PPO training.

High actions use the fixed indices 0=Serve, 1=Recharge.  Waiting is a fixed
Low candidate representing the MCS current position.  Both actor
levels apply boolean masks directly to logits; True means the action/candidate
was physically valid at sampling time.  IEV movement is environment-controlled
and has no trainable actor in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

TensorLike = Union[np.ndarray, torch.Tensor]
MCS_ACTION_NAMES = ("Serve", "Recharge")


def masked_categorical(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[Categorical, torch.Tensor]:
    """Create a categorical distribution after masking invalid logits.

    ``valid_mask`` must have the same shape as ``logits`` and uses True for a
    valid action.  At least one action must be valid in every batch row.
    """
    mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != logits shape {tuple(logits.shape)}"
        )
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("each categorical row must contain at least one valid action")
    dtype_min = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, dtype_min)
    return Categorical(logits=masked_logits), masked_logits


class MCSHighActor(nn.Module):
    """Local High Actor: normalized high state -> Serve/Recharge logits."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(MCS_ACTION_NAMES)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def distribution(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Categorical:
        logits = self.forward(state)
        distribution, _ = masked_categorical(logits, action_mask)
        return distribution


class MCSLowActor(nn.Module):
    """Conditional Low Actor that scores the padded quasi candidate set."""

    def __init__(self, self_dim: int, candidate_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        self_state: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = self_state.dim() == 1
        if unbatched:
            self_state = self_state.unsqueeze(0)
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(0)
        if self_state.shape[0] != candidates.shape[0]:
            raise ValueError("low self-state and candidate batch sizes differ")

        self_embedding = self.self_encoder(self_state)
        candidate_embedding = self.candidate_encoder(candidates)
        expanded_self = self_embedding.unsqueeze(1).expand(
            -1, candidates.shape[1], -1
        )
        logits = self.score_head(
            torch.cat((expanded_self, candidate_embedding), dim=-1)
        ).squeeze(-1)
        return logits.squeeze(0) if unbatched else logits

    def distribution(
        self,
        self_state: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Categorical:
        logits = self.forward(self_state, candidates)
        distribution, _ = masked_categorical(logits, candidate_mask)
        return distribution


class CentralizedCritic(nn.Module):
    """High Critic：全局状态与 High 局部状态拼接后的价值函数。"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class LowCentralizedCritic(nn.Module):
    """Low Critic：使用掩码候选集合池化，避免依赖固定 TopK 展平。

    每个候选先经过共享编码器，再对合法候选执行 masked mean/max
    pooling。这样以后扩大候选数量时，无需改变 Critic 输入宽度。
    """

    def __init__(
        self,
        global_state_dim: int,
        self_dim: int,
        candidate_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(global_state_dim + self_dim + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        global_state: torch.Tensor,
        self_state: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        if self_state.dim() == 1:
            self_state = self_state.unsqueeze(0)
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(0)
        if candidate_mask.dim() == 1:
            candidate_mask = candidate_mask.unsqueeze(0)

        mask = candidate_mask.to(device=candidates.device, dtype=torch.bool)
        if not torch.all(mask.any(dim=-1)):
            raise ValueError('Low Critic 的每个样本至少需要一个合法候选')
        encoded = self.candidate_encoder(candidates)
        expanded_mask = mask.unsqueeze(-1)
        count = expanded_mask.sum(dim=1).clamp_min(1).to(encoded.dtype)
        pooled_mean = (encoded * expanded_mask).sum(dim=1) / count
        dtype_min = torch.finfo(encoded.dtype).min
        pooled_max = encoded.masked_fill(~expanded_mask, dtype_min).max(dim=1).values
        value_input = torch.cat(
            (global_state, self_state, pooled_mean, pooled_max), dim=-1
        )
        return self.value_head(value_input).squeeze(-1)


class MCSMAPPOAgent:
    """MCS High/Low Actor、双 Critic 及完全独立的优化器。"""

    CHECKPOINT_VERSION = 3

    def __init__(
        self,
        high_state_dim: int,
        low_self_dim: int,
        low_candidate_dim: int,
        critic_state_dim: int,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        low_actor_lr: float | None = None,
        low_critic_lr: float | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.high_actor = MCSHighActor(high_state_dim, hidden_dim).to(self.device)
        self.low_actor = MCSLowActor(
            low_self_dim, low_candidate_dim, hidden_dim
        ).to(self.device)
        global_state_dim = critic_state_dim - high_state_dim
        if global_state_dim <= 0:
            raise ValueError('critic_state_dim 必须大于 high_state_dim')
        self.high_critic = CentralizedCritic(
            critic_state_dim, hidden_dim
        ).to(self.device)
        self.low_critic = LowCentralizedCritic(
            global_state_dim,
            low_self_dim,
            low_candidate_dim,
            hidden_dim,
        ).to(self.device)
        self.high_optimizer = torch.optim.Adam(
            self.high_actor.parameters(), lr=actor_lr
        )
        self.low_optimizer = torch.optim.Adam(
            self.low_actor.parameters(), lr=(low_actor_lr or actor_lr)
        )
        self.high_critic_optimizer = torch.optim.Adam(
            self.high_critic.parameters(), lr=critic_lr
        )
        self.low_critic_optimizer = torch.optim.Adam(
            self.low_critic.parameters(), lr=(low_critic_lr or critic_lr)
        )
        self.last_load_info: Dict = {}

    def train(self) -> None:
        self.high_actor.train()
        self.low_actor.train()
        self.high_critic.train()
        self.low_critic.train()

    def eval(self) -> None:
        self.high_actor.eval()
        self.low_actor.eval()
        self.high_critic.eval()
        self.low_critic.eval()

    def _tensor(self, value: TensorLike, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _load_high_actor_state_dict(self, state_dict: Dict) -> bool:
        """加载两动作High头，并兼容旧三动作Serve/Recharge/Wait权重。

        旧模型最后一层的前两行恰好对应 Serve/Recharge，因此只裁掉 Wait
        输出行；其余共享表征参数原样迁移。返回值表示是否发生了该迁移。
        """
        current_state = self.high_actor.state_dict()
        adapted_state = dict(state_dict)
        migrated_wait_head = False
        for name, target in current_state.items():
            source = adapted_state.get(name)
            if source is None or tuple(source.shape) == tuple(target.shape):
                continue
            if (
                source.ndim >= 1
                and source.shape[0] == 3
                and target.shape[0] == 2
                and tuple(source.shape[1:]) == tuple(target.shape[1:])
            ):
                adapted_state[name] = source[:2].clone()
                migrated_wait_head = True
                continue
            raise RuntimeError(
                f'High Actor参数{name}形状不兼容：'
                f'{tuple(source.shape)} -> {tuple(target.shape)}'
            )
        self.high_actor.load_state_dict(adapted_state)
        return migrated_wait_head

    @torch.no_grad()
    def get_high_values_batch(self, critic_states: TensorLike) -> np.ndarray:
        states = self._tensor(critic_states)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        return self.high_critic(states).detach().cpu().numpy()

    # 兼容旧推理/测试调用方；语义明确等同于 High Critic。
    get_values_batch = get_high_values_batch

    @torch.no_grad()
    def get_value(self, critic_state: TensorLike) -> float:
        return float(self.get_high_values_batch(critic_state)[0])

    @torch.no_grad()
    def get_low_values_batch(
        self,
        global_states: TensorLike,
        self_states: TensorLike,
        candidates: TensorLike,
        masks: TensorLike,
    ) -> np.ndarray:
        values = self.low_critic(
            self._tensor(global_states),
            self._tensor(self_states),
            self._tensor(candidates),
            self._tensor(masks, dtype=torch.bool),
        )
        return values.detach().cpu().numpy()

    @torch.no_grad()
    def select_high_actions_batch(
        self, observations: list[Dict]
    ) -> list[Dict]:
        """仅在 High option 边界批量采样 Serve/Recharge。"""
        if not observations:
            return []
        high_states = self._tensor(np.stack([
            observation['high_state'] for observation in observations
        ]))
        high_masks = self._tensor(np.stack([
            observation['high_action_mask'] for observation in observations
        ]), dtype=torch.bool)
        high_distribution = self.high_actor.distribution(
            high_states, high_masks
        )
        high_actions = high_distribution.sample()
        high_log_probs = high_distribution.log_prob(high_actions)

        action_indices = high_actions.detach().cpu().numpy().astype(int)
        high_log_prob_values = high_log_probs.detach().cpu().numpy()
        return [{
            'mode': MCS_ACTION_NAMES[action_index],
            'high_action': int(action_index),
            'high_log_prob': float(high_log_prob_values[index]),
        } for index, action_index in enumerate(action_indices)]

    @torch.no_grad()
    def select_low_actions_batch(
        self, observations: list[Dict]
    ) -> list[Dict]:
        """在已激活的 High Serve option 内批量采样 Low 子决策。"""
        if not observations:
            return []
        low_self_states = self._tensor(np.stack([
            observation['low_self_state'] for observation in observations
        ]))
        low_candidates = self._tensor(np.stack([
            observation['low_candidates'] for observation in observations
        ]))
        low_masks = self._tensor(np.stack([
            observation['low_candidate_mask'] for observation in observations
        ]), dtype=torch.bool)
        low_distribution = self.low_actor.distribution(
            low_self_states, low_candidates, low_masks
        )
        low_actions = low_distribution.sample()
        low_log_probs = low_distribution.log_prob(low_actions)
        action_values = low_actions.detach().cpu().numpy().astype(int)
        log_prob_values = low_log_probs.detach().cpu().numpy()
        return [{
            'low_action': int(action_values[index]),
            'low_log_prob': float(log_prob_values[index]),
        } for index in range(len(observations))]

    @torch.no_grad()
    def select_mcs_actions_batch(self, observations: list[Dict]) -> list[Dict]:
        """兼容旧调用方：同时采样 High 及 Serve 子集的 Low。"""
        results = self.select_high_actions_batch(observations)
        if not results:
            return []
        for item in results:
            item['low_action'] = -1
            item['low_log_prob'] = 0.0

        serve_indices = np.asarray([
            index for index, item in enumerate(results)
            if item['high_action'] == 0
        ], dtype=int)
        if serve_indices.size:
            low_results = self.select_low_actions_batch([
                observations[index] for index in serve_indices
            ])
            for batch_index, observation_index in enumerate(serve_indices):
                results[int(observation_index)].update(
                    low_results[batch_index]
                )
        return results

    @torch.no_grad()
    def select_mcs_action(self, observation: Dict) -> Dict:
        return self.select_mcs_actions_batch([observation])[0]

    def evaluate_high(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.high_actor.distribution(
            states.to(self.device), masks.to(self.device)
        )
        actions = actions.to(self.device)
        return distribution.log_prob(actions), distribution.entropy()

    def evaluate_low(
        self,
        self_states: torch.Tensor,
        candidates: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.low_actor.distribution(
            self_states.to(self.device),
            candidates.to(self.device),
            masks.to(self.device),
        )
        actions = actions.to(self.device)
        return distribution.log_prob(actions), distribution.entropy()

    def high_values(self, critic_states: torch.Tensor) -> torch.Tensor:
        return self.high_critic(critic_states.to(self.device))

    # 兼容旧训练辅助代码；新训练必须显式调用 high_values/low_values。
    values = high_values

    def low_values(
        self,
        global_states: torch.Tensor,
        self_states: torch.Tensor,
        candidates: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.low_critic(
            global_states.to(self.device),
            self_states.to(self.device),
            candidates.to(self.device),
            masks.to(self.device),
        )

    def save(self, path: Union[str, Path], metadata: Dict | None = None) -> None:
        checkpoint = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "high_action_names": MCS_ACTION_NAMES,
            "high_actor": self.high_actor.state_dict(),
            "low_actor": self.low_actor.state_dict(),
            "high_critic": self.high_critic.state_dict(),
            "low_critic": self.low_critic.state_dict(),
            "high_optimizer": self.high_optimizer.state_dict(),
            "low_optimizer": self.low_optimizer.state_dict(),
            "high_critic_optimizer": self.high_critic_optimizer.state_dict(),
            "low_critic_optimizer": self.low_critic_optimizer.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, Path(path))

    def load_high_branch(
        self,
        path: Union[str, Path],
        load_high_critic: bool = True,
        freeze: bool = True,
    ) -> Dict:
        """只迁移 High 分支，保持 Low Actor/Critic 为当前随机初始化。

        该入口用于 V6 的固定 High 消融训练。旧 V5 ``critic`` 与当前
        High Critic 结构兼容，因此可作为只读诊断价值函数加载；High 的
        optimizer 状态不会恢复，避免误以为后续仍会继续训练 High。
        """
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        migrated_wait_head = self._load_high_actor_state_dict(
            checkpoint['high_actor']
        )
        high_critic_loaded = False
        if load_high_critic:
            if 'high_critic' in checkpoint:
                self.high_critic.load_state_dict(checkpoint['high_critic'])
                high_critic_loaded = True
            elif 'critic' in checkpoint:
                self.high_critic.load_state_dict(checkpoint['critic'])
                high_critic_loaded = True
        if freeze:
            for parameter in self.high_actor.parameters():
                parameter.requires_grad_(False)
            for parameter in self.high_critic.parameters():
                parameter.requires_grad_(False)
            self.high_actor.eval()
            self.high_critic.eval()
        self.last_load_info = {
            'high_branch_only': True,
            'source_checkpoint_version': int(
                checkpoint.get('checkpoint_version', 1)
            ),
            'high_actor_loaded': True,
            'migrated_high_wait_head': migrated_wait_head,
            'high_critic_loaded': high_critic_loaded,
            'high_frozen': bool(freeze),
            'low_actor_initialized_fresh': True,
            'low_critic_initialized_fresh': True,
            'optimizers_loaded': False,
        }
        return checkpoint.get('metadata', {})

    def load(
        self,
        path: Union[str, Path],
        load_optimizers: bool = False,
    ) -> Dict:
        """加载新旧 checkpoint。

        V5 旧格式的 ``critic`` 仅迁移到 High Critic；Low Critic 保持新
        初始化。优化器默认不恢复，避免旧单 Critic optimizer 与新结构
        不兼容；只有当前V3两动作checkpoint才恢复optimizer。旧三动作High
        输出头会自动保留Serve/Recharge两行并裁掉Wait行。
        """
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        migrated_wait_head = self._load_high_actor_state_dict(
            checkpoint["high_actor"]
        )
        self.low_actor.load_state_dict(checkpoint["low_actor"])
        version = int(checkpoint.get('checkpoint_version', 1))
        migrated_from_v5 = version < self.CHECKPOINT_VERSION
        if 'high_critic' in checkpoint:
            self.high_critic.load_state_dict(checkpoint['high_critic'])
        elif 'critic' in checkpoint:
            self.high_critic.load_state_dict(checkpoint['critic'])
        else:
            raise KeyError('checkpoint 缺少 high_critic/critic 参数')
        if 'low_critic' in checkpoint:
            self.low_critic.load_state_dict(checkpoint['low_critic'])

        optimizers_loaded = False
        if load_optimizers and version >= self.CHECKPOINT_VERSION:
            self.high_optimizer.load_state_dict(checkpoint['high_optimizer'])
            self.low_optimizer.load_state_dict(checkpoint['low_optimizer'])
            self.high_critic_optimizer.load_state_dict(
                checkpoint['high_critic_optimizer']
            )
            self.low_critic_optimizer.load_state_dict(
                checkpoint['low_critic_optimizer']
            )
            optimizers_loaded = True
        self.last_load_info = {
            'checkpoint_version': version,
            'migrated_from_v5': migrated_from_v5,
            'low_critic_initialized_fresh': 'low_critic' not in checkpoint,
            'optimizers_loaded': optimizers_loaded,
            'migrated_high_wait_head': migrated_wait_head,
        }
        return checkpoint.get("metadata", {})


# Backward-compatible name for callers that previously imported MAPPOAgent.
MAPPOAgent = MCSMAPPOAgent
