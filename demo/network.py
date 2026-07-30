"""
demo/network.py — Option-inspired MAPPO 网络模块 (无GNN, 纯MLP)

包含:
  - MCSHighActor:      模式选择策略 π_mode(z|obs) → Serve/Recharge
  - MCSLowActor:       条件目标选择策略 π_target(a|obs, z=Serve) → quasi EV 候选
  - IEVActor:          IEV等待目标选择策略 → task MCS / occupied FCS 候选
  - CentralizedCritic: 中心化价值网络 V(global_state) → 标量

所有 Actor 均支持动态候选数量 (无padding), 逐候选计算 score → softmax。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Optional, Dict


# ============================================================
# MCS High Actor — 模式选择 (Serve / Recharge)
# ============================================================

class MCSHighActor(nn.Module):
    """MCS 高层策略: 输入自身状态+环境统计, 输出 Serve/Recharge 概率"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Serve, Recharge
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回 logits [batch, 2]"""
        return self.net(state)

    def get_action(self, state: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """
        采样动作并返回 (mode_idx, log_prob, logits).
        state: [state_dim] 或 [batch, state_dim]
        """
        logits = self.forward(state.unsqueeze(0) if state.dim() == 1 else state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, logits


# ============================================================
# MCS Low Actor — 条件目标选择 (仅 Serve 模式)
# ============================================================

class MCSLowActor(nn.Module):
    """
    MCS 低层策略: 在 Serve 模式下选择 quasi EV 跟踪目标。
    动态动作空间 — 逐候选编码 → score → softmax。
    """

    def __init__(self, self_dim: int, cand_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cand_encoder = nn.Sequential(
            nn.Linear(cand_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, self_state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        self_state: [self_dim]
        candidates: [N, cand_dim], N 可变
        返回: [N] scores (logits)
        """
        h_self = self.self_encoder(self_state.unsqueeze(0))  # [1, hidden_dim]
        h_cand = self.cand_encoder(candidates)               # [N, hidden_dim]
        h_self_exp = h_self.expand(len(candidates), -1)       # [N, hidden_dim]
        h_concat = torch.cat([h_self_exp, h_cand], dim=-1)   # [N, hidden_dim*2]
        scores = self.score_head(h_concat).squeeze(-1)        # [N]
        return scores

    def get_action(self, self_state: torch.Tensor, candidates: torch.Tensor
                   ) -> Tuple[Optional[int], Optional[float], Optional[torch.Tensor]]:
        """
        采样目标候选索引。
        若 candidates 为空, 返回 (None, None, None)。
        """
        if candidates.shape[0] == 0:
            return None, None, None
        scores = self.forward(self_state, candidates)
        probs = F.softmax(scores, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, scores


# ============================================================
# IEV Actor — 等待目标选择
# ============================================================

class IEVActor(nn.Module):
    """
    IEV 策略: 选择等待目标 (task MCS / occupied FCS)。
    动态动作空间 — 逐候选编码 → score → softmax。
    """

    def __init__(self, self_dim: int, cand_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cand_encoder = nn.Sequential(
            nn.Linear(cand_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, self_state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        h_self = self.self_encoder(self_state.unsqueeze(0))
        h_cand = self.cand_encoder(candidates)
        h_self_exp = h_self.expand(len(candidates), -1)
        h_concat = torch.cat([h_self_exp, h_cand], dim=-1)
        scores = self.score_head(h_concat).squeeze(-1)
        return scores

    def get_action(self, self_state: torch.Tensor, candidates: torch.Tensor
                   ) -> Tuple[Optional[int], Optional[float], Optional[torch.Tensor]]:
        if candidates.shape[0] == 0:
            return None, None, None
        scores = self.forward(self_state, candidates)
        probs = F.softmax(scores, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, scores


# ============================================================
# Centralized Critic — 中心化价值网络
# ============================================================

class CentralizedCritic(nn.Module):
    """CTDE 中心化 Critic: 输入全局状态, 输出标量 V(s)"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """global_state: [batch, state_dim] → [batch, 1]"""
        return self.net(global_state)


# ============================================================
# MAPPO Agent — 包装所有网络
# ============================================================

class MAPPOAgent:
    """MAPPO 智能体容器: 包含所有 Actor/Critic 网络及优化器"""

    def __init__(self,
                 mcs_high_dim: int, mcs_low_self_dim: int, mcs_low_cand_dim: int,
                 iev_self_dim: int, iev_cand_dim: int,
                 global_state_dim: int,
                 hidden_dim: int = 128,
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 device: str = 'cpu'):
        self.device = torch.device(device)

        # Actor 网络
        self.mcs_high_actor = MCSHighActor(mcs_high_dim, hidden_dim).to(self.device)
        self.mcs_low_actor = MCSLowActor(mcs_low_self_dim, mcs_low_cand_dim, hidden_dim).to(self.device)
        self.iev_actor = IEVActor(iev_self_dim, iev_cand_dim, hidden_dim).to(self.device)

        # Critic 网络
        self.critic = CentralizedCritic(global_state_dim, hidden_dim).to(self.device)

        # 优化器
        self.high_optimizer = torch.optim.Adam(self.mcs_high_actor.parameters(), lr=lr_actor)
        self.low_optimizer = torch.optim.Adam(self.mcs_low_actor.parameters(), lr=lr_actor)
        self.iev_optimizer = torch.optim.Adam(self.iev_actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.train()

    def train(self):
        self.mcs_high_actor.train()
        self.mcs_low_actor.train()
        self.iev_actor.train()
        self.critic.train()

    def eval(self):
        self.mcs_high_actor.eval()
        self.mcs_low_actor.eval()
        self.iev_actor.eval()
        self.critic.eval()

    def get_value(self, global_state: torch.Tensor) -> float:
        """获取全局状态的价值估计"""
        with torch.no_grad():
            if global_state.dim() == 1:
                global_state = global_state.unsqueeze(0)
            return self.critic(global_state.to(self.device)).item()

    def select_mcs_action(self, high_state: torch.Tensor,
                          low_self_state: torch.Tensor,
                          low_candidates: torch.Tensor
                          ) -> Dict:
        """
        MCS 动作选择 (两级决策)。

        Returns:
            {'mode': 'Serve'|'Recharge',
             'mode_idx': int,
             'log_prob_mode': float,
             'target_idx': int or None,
             'log_prob_target': float or None,
             'total_log_prob': float}
        """
        with torch.no_grad():
            mode_idx, log_prob_mode, _ = self.mcs_high_actor.get_action(
                high_state.to(self.device))
            mode = 'Serve' if mode_idx == 0 else 'Recharge'

            if mode == 'Serve':
                target_idx, log_prob_target, _ = self.mcs_low_actor.get_action(
                    low_self_state.to(self.device),
                    low_candidates.to(self.device))
                total_log_prob = log_prob_mode
                if log_prob_target is not None:
                    total_log_prob = total_log_prob + log_prob_target
            else:
                target_idx = None
                log_prob_target = None
                total_log_prob = log_prob_mode

        return {
            'mode': mode, 'mode_idx': mode_idx,
            'log_prob_mode': log_prob_mode,
            'target_idx': target_idx,
            'log_prob_target': log_prob_target,
            'total_log_prob': total_log_prob,
        }

    def select_iev_action(self, self_state: torch.Tensor,
                          candidates: torch.Tensor) -> Dict:
        """IEV 动作选择。"""
        with torch.no_grad():
            target_idx, log_prob_target, _ = self.iev_actor.get_action(
                self_state.to(self.device),
                candidates.to(self.device))
        return {
            'target_idx': target_idx,
            'log_prob_target': log_prob_target if log_prob_target is not None else 0.0,
            'total_log_prob': log_prob_target if log_prob_target is not None else 0.0,
        }

    def evaluate_mcs_high(self, high_states: torch.Tensor, mode_actions: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量评估 High Actor: 返回 (log_probs, entropies)"""
        logits = self.mcs_high_actor(high_states.to(self.device))
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(mode_actions.to(self.device))
        entropies = dist.entropy()
        return log_probs, entropies

    def evaluate_mcs_low(self, self_states: torch.Tensor,
                         candidates_list: List[torch.Tensor],
                         target_actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量评估 Low Actor: 逐样本计算, 返回 (log_probs, entropies)"""
        log_probs = []
        entropies = []
        for i, (ss, cand, act) in enumerate(zip(self_states, candidates_list, target_actions)):
            ss = ss.to(self.device)
            cand = cand.to(self.device)
            if cand.shape[0] == 0 or act.item() < 0:
                log_probs.append(torch.tensor(0.0, device=self.device))
                entropies.append(torch.tensor(0.0, device=self.device))
            else:
                scores = self.mcs_low_actor(ss, cand)
                probs = F.softmax(scores, dim=-1)
                dist = Categorical(probs)
                lp = dist.log_prob(act.to(self.device))
                ent = dist.entropy()
                log_probs.append(lp)
                entropies.append(ent)
        return torch.stack(log_probs), torch.stack(entropies)

    def evaluate_iev(self, self_states: torch.Tensor,
                     candidates_list: List[torch.Tensor],
                     target_actions: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量评估 IEV Actor: 逐样本计算"""
        log_probs = []
        entropies = []
        for i, (ss, cand, act) in enumerate(zip(self_states, candidates_list, target_actions)):
            ss = ss.to(self.device)
            cand = cand.to(self.device)
            if cand.shape[0] == 0 or act.item() < 0:
                log_probs.append(torch.tensor(0.0, device=self.device))
                entropies.append(torch.tensor(0.0, device=self.device))
            else:
                scores = self.iev_actor(ss, cand)
                probs = F.softmax(scores, dim=-1)
                dist = Categorical(probs)
                lp = dist.log_prob(act.to(self.device))
                ent = dist.entropy()
                log_probs.append(lp)
                entropies.append(ent)
        return torch.stack(log_probs), torch.stack(entropies)

    def get_critic_value(self, global_states: torch.Tensor) -> torch.Tensor:
        return self.critic(global_states.to(self.device))
