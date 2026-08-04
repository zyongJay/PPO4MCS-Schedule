"""Neural networks for option-aware MCS-only MAPPO/PPO training.

High actions use the fixed indices 0=Serve, 1=Recharge, 2=Wait.  Both actor
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
MCS_ACTION_NAMES = ("Serve", "Recharge", "Wait")


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
    """Local High Actor: normalized high state -> Serve/Recharge/Wait logits."""

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
    """Centralized value function over a normalized fixed-width global state."""

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


class MCSMAPPOAgent:
    """MCS-only High/Low actors, centralized critic, and their optimizers."""

    def __init__(
        self,
        high_state_dim: int,
        low_self_dim: int,
        low_candidate_dim: int,
        critic_state_dim: int,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.high_actor = MCSHighActor(high_state_dim, hidden_dim).to(self.device)
        self.low_actor = MCSLowActor(
            low_self_dim, low_candidate_dim, hidden_dim
        ).to(self.device)
        self.critic = CentralizedCritic(critic_state_dim, hidden_dim).to(self.device)
        self.high_optimizer = torch.optim.Adam(
            self.high_actor.parameters(), lr=actor_lr
        )
        self.low_optimizer = torch.optim.Adam(
            self.low_actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

    def train(self) -> None:
        self.high_actor.train()
        self.low_actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.high_actor.eval()
        self.low_actor.eval()
        self.critic.eval()

    def _tensor(self, value: TensorLike, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @torch.no_grad()
    def get_values_batch(self, critic_states: TensorLike) -> np.ndarray:
        states = self._tensor(critic_states)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        return self.critic(states).detach().cpu().numpy()

    @torch.no_grad()
    def get_value(self, critic_state: TensorLike) -> float:
        return float(self.get_values_batch(critic_state)[0])

    @torch.no_grad()
    def select_mcs_actions_batch(self, observations: list[Dict]) -> list[Dict]:
        """Sample High actions and the Serve-only Low subset in two GPU batches."""
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
        results = [{
            'mode': MCS_ACTION_NAMES[action_index],
            'high_action': int(action_index),
            'high_log_prob': float(high_log_prob_values[index]),
            'low_action': -1,
            'low_log_prob': 0.0,
        } for index, action_index in enumerate(action_indices)]

        serve_indices = np.flatnonzero(action_indices == 0)
        if serve_indices.size:
            low_self_states = self._tensor(np.stack([
                observations[index]['low_self_state']
                for index in serve_indices
            ]))
            low_candidates = self._tensor(np.stack([
                observations[index]['low_candidates']
                for index in serve_indices
            ]))
            low_masks = self._tensor(np.stack([
                observations[index]['low_candidate_mask']
                for index in serve_indices
            ]), dtype=torch.bool)
            low_distribution = self.low_actor.distribution(
                low_self_states, low_candidates, low_masks
            )
            low_actions = low_distribution.sample()
            low_log_probs = low_distribution.log_prob(low_actions)
            low_action_values = low_actions.detach().cpu().numpy().astype(int)
            low_log_prob_values = low_log_probs.detach().cpu().numpy()
            for batch_index, observation_index in enumerate(serve_indices):
                results[int(observation_index)]['low_action'] = int(
                    low_action_values[batch_index]
                )
                results[int(observation_index)]['low_log_prob'] = float(
                    low_log_prob_values[batch_index]
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

    def values(self, critic_states: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_states.to(self.device))

    def save(self, path: Union[str, Path], metadata: Dict | None = None) -> None:
        checkpoint = {
            "high_actor": self.high_actor.state_dict(),
            "low_actor": self.low_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "high_optimizer": self.high_optimizer.state_dict(),
            "low_optimizer": self.low_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, Path(path))

    def load(self, path: Union[str, Path]) -> Dict:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        self.high_actor.load_state_dict(checkpoint["high_actor"])
        self.low_actor.load_state_dict(checkpoint["low_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.high_optimizer.load_state_dict(checkpoint["high_optimizer"])
        self.low_optimizer.load_state_dict(checkpoint["low_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return checkpoint.get("metadata", {})


# Backward-compatible name for callers that previously imported MAPPOAgent.
MAPPOAgent = MCSMAPPOAgent
