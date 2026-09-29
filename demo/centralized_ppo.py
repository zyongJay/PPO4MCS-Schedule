"""Single-agent PPO for centralized joint MCS dispatch.

The policy owns one joint action per physical step.  It autoregressively
assigns one destination to every currently eligible MCS while task MCSs are
excluded from the policy probability.  The actor and critic both consume the
same canonical per-step graph snapshot, but use separate neural encoders.

``encoder_mode='nodeonly'`` preserves every raw node feature, downstream
action-query module and parameter shape while zeroing only the five relation
matrices.  This is the matched no-message-passing control for later ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from config import NUM_MCS
from global_graph_v14 import GlobalLocalGraphEncoder
from observation_v14 import (
    EV_NODE_COUNT,
    FCS_NODE_COUNT,
    MCS_NODE_COUNT,
    MCS_NODE_DIM,
    V14_LAYOUT,
)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


RELATION_OFFSET = (
    V14_LAYOUT.mcs_features
    + V14_LAYOUT.fcs_features
    + V14_LAYOUT.ev_features
)


class CentralizedEntityEncoder(nn.Module):
    """Encode all canonical nodes with graph or matched node-only treatment."""

    def __init__(
        self, hidden_dim: int = 32, layers: int = 2,
        encoder_mode: str = 'graph',
    ):
        super().__init__()
        if encoder_mode not in {'graph', 'nodeonly'}:
            raise ValueError("encoder_mode must be 'graph' or 'nodeonly'")
        self.encoder_mode = encoder_mode
        self.encoder = GlobalLocalGraphEncoder(hidden_dim, layers)

    def forward(self, graph: torch.Tensor):
        if graph.dim() == 1:
            graph = graph.unsqueeze(0)
        if graph.shape[-1] != V14_LAYOUT.graph_dim:
            raise ValueError(
                f'centralized graph width {graph.shape[-1]} != '
                f'{V14_LAYOUT.graph_dim}'
            )
        effective = graph
        if self.encoder_mode == 'nodeonly':
            effective = graph.clone()
            effective[:, RELATION_OFFSET:] = 0.0
        addressing = torch.zeros(
            (effective.shape[0], 1),
            dtype=effective.dtype,
            device=effective.device,
        )
        packed = torch.cat((addressing, effective), dim=-1)
        _current, mcs_h, fcs_h, ev_h = self.encoder(packed)
        return mcs_h, fcs_h, ev_h


class CentralizedDispatchActor(nn.Module):
    """Autoregressive centralized Actor with dispatch-point action queries."""

    def __init__(
        self,
        dispatch_points: np.ndarray,
        graph_hidden_dim: int = 32,
        hidden_dim: int = 128,
        graph_layers: int = 2,
        encoder_mode: str = 'graph',
    ):
        super().__init__()
        points = torch.as_tensor(dispatch_points, dtype=torch.float32)
        if points.dim() != 2 or points.shape[1] != 2:
            raise ValueError('dispatch_points must have shape [R, 2]')
        self.register_buffer('dispatch_points', points)
        self.dispatch_count = int(points.shape[0])
        self.action_count = self.dispatch_count + 1
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.entity_encoder = CentralizedEntityEncoder(
            graph_hidden_dim, graph_layers, encoder_mode
        )
        self.query_encoder = _mlp(2, graph_hidden_dim, graph_hidden_dim)
        self.key = nn.Linear(graph_hidden_dim, graph_hidden_dim, bias=False)
        self.value = nn.Linear(graph_hidden_dim, graph_hidden_dim, bias=False)
        self.pair_encoder = _mlp(3, graph_hidden_dim, graph_hidden_dim)
        self.region_scorer = _mlp(
            graph_hidden_dim * 4, hidden_dim, 1
        )
        self.stay_token = nn.Parameter(torch.zeros(graph_hidden_dim))
        nn.init.normal_(self.stay_token, std=0.02)
        self.stay_scorer = _mlp(
            graph_hidden_dim * 3, hidden_dim, 1
        )

    def encode(self, graph: torch.Tensor) -> Dict[str, torch.Tensor]:
        mcs_h, fcs_h, ev_h = self.entity_encoder(graph)
        entities = torch.cat((mcs_h, fcs_h, ev_h), dim=1)
        query = self.query_encoder(self.dispatch_points).unsqueeze(0).expand(
            graph.shape[0], -1, -1
        )
        attention_logits = torch.einsum(
            'brh,bvh->brv', query, self.key(entities)
        ) / max(self.graph_hidden_dim ** 0.5, 1.0)
        attention = torch.softmax(attention_logits, dim=-1)
        context = torch.bmm(attention, self.value(entities))
        global_context = entities.mean(dim=1)
        return {
            'mcs': mcs_h,
            'query': query,
            'context': context,
            'global': global_context,
        }

    def logits_one(
        self,
        encoded: Dict[str, torch.Tensor],
        pair_features: torch.Tensor,
        mcs_indices: torch.Tensor,
        selected_counts: torch.Tensor,
    ) -> torch.Tensor:
        batch = pair_features.shape[0]
        batch_index = torch.arange(batch, device=pair_features.device)
        own = encoded['mcs'][batch_index, mcs_indices]
        own_expanded = own.unsqueeze(1).expand(-1, self.dispatch_count, -1)
        relation = torch.cat((
            pair_features[batch_index, mcs_indices, :self.dispatch_count],
            selected_counts.unsqueeze(-1),
        ), dim=-1)
        region_logits = self.region_scorer(torch.cat((
            own_expanded,
            encoded['context'],
            encoded['query'],
            self.pair_encoder(relation),
        ), dim=-1)).squeeze(-1)
        stay_token = self.stay_token.unsqueeze(0).expand(batch, -1)
        stay_logit = self.stay_scorer(torch.cat((
            own,
            encoded['global'],
            stay_token,
        ), dim=-1)).squeeze(-1)
        return torch.cat((region_logits, stay_logit.unsqueeze(-1)), dim=-1)

    @staticmethod
    def decision_order(eligible_mask: np.ndarray) -> np.ndarray:
        """Stable MCS-ID order; -1 pads rows with fewer decisions."""
        batch = eligible_mask.shape[0]
        result = np.full((batch, NUM_MCS), -1, dtype=np.int64)
        for row in range(batch):
            indices = np.flatnonzero(eligible_mask[row])
            result[row, :len(indices)] = indices
        return result

    @torch.no_grad()
    def sample(
        self,
        graph: torch.Tensor,
        pair_features: torch.Tensor,
        action_mask: torch.Tensor,
        eligible_mask: torch.Tensor,
        deterministic: bool = False,
    ):
        if graph.dim() == 1:
            graph = graph.unsqueeze(0)
            pair_features = pair_features.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            eligible_mask = eligible_mask.unsqueeze(0)
        encoded = self.encode(graph)
        eligible_np = eligible_mask.detach().cpu().numpy().astype(bool)
        order_np = self.decision_order(eligible_np)
        order = torch.as_tensor(order_np, device=graph.device)
        actions = torch.full(
            (graph.shape[0], NUM_MCS), -1,
            dtype=torch.long, device=graph.device,
        )
        selected_counts = torch.zeros(
            (graph.shape[0], self.dispatch_count),
            dtype=graph.dtype, device=graph.device,
        )
        joint_log_prob = torch.zeros(graph.shape[0], device=graph.device)
        joint_entropy = torch.zeros(graph.shape[0], device=graph.device)
        decision_count = torch.zeros(graph.shape[0], device=graph.device)
        for slot in range(NUM_MCS):
            current = order[:, slot]
            active = current >= 0
            if not bool(active.any()):
                continue
            rows = torch.nonzero(active, as_tuple=False).flatten()
            current_active = current[rows]
            sub_encoded = {key: value[rows] for key, value in encoded.items()}
            logits = self.logits_one(
                sub_encoded,
                pair_features[rows],
                current_active,
                selected_counts[rows],
            )
            valid = action_mask[rows, current_active]
            if not bool(valid.any(dim=-1).all()):
                raise RuntimeError('eligible MCS has no legal dispatch action')
            distribution = Categorical(logits=logits.masked_fill(
                ~valid, torch.finfo(logits.dtype).min
            ))
            chosen = (
                distribution.probs.argmax(dim=-1)
                if deterministic else distribution.sample()
            )
            actions[rows, current_active] = chosen
            joint_log_prob[rows] += distribution.log_prob(chosen)
            joint_entropy[rows] += distribution.entropy()
            decision_count[rows] += 1.0
            region_choice = chosen < self.dispatch_count
            if bool(region_choice.any()):
                update_rows = rows[region_choice]
                update_columns = chosen[region_choice]
                selected_counts[update_rows, update_columns] += 1.0
        mean_entropy = joint_entropy / decision_count.clamp_min(1.0)
        return actions, order, joint_log_prob, mean_entropy, decision_count

    def evaluate_joint(
        self,
        graph: torch.Tensor,
        pair_features: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
        order: torch.Tensor,
    ):
        encoded = self.encode(graph)
        selected_counts = torch.zeros(
            (graph.shape[0], self.dispatch_count),
            dtype=graph.dtype, device=graph.device,
        )
        joint_log_prob = torch.zeros(graph.shape[0], device=graph.device)
        joint_entropy = torch.zeros(graph.shape[0], device=graph.device)
        decision_count = torch.zeros(graph.shape[0], device=graph.device)
        for slot in range(NUM_MCS):
            current = order[:, slot]
            active = current >= 0
            if not bool(active.any()):
                continue
            rows = torch.nonzero(active, as_tuple=False).flatten()
            current_active = current[rows]
            sub_encoded = {key: value[rows] for key, value in encoded.items()}
            logits = self.logits_one(
                sub_encoded,
                pair_features[rows],
                current_active,
                selected_counts[rows],
            )
            valid = action_mask[rows, current_active]
            distribution = Categorical(logits=logits.masked_fill(
                ~valid, torch.finfo(logits.dtype).min
            ))
            chosen = actions[rows, current_active]
            joint_log_prob[rows] += distribution.log_prob(chosen)
            joint_entropy[rows] += distribution.entropy()
            decision_count[rows] += 1.0
            region_choice = chosen < self.dispatch_count
            if bool(region_choice.any()):
                update_rows = rows[region_choice]
                update_columns = chosen[region_choice]
                selected_counts[update_rows, update_columns] += 1.0
        return (
            joint_log_prob,
            joint_entropy / decision_count.clamp_min(1.0),
            decision_count,
        )


class CentralizedDispatchCritic(nn.Module):
    """One scalar centralized value for the complete physical state."""

    def __init__(
        self,
        graph_hidden_dim: int = 32,
        hidden_dim: int = 128,
        graph_layers: int = 2,
        encoder_mode: str = 'graph',
    ):
        super().__init__()
        self.entity_encoder = CentralizedEntityEncoder(
            graph_hidden_dim, graph_layers, encoder_mode
        )
        self.value_head = _mlp(graph_hidden_dim * 6, hidden_dim, 1)

    def forward(self, graph: torch.Tensor) -> torch.Tensor:
        if graph.dim() == 1:
            graph = graph.unsqueeze(0)
        mcs_h, fcs_h, ev_h = self.entity_encoder(graph)
        features = torch.cat((
            mcs_h.mean(dim=1),
            mcs_h.max(dim=1).values,
            fcs_h.mean(dim=1),
            fcs_h.max(dim=1).values,
            ev_h.mean(dim=1),
            ev_h.max(dim=1).values,
        ), dim=-1)
        return self.value_head(features).squeeze(-1)


@dataclass
class CentralizedStep:
    graph: np.ndarray
    pair_features: np.ndarray
    action_mask: np.ndarray
    actions: np.ndarray
    order: np.ndarray
    old_log_prob: float
    value: float
    reward: float
    done: bool
    has_decision: bool
    advantage: float = 0.0
    return_target: float = 0.0


class CentralizedRolloutBuffer:
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.steps: List[CentralizedStep] = []

    def add(self, step: CentralizedStep) -> None:
        self.steps.append(step)

    def finish(self, last_value: float = 0.0) -> None:
        gae = 0.0
        next_value = float(last_value)
        for item in reversed(self.steps):
            continuation = 0.0 if item.done else 1.0
            delta = (
                item.reward
                + self.gamma * continuation * next_value
                - item.value
            )
            gae = (
                delta
                + self.gamma * self.gae_lambda * continuation * gae
            )
            item.advantage = float(gae)
            item.return_target = float(gae + item.value)
            next_value = item.value

    def extend(self, other: 'CentralizedRolloutBuffer') -> None:
        self.steps.extend(other.steps)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)


class CentralizedPPOAgent:
    CHECKPOINT_VERSION = 1
    ARCHITECTURE = 'centralized_single_ppo_global_dispatch'

    def __init__(
        self,
        dispatch_points: np.ndarray,
        encoder_mode: str = 'graph',
        hidden_dim: int = 128,
        graph_hidden_dim: int = 32,
        graph_layers: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.encoder_mode = str(encoder_mode)
        self.hidden_dim = int(hidden_dim)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.graph_layers = int(graph_layers)
        self.dispatch_points = np.asarray(dispatch_points, np.float32)
        self.actor = CentralizedDispatchActor(
            self.dispatch_points,
            graph_hidden_dim,
            hidden_dim,
            graph_layers,
            encoder_mode,
        ).to(self.device)
        self.critic = CentralizedDispatchCritic(
            graph_hidden_dim,
            hidden_dim,
            graph_layers,
            encoder_mode,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @torch.no_grad()
    def act(self, observation: Dict, deterministic: bool = False):
        graph = self._tensor(observation['graph'])
        pair = self._tensor(observation['pair_features'])
        mask = self._tensor(observation['action_mask'], torch.bool)
        eligible = self._tensor(observation['eligible_mask'], torch.bool)
        actions, order, log_prob, entropy, count = self.actor.sample(
            graph, pair, mask, eligible, deterministic
        )
        value = self.critic(graph)
        return {
            'actions': actions.squeeze(0).cpu().numpy(),
            'order': order.squeeze(0).cpu().numpy(),
            'log_prob': float(log_prob.squeeze(0).item()),
            'entropy': float(entropy.squeeze(0).item()),
            'decision_count': int(count.squeeze(0).item()),
            'value': float(value.squeeze(0).item()),
        }

    @torch.no_grad()
    def value(self, graph: np.ndarray) -> float:
        return float(self.critic(self._tensor(graph)).squeeze(0).item())

    def save(self, path: Path, metadata: Dict | None = None) -> None:
        torch.save({
            'checkpoint_version': self.CHECKPOINT_VERSION,
            'architecture': self.ARCHITECTURE,
            'encoder_mode': self.encoder_mode,
            'hidden_dim': self.hidden_dim,
            'graph_hidden_dim': self.graph_hidden_dim,
            'graph_layers': self.graph_layers,
            'dispatch_points': self.dispatch_points,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'metadata': metadata or {},
        }, Path(path))

    def load(self, path: Path, load_optimizers: bool = False) -> Dict:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        if checkpoint.get('architecture') != self.ARCHITECTURE:
            raise ValueError('checkpoint is not a centralized PPO model')
        if checkpoint.get('encoder_mode') != self.encoder_mode:
            raise ValueError('centralized checkpoint encoder mode mismatch')
        if not np.allclose(
            checkpoint['dispatch_points'], self.dispatch_points
        ):
            raise ValueError('centralized checkpoint dispatch grid mismatch')
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer']
            )
            self.critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer']
            )
        return dict(checkpoint.get('metadata', {}))


def _normalise(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return values - values.mean()
    return (values - values.mean()) / (
        values.std(unbiased=False) + 1e-8
    )


def ppo_update(
    agent: CentralizedPPOAgent,
    buffer: CentralizedRolloutBuffer,
    update_epochs: int = 8,
    minibatch_size: int = 64,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    if not buffer.steps:
        return {'actor_loss': 0.0, 'critic_loss': 0.0}
    graph = agent._tensor(np.stack([item.graph for item in buffer.steps]))
    pair = agent._tensor(np.stack([
        item.pair_features for item in buffer.steps
    ]))
    masks = agent._tensor(np.stack([
        item.action_mask for item in buffer.steps
    ]), torch.bool)
    actions = agent._tensor(np.stack([
        item.actions for item in buffer.steps
    ]), torch.long)
    orders = agent._tensor(np.stack([
        item.order for item in buffer.steps
    ]), torch.long)
    old_log_prob = agent._tensor(np.asarray([
        item.old_log_prob for item in buffer.steps
    ], np.float32))
    advantages = agent._tensor(np.asarray([
        item.advantage for item in buffer.steps
    ], np.float32))
    returns = agent._tensor(np.asarray([
        item.return_target for item in buffer.steps
    ], np.float32))
    decision_mask = agent._tensor(np.asarray([
        item.has_decision for item in buffer.steps
    ], bool), torch.bool)
    actor_advantages = advantages.clone()
    if bool(decision_mask.any()):
        actor_advantages[decision_mask] = _normalise(
            advantages[decision_mask]
        )

    actor_losses: List[float] = []
    critic_losses: List[float] = []
    entropies: List[float] = []
    approx_kls: List[float] = []
    clip_fractions: List[float] = []
    sample_count = len(buffer.steps)
    for _epoch in range(int(update_epochs)):
        permutation = torch.randperm(sample_count, device=agent.device)
        for start in range(0, sample_count, int(minibatch_size)):
            indices = permutation[start:start + int(minibatch_size)]
            actor_indices = indices[decision_mask[indices]]
            if actor_indices.numel() > 0:
                new_log_prob, entropy, _count = agent.actor.evaluate_joint(
                    graph[actor_indices],
                    pair[actor_indices],
                    masks[actor_indices],
                    actions[actor_indices],
                    orders[actor_indices],
                )
                log_ratio = new_log_prob - old_log_prob[actor_indices]
                ratio = torch.exp(log_ratio)
                advantage = actor_advantages[actor_indices]
                surrogate = ratio * advantage
                clipped = torch.clamp(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                ) * advantage
                actor_loss = -torch.minimum(surrogate, clipped).mean()
                actor_objective = actor_loss - entropy_coef * entropy.mean()
                agent.actor_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), max_grad_norm
                )
                agent.actor_optimizer.step()
                actor_losses.append(float(actor_loss.item()))
                entropies.append(float(entropy.mean().item()))
                approx_kls.append(float((-log_ratio).mean().item()))
                clip_fractions.append(float((
                    torch.abs(ratio - 1.0) > clip_ratio
                ).float().mean().item()))

            values = agent.critic(graph[indices])
            critic_loss = value_coef * torch.mean(
                (values - returns[indices]) ** 2
            )
            agent.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                agent.critic.parameters(), max_grad_norm
            )
            agent.critic_optimizer.step()
            critic_losses.append(float(critic_loss.item()))

    return {
        'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
        'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
        'entropy': float(np.mean(entropies)) if entropies else 0.0,
        'approx_kl': float(np.mean(approx_kls)) if approx_kls else 0.0,
        'clip_fraction': float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        'update_step_count': int(sample_count),
        'update_decision_step_count': int(decision_mask.sum().item()),
    }
