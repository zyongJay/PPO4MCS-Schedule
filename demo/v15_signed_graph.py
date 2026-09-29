"""Fixed-threshold Low MAPPO agent for the v15 signed global graph."""
from __future__ import annotations

from pathlib import Path

import torch

from config import MCS_BATTERY_CAPACITY, MCS_RECHARGE_THRESHOLD
from global_signed_graph_v15 import (
    V15SignedGraphActor,
    V15SignedGraphCritic,
)
from observation_v15 import (
    FEATURE_SCHEMA,
    V15_CANDIDATE_DIM,
    V15_LAYOUT,
    V15_SELF_DIM,
)
from v14_global_graph import V14GlobalGraphMAPPOAgent


class V15SignedGraphMAPPOAgent(V14GlobalGraphMAPPOAgent):
    """Shared Low Actor and centralized Low Critic with signed attention."""

    CHECKPOINT_VERSION = 1
    ARCHITECTURE = 'v15_canonical_global_binary_signed_attention_low_mappo'

    def __init__(
        self,
        original_global_state_dim: int,
        hidden_dim: int = 128,
        graph_hidden_dim: int = 10,
        graph_layers: int = 2,
        attention_heads: int = 2,
        low_actor_lr: float = 3e-4,
        low_critic_lr: float = 5e-4,
        recharge_threshold_kwh: float = MCS_RECHARGE_THRESHOLD,
        battery_capacity_kwh: float = MCS_BATTERY_CAPACITY,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.hidden_dim = int(hidden_dim)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.graph_layers = int(graph_layers)
        self.attention_heads = int(attention_heads)
        self.original_global_state_dim = int(original_global_state_dim)
        self.low_self_dim = int(V15_SELF_DIM)
        self.low_candidate_dim = int(V15_CANDIDATE_DIM)
        self.recharge_threshold_kwh = float(recharge_threshold_kwh)
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.low_actor = V15SignedGraphActor(
            hidden_dim,
            graph_hidden_dim,
            graph_layers,
            attention_heads,
        ).to(self.device)
        self.low_critic = V15SignedGraphCritic(
            original_global_state_dim,
            hidden_dim,
            graph_hidden_dim,
            graph_layers,
            attention_heads,
        ).to(self.device)
        self.low_optimizer = torch.optim.Adam(
            self.low_actor.parameters(), lr=low_actor_lr
        )
        self.low_critic_optimizer = torch.optim.Adam(
            self.low_critic.parameters(), lr=low_critic_lr
        )
        self.deterministic_low_actions = False

    @staticmethod
    def _edge_ablated_states(self_states):
        edge_start = 1 + V15_LAYOUT.node_feature_dim
        ablated = self_states.clone()
        ablated[:, edge_start:] = 0.0
        return ablated

    def pop_training_diagnostics(self):
        actor_grad, actor_finite = self._gradient_norm(
            self.low_actor.graph_encoder
        )
        critic_grad, critic_finite = self._gradient_norm(
            self.low_critic.graph_encoder
        )
        result = self.low_actor.pop_diagnostics()
        result.update(self.low_critic.graph_encoder.pop_diagnostics(
            'critic_graph'
        ))
        result.update({
            'actor_graph_grad_norm': float(actor_grad),
            'critic_graph_grad_norm': float(critic_grad),
            'actor_graph_grad_finite': int(actor_finite),
            'critic_graph_grad_finite': int(critic_finite),
            'actor_graph_parameter_count': int(sum(
                parameter.numel()
                for parameter in self.low_actor.graph_encoder.parameters()
            )),
            'critic_graph_parameter_count': int(sum(
                parameter.numel()
                for parameter in self.low_critic.graph_encoder.parameters()
            )),
            'v15_graph_state_dim': int(V15_LAYOUT.graph_dim),
            'v15_policy_self_dim': int(V15_SELF_DIM),
            'v15_policy_candidate_dim': int(V15_CANDIDATE_DIM),
        })
        return result

    def save(self, path, metadata=None):
        torch.save({
            'checkpoint_version': self.CHECKPOINT_VERSION,
            'architecture': self.ARCHITECTURE,
            'hidden_dim': self.hidden_dim,
            'graph_hidden_dim': self.graph_hidden_dim,
            'graph_layers': self.graph_layers,
            'attention_heads': self.attention_heads,
            'original_global_state_dim': self.original_global_state_dim,
            'low_self_dim': self.low_self_dim,
            'low_candidate_dim': self.low_candidate_dim,
            'recharge_threshold_kwh': self.recharge_threshold_kwh,
            'battery_capacity_kwh': self.battery_capacity_kwh,
            'feature_schema': FEATURE_SCHEMA,
            'low_actor': self.low_actor.state_dict(),
            'low_critic': self.low_critic.state_dict(),
            'low_optimizer': self.low_optimizer.state_dict(),
            'low_critic_optimizer': self.low_critic_optimizer.state_dict(),
            'metadata': metadata or {},
        }, Path(path))

    @classmethod
    def _validate_checkpoint(cls, checkpoint):
        if checkpoint.get('architecture') != cls.ARCHITECTURE:
            raise ValueError('checkpoint is not a v15 signed global graph')
        if checkpoint.get('feature_schema') != FEATURE_SCHEMA:
            raise ValueError('v15 checkpoint feature schema mismatch')
        if int(checkpoint['low_self_dim']) != int(V15_SELF_DIM):
            raise ValueError('v15 checkpoint self-state width mismatch')
        if int(checkpoint['low_candidate_dim']) != int(V15_CANDIDATE_DIM):
            raise ValueError('v15 checkpoint candidate width mismatch')

    @classmethod
    def from_checkpoint(cls, path, device='cpu', load_optimizers=False):
        checkpoint = torch.load(
            Path(path), map_location='cpu', weights_only=False
        )
        cls._validate_checkpoint(checkpoint)
        agent = cls(
            checkpoint['original_global_state_dim'],
            checkpoint['hidden_dim'],
            checkpoint['graph_hidden_dim'],
            checkpoint['graph_layers'],
            checkpoint.get('attention_heads', 2),
            recharge_threshold_kwh=checkpoint[
                'recharge_threshold_kwh'
            ],
            battery_capacity_kwh=checkpoint['battery_capacity_kwh'],
            device=device,
        )
        agent.low_actor.load_state_dict(checkpoint['low_actor'])
        agent.low_critic.load_state_dict(checkpoint['low_critic'])
        if load_optimizers:
            agent.low_optimizer.load_state_dict(checkpoint['low_optimizer'])
            agent.low_critic_optimizer.load_state_dict(
                checkpoint['low_critic_optimizer']
            )
        return agent, checkpoint.get('metadata', {})
