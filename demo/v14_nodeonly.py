"""Node-only ablation of the v14 canonical global graph model.

This module intentionally leaves every v14 source file unchanged.  It reuses
the same node features, candidate set, Actor/Critic heads and two residual
relation-update layers, but zeroes all five relation matrices inside a copied
policy input before encoding.  Consequently no node can receive information
from any neighbor while the non-edge architecture and parameter shapes remain
identical to v14.
"""
from __future__ import annotations

import torch

from global_graph_v14 import (
    GlobalLocalGraphEncoder,
    V14GlobalGraphActor,
    V14GlobalGraphCritic,
)
from observation_v14 import V14_LAYOUT
from v14_global_graph import V14GlobalGraphMAPPOAgent


NODE_FEATURE_END = 1 + (
    V14_LAYOUT.mcs_features
    + V14_LAYOUT.fcs_features
    + V14_LAYOUT.ev_features
)


class V14NodeOnlyEncoder(GlobalLocalGraphEncoder):
    """V14 encoder with all neighbor messages deterministically disabled."""

    @staticmethod
    def remove_edges(policy_self_state: torch.Tensor) -> torch.Tensor:
        copied = policy_self_state.clone()
        copied[..., NODE_FEATURE_END:] = 0.0
        return copied

    def forward(self, policy_self_state: torch.Tensor):
        return super().forward(self.remove_edges(policy_self_state))


class V14NodeOnlyActor(V14GlobalGraphActor):
    """Same v14 scoring head driven by independently encoded nodes."""

    def __init__(
        self,
        hidden_dim: int = 128,
        graph_hidden_dim: int = 16,
        graph_layers: int = 2,
    ):
        super().__init__(hidden_dim, graph_hidden_dim, graph_layers)
        self.graph_encoder = V14NodeOnlyEncoder(
            graph_hidden_dim, graph_layers
        )


class V14NodeOnlyCritic(V14GlobalGraphCritic):
    """Same centralized v14 Critic without inter-node messages."""

    def __init__(
        self,
        original_global_dim: int,
        hidden_dim: int = 128,
        graph_hidden_dim: int = 16,
        graph_layers: int = 2,
    ):
        super().__init__(
            original_global_dim,
            hidden_dim,
            graph_hidden_dim,
            graph_layers,
        )
        self.graph_encoder = V14NodeOnlyEncoder(
            graph_hidden_dim, graph_layers
        )


class V14NodeOnlyMAPPOAgent(V14GlobalGraphMAPPOAgent):
    """Fixed-High Low-MAPPO ablation differing from v14 only by zero edges."""

    CHECKPOINT_VERSION = 1
    ARCHITECTURE = 'v14_nodeonly_canonical_nodes_no_message_passing_low_mappo'

    def __init__(
        self,
        original_global_state_dim: int,
        hidden_dim: int = 128,
        graph_hidden_dim: int = 16,
        graph_layers: int = 2,
        low_actor_lr: float = 3e-4,
        low_critic_lr: float = 5e-4,
        recharge_threshold_kwh: float = 40.0,
        battery_capacity_kwh: float = 300.0,
        device: str = 'cpu',
    ):
        super().__init__(
            original_global_state_dim=original_global_state_dim,
            hidden_dim=hidden_dim,
            graph_hidden_dim=graph_hidden_dim,
            graph_layers=graph_layers,
            low_actor_lr=low_actor_lr,
            low_critic_lr=low_critic_lr,
            recharge_threshold_kwh=recharge_threshold_kwh,
            battery_capacity_kwh=battery_capacity_kwh,
            device=device,
        )
        # Replace both branches before any rollout or optimizer update.  The
        # discarded parent modules are never exposed to training.
        self.low_actor = V14NodeOnlyActor(
            hidden_dim, graph_hidden_dim, graph_layers
        ).to(self.device)
        self.low_critic = V14NodeOnlyCritic(
            original_global_state_dim,
            hidden_dim,
            graph_hidden_dim,
            graph_layers,
        ).to(self.device)
        self.low_optimizer = torch.optim.Adam(
            self.low_actor.parameters(), lr=low_actor_lr
        )
        self.low_critic_optimizer = torch.optim.Adam(
            self.low_critic.parameters(), lr=low_critic_lr
        )

