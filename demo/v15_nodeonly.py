"""Node-only ablation of the v15 signed global graph model.

The canonical v15 node slots, primitive node features, candidate actions,
Actor/Critic heads and optimizer settings are preserved.  The only ablated
operation is inter-node signed message passing: all packed positive/negative
edge bits are cleared immediately before every encoder forward pass.  The two
residual per-node transformations remain, so this is a controlled comparison
between independently encoded nodes and the full v15 signed aggregation.
"""
from __future__ import annotations

import torch

from global_signed_graph_v15 import (
    GlobalSignedGraphEncoder,
    V15SignedGraphActor,
    V15SignedGraphCritic,
)
from observation_v15 import V15_LAYOUT
from v15_signed_graph import V15SignedGraphMAPPOAgent


# Index 0 is the current-MCS address.  Everything after the primitive node
# features is losslessly bit-packed signed-edge storage in the v15 state.
NODE_FEATURE_END = 1 + V15_LAYOUT.node_feature_dim


class V15NodeOnlyEncoder(GlobalSignedGraphEncoder):
    """V15 encoder with every positive and negative neighbor link disabled."""

    @staticmethod
    def remove_edges(policy_self_state: torch.Tensor) -> torch.Tensor:
        node_only_state = policy_self_state.clone()
        node_only_state[..., NODE_FEATURE_END:] = 0.0
        return node_only_state

    def forward(self, policy_self_state: torch.Tensor):
        return super().forward(self.remove_edges(policy_self_state))


class V15NodeOnlyActor(V15SignedGraphActor):
    """Unchanged v15 Actor head over independently encoded MCS/EV nodes."""

    def __init__(
        self,
        hidden_dim: int = 128,
        graph_hidden_dim: int = 10,
        graph_layers: int = 2,
        attention_heads: int = 2,
    ):
        super().__init__(
            hidden_dim, graph_hidden_dim, graph_layers, attention_heads
        )
        self.graph_encoder = V15NodeOnlyEncoder(
            graph_hidden_dim, graph_layers, attention_heads
        )


class V15NodeOnlyCritic(V15SignedGraphCritic):
    """Unchanged v15 Critic head with no inter-node signed messages."""

    def __init__(
        self,
        original_global_dim: int,
        hidden_dim: int = 128,
        graph_hidden_dim: int = 10,
        graph_layers: int = 2,
        attention_heads: int = 2,
    ):
        super().__init__(
            original_global_dim,
            hidden_dim,
            graph_hidden_dim,
            graph_layers,
            attention_heads,
        )
        self.graph_encoder = V15NodeOnlyEncoder(
            graph_hidden_dim, graph_layers, attention_heads
        )


class V15NodeOnlyMAPPOAgent(V15SignedGraphMAPPOAgent):
    """Fixed-High Low-MAPPO ablation that differs from v15 only by its edges."""

    CHECKPOINT_VERSION = 1
    ARCHITECTURE = 'v15_nodeonly_canonical_nodes_no_signed_message_passing_low_mappo'

    def __init__(
        self,
        original_global_state_dim: int,
        hidden_dim: int = 128,
        graph_hidden_dim: int = 10,
        graph_layers: int = 2,
        attention_heads: int = 2,
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
            attention_heads=attention_heads,
            low_actor_lr=low_actor_lr,
            low_critic_lr=low_critic_lr,
            recharge_threshold_kwh=recharge_threshold_kwh,
            battery_capacity_kwh=battery_capacity_kwh,
            device=device,
        )
        # Replace the parent modules before the first rollout.  Parameter
        # shapes and all non-edge operations stay exactly aligned with v15.
        self.low_actor = V15NodeOnlyActor(
            hidden_dim, graph_hidden_dim, graph_layers, attention_heads
        ).to(self.device)
        self.low_critic = V15NodeOnlyCritic(
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
