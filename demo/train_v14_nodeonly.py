"""Train the v14 node-only controlled ablation.

Everything outside inter-node message passing is inherited unchanged from
v14/v10-onlylow: simulator, reward, seeded scenarios, candidate actions,
40 kWh fixed High rule, rollout/GAE/PPO settings and checkpoint selection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import train as base_train
import train_v14 as v14_train
from config import (
    COMM_RANGE,
    MCS_BATTERY_CAPACITY,
    MCS_RECHARGE_THRESHOLD,
)
from observation_v14 import (
    FEATURE_SCHEMA,
    V14_CANDIDATE_DIM,
    V14_LAYOUT,
    V14_SELF_DIM,
)
from v14_nodeonly import V14NodeOnlyMAPPOAgent


GRAPH_HIDDEN_DIM = v14_train.GRAPH_HIDDEN_DIM
GRAPH_LAYERS = v14_train.GRAPH_LAYERS


class TrainV14NodeOnlyAgent(V14NodeOnlyMAPPOAgent):
    """Constructor-compatible adapter for the unchanged base trainer."""

    def __init__(
        self,
        high_state_dim,
        low_self_dim,
        low_candidate_dim,
        critic_state_dim,
        hidden_dim=128,
        actor_lr=3e-4,
        critic_lr=5e-4,
        low_actor_lr=None,
        low_critic_lr=None,
        device='cpu',
    ):
        del low_self_dim, low_candidate_dim, actor_lr, critic_lr
        super().__init__(
            original_global_state_dim=int(critic_state_dim - high_state_dim),
            hidden_dim=hidden_dim,
            graph_hidden_dim=GRAPH_HIDDEN_DIM,
            graph_layers=GRAPH_LAYERS,
            low_actor_lr=(low_actor_lr or 3e-4),
            low_critic_lr=(low_critic_lr or 5e-4),
            recharge_threshold_kwh=float(MCS_RECHARGE_THRESHOLD),
            battery_capacity_kwh=float(MCS_BATTERY_CAPACITY),
            device=device,
        )


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--fixed-high-threshold' not in arguments:
        arguments.append('--fixed-high-threshold')
    if '--output-dir' not in arguments:
        arguments.extend([
            '--output-dir',
            str(
                Path(__file__).resolve().parent.parent
                / 'training_results_v14_nodeonly'
            ),
        ])
    sys.argv = [sys.argv[0], *arguments]

    # Reuse v14's observation-only environment adapter and diagnostic wrappers.
    # No simulator/environment/reward source file is modified.
    base_train.MultiAgentEnv = v14_train.V14MultiAgentEnv
    base_train.MCSMAPPOAgent = TrainV14NodeOnlyAgent
    base_train.collect_episode = v14_train.collect_episode_with_v14_metrics
    base_train.ppo_update = v14_train.ppo_update_with_v14_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_nodeonly_metadata():
        args = original_parse_args()
        args.graph_model = 'v14_canonical_global_nodeonly_ablation'
        args.ablation_reference = 'v14_global_graph_local_aggregation'
        args.ablation_variable = 'all_inter_node_messages_disabled'
        args.graph_scope = 'same_v14_10_mcs_5_fcs_300_ev_node_slots'
        args.graph_relations_constructed_for_audit = [
            'mcs_ev_attraction', 'fcs_ev_attraction',
            'mcs_mcs_competition', 'mcs_fcs_competition',
            'fcs_fcs_competition',
        ]
        args.graph_relations_policy_visible = False
        args.graph_message_passing = (
            'disabled_zero_all_relation_matrices_before_every_encoder_forward'
        )
        args.graph_layers = int(GRAPH_LAYERS)
        args.graph_hidden_dim = int(GRAPH_HIDDEN_DIM)
        args.graph_communication_range_km = float(COMM_RANGE)
        args.graph_global_pooling = False
        args.graph_regional_tokens = False
        args.actor_global_readout = False
        args.actor_visibility = (
            'own_mcs_and_candidate_ev_independent_node_embeddings'
        )
        args.actor_legacy_manual_candidate_features = False
        args.actor_candidate_primitive_relations = [
            'is_stay', 'distance_ratio', 'mcs_energy_margin_ratio',
        ]
        args.critic_global_state = 'unchanged_v10_20_dimensional_state'
        args.graph_actor_critic_share_encoder = False
        args.graph_auxiliary_loss = False
        args.graph_feature_schema = FEATURE_SCHEMA
        args.node_feature_schema_identical_to_v14 = True
        args.candidate_action_set_identical_to_v14_v10_onlylow = True
        args.actor_critic_head_identical_to_v14 = True
        args.parameter_shapes_identical_to_v14 = True
        args.local_soft_occupancy = False
        args.local_sequential_reservation = False
        args.action_postprocessing = False
        args.additional_graph_reward_shaping = False
        args.reward_design = 'unchanged_from_v10_onlylow_and_v14'
        args.simulator_changes = 'none'
        args.patrol_enabled = False
        args.v14_graph_state_dim = int(V14_LAYOUT.graph_dim)
        args.v14_policy_self_dim = int(V14_SELF_DIM)
        args.v14_policy_candidate_dim = int(V14_CANDIDATE_DIM)
        args.training_diagnostic_expectation = (
            'edge_ablation_counterfactual_metrics_must_be_exactly_zero'
        )
        return args

    base_train.parse_args = parse_args_with_nodeonly_metadata
    base_train.main()


if __name__ == '__main__':
    main()
