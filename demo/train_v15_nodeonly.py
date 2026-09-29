"""Train the controlled node-only ablation of v15.

It uses the exact v15 simulator, reward, 40 kWh High threshold, candidate
action set, PPO settings and canonical global graph snapshots.  Only the
packed signed edges are hidden from both Low networks before encoding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import train as base_train
import train_v15 as v15_train
from config import COMM_RANGE, MCS_BATTERY_CAPACITY, MCS_RECHARGE_THRESHOLD
from observation_v15 import (
    FEATURE_SCHEMA,
    V15_CANDIDATE_DIM,
    V15_LAYOUT,
    V15_SELF_DIM,
)
from v15_nodeonly import V15NodeOnlyMAPPOAgent


GRAPH_HIDDEN_DIM = v15_train.GRAPH_HIDDEN_DIM
GRAPH_LAYERS = v15_train.GRAPH_LAYERS
ATTENTION_HEADS = v15_train.ATTENTION_HEADS


class TrainV15NodeOnlyAgent(V15NodeOnlyMAPPOAgent):
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
            attention_heads=ATTENTION_HEADS,
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
                / 'training_results_v15_nodeonly'
            ),
        ])
    sys.argv = [sys.argv[0], *arguments]

    # V15's environment builds the same canonical snapshots and records the
    # same topology metrics.  The node-only encoder discards their edge bytes.
    base_train.MultiAgentEnv = v15_train.V15MultiAgentEnv
    base_train.MCSMAPPOAgent = TrainV15NodeOnlyAgent
    base_train.collect_episode = v15_train.collect_episode_with_v15_metrics
    base_train.ppo_update = v15_train.ppo_update_with_v15_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_nodeonly_metadata():
        args = original_parse_args()
        args.graph_model = 'v15_canonical_global_signed_nodeonly_ablation'
        args.ablation_reference = (
            'v15_global_binary_signed_attention_graph'
        )
        args.ablation_variable = 'all_positive_and_negative_messages_disabled'
        args.graph_scope = 'same_v15_10_mcs_5_fcs_300_ev_fixed_node_slots'
        args.graph_active_ev = 'quasi_or_iev'
        args.graph_active_mcs = 'not_broken_and_not_energy_stranded'
        args.graph_edge_storage = 'lossless_bit_packed_binary_relations'
        args.graph_relations_constructed_for_audit = [
            'mcs_ev_positive', 'fcs_ev_positive',
            'mcs_mcs_negative', 'mcs_fcs_negative',
            'fcs_fcs_negative', 'ev_ev_negative',
        ]
        args.graph_relations_policy_visible = False
        args.graph_message_passing = (
            'disabled_zero_all_packed_signed_edges_before_every_encoder_forward'
        )
        args.graph_layers = int(GRAPH_LAYERS)
        args.graph_hidden_dim = int(GRAPH_HIDDEN_DIM)
        args.graph_attention_heads = int(ATTENTION_HEADS)
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
        args.node_feature_schema_identical_to_v15 = True
        args.candidate_action_set_identical_to_v15_v10_onlylow = True
        args.actor_critic_head_identical_to_v15 = True
        args.parameter_shapes_identical_to_v15 = True
        args.local_soft_occupancy = False
        args.local_sequential_reservation = False
        args.action_postprocessing = False
        args.additional_graph_reward_shaping = False
        args.reward_design = 'unchanged_from_v10_onlylow_v14_v15'
        args.simulator_changes = 'none'
        args.patrol_enabled = False
        args.v15_graph_state_dim = int(V15_LAYOUT.graph_dim)
        args.v15_policy_self_dim = int(V15_SELF_DIM)
        args.v15_policy_candidate_dim = int(V15_CANDIDATE_DIM)
        args.training_diagnostic_expectation = (
            'edge_ablation_counterfactual_metrics_must_be_exactly_zero'
        )
        return args

    base_train.parse_args = parse_args_with_nodeonly_metadata
    base_train.main()


if __name__ == '__main__':
    main()
