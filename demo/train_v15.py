"""Train v15: binary signed global graph + two-hop sparse attention MAPPO.

The simulator, reward, fixed 40 kWh High rule, candidate actions, rollout
semantics and PPO defaults are unchanged from v14/v10-onlylow.  V15 changes
only graph construction and the Low Actor/Critic graph encoder.
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
from observation_v15 import (
    FEATURE_SCHEMA,
    ObservationBuilderV15,
    V15_CANDIDATE_DIM,
    V15_LAYOUT,
    V15_SELF_DIM,
)
from v15_signed_graph import V15SignedGraphMAPPOAgent


GRAPH_HIDDEN_DIM = 10
GRAPH_LAYERS = 2
ATTENTION_HEADS = 2


class TrainV15Agent(V15SignedGraphMAPPOAgent):
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


class V15MultiAgentEnv(v14_train.V14MultiAgentEnv):
    """Observation-only v15 adapter; the physical simulator is unchanged."""

    instances = {}

    def __init__(self, seed=42):
        super().__init__(seed)
        self.world.obs_builder = ObservationBuilderV15(self.world)

    def metrics(self):
        result = self.world.obs_builder.graph_builder.metrics()
        result.update({
            'target_conflict_rate': (
                self.target_conflict_numerator
                / max(self.target_conflict_denominator, 1)
            ),
            'raw_sample_target_conflict_rate': (
                self.target_conflict_numerator
                / max(self.target_conflict_denominator, 1)
            ),
            'spatial_conflict_rate': (
                self.spatial_conflict_numerator
                / max(self.spatial_conflict_denominator, 1)
            ),
            'reservation_reassignment_rate': 0.0,
            'stay_due_to_reservation_rate': 0.0,
            'reservation_decision_count': 0,
            'reservation_reassignment_count': 0,
            'reservation_forced_stay_count': 0,
            'v15_graph_state_dim': int(V15_LAYOUT.graph_dim),
            'v15_policy_self_dim': int(V15_SELF_DIM),
            'v15_policy_candidate_dim': int(V15_CANDIDATE_DIM),
        })
        return result


_collect_episode = base_train.collect_episode


def collect_episode_with_v15_metrics(agent, args, episode):
    result = _collect_episode(agent, args, episode)
    seed = int(args.seed + episode - 1)
    environment = V15MultiAgentEnv.instances.pop(seed)
    result[0].update(environment.metrics())
    return result


_ppo_update = base_train.ppo_update


def ppo_update_with_v15_metrics(*args, **kwargs):
    agent = args[0]
    low_buffer = args[2]
    graph_parameters_before = agent.snapshot_graph_parameters()
    metrics = _ppo_update(*args, **kwargs)
    clip_ratio = float(
        kwargs.get('clip_ratio', args[6] if len(args) > 6 else 0.2)
    )
    metrics.update(agent.graph_parameter_update_diagnostics(
        graph_parameters_before
    ))
    metrics.update(agent.graph_counterfactual_diagnostics(
        low_buffer, clip_ratio
    ))
    metrics.update(agent.pop_training_diagnostics())
    return metrics


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--fixed-high-threshold' not in arguments:
        arguments.append('--fixed-high-threshold')
    if '--output-dir' not in arguments:
        arguments.extend([
            '--output-dir',
            str(
                Path(__file__).resolve().parent.parent
                / 'training_results_v15'
            ),
        ])
    sys.argv = [sys.argv[0], *arguments]
    base_train.MultiAgentEnv = V15MultiAgentEnv
    base_train.MCSMAPPOAgent = TrainV15Agent
    base_train.collect_episode = collect_episode_with_v15_metrics
    base_train.ppo_update = ppo_update_with_v15_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_v15_metadata():
        args = original_parse_args()
        args.graph_model = (
            'canonical_global_heterogeneous_binary_signed_attention_graph'
        )
        args.graph_scope = 'all_10_mcs_5_fcs_300_ev_fixed_node_slots'
        args.graph_active_ev = 'quasi_or_iev'
        args.graph_active_mcs = 'not_broken_and_not_energy_stranded'
        args.graph_edge_storage = 'lossless_bit_packed_binary_relations'
        args.graph_message_passing = (
            'two_hop_sparse_signed_multihead_attention_with_residuals'
        )
        args.graph_layers = int(GRAPH_LAYERS)
        args.graph_hidden_dim = int(GRAPH_HIDDEN_DIM)
        args.graph_attention_heads = int(ATTENTION_HEADS)
        args.graph_positive_relations = ['mcs_ev', 'fcs_ev']
        args.graph_negative_relations = [
            'mcs_mcs', 'mcs_fcs', 'fcs_fcs', 'ev_ev',
        ]
        args.graph_communication_range_km = float(COMM_RANGE)
        args.graph_edge_weight = 'binary_sign_only'
        args.graph_attention_normalization = (
            'separate_softmax_for_positive_and_negative_neighbors'
        )
        args.graph_negative_message_rule = (
            'independent_competition_channel_not_numeric_subtraction'
        )
        args.graph_global_pooling = False
        args.graph_regional_tokens = False
        args.actor_global_readout = False
        args.actor_visibility = (
            'own_mcs_and_candidate_ev_embeddings_after_two_signed_hops'
        )
        args.actor_legacy_manual_candidate_features = False
        args.critic_global_state = 'unchanged_v10_20_dimensional_state'
        args.graph_actor_critic_share_encoder = False
        args.graph_auxiliary_loss = False
        args.graph_training_diagnostics = [
            'signed_edge_counts_density_degrees_components_isolates',
            'positive_negative_attention_entropy_and_max_weight',
            'positive_negative_message_norm_and_gate_mean',
            'gradient_norm_and_parameter_updates',
            'same_type_embedding_dispersion',
            'all_edge_ablation_actor_and_critic_counterfactuals',
        ]
        args.graph_counterfactual_ablation = (
            'zero_all_bit_packed_signed_edges_keep_nodes_and_candidates'
        )
        args.graph_feature_schema = FEATURE_SCHEMA
        args.local_soft_occupancy = False
        args.local_sequential_reservation = False
        args.action_postprocessing = False
        args.additional_graph_reward_shaping = False
        args.reward_design = 'unchanged_from_v10_onlylow_and_v14'
        args.simulator_changes = 'none'
        args.candidate_action_set = 'unchanged_v14_v10_onlylow'
        args.patrol_enabled = False
        return args

    base_train.parse_args = parse_args_with_v15_metadata
    base_train.main()


if __name__ == '__main__':
    main()
