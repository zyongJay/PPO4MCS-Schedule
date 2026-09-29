"""Train v15 signed graph with the v14 outcome-only Low reward.

The simulator, 40 kWh fixed High rule, candidate actions, PPO settings and
signed graph architecture are unchanged from :mod:`train_v15`.  Only the Low
reward is switched to ``V14OutcomeRewardBuilder`` so the graph is not trained
alongside hand-designed spatial reward shaping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import train as base_train
import train_v15 as v15_train
from reward_v14 import V14OutcomeRewardBuilder


class V15OutcomeRewardEnv(v15_train.V15MultiAgentEnv):
    """v15 signed observations and simulator with outcome-only reward."""

    instances = {}

    def __init__(self, seed=42):
        super().__init__(seed)
        self.world.reward_builder = V14OutcomeRewardBuilder()


def collect_episode_with_v15_outcome_metrics(agent, args, episode):
    result = v15_train._collect_episode(agent, args, episode)
    environment = V15OutcomeRewardEnv.instances.pop(
        int(args.seed + episode - 1)
    )
    result[0].update(environment.metrics())
    return result


def _output_dir(arguments) -> Path:
    for index, value in enumerate(arguments):
        if value == '--output-dir' and index + 1 < len(arguments):
            return Path(arguments[index + 1])
    return (
        Path(__file__).resolve().parent.parent
        / 'training_results_v15_outcome_reward'
    )


def _write_metadata(output_dir: Path) -> None:
    path = output_dir / 'training_config.json'
    if not path.is_file():
        return
    config = json.loads(path.read_text(encoding='utf-8'))
    config.update({
        'reward_design': 'v14_outcome_only_no_manual_spatial_shaping',
        'low_reward_design_version': 'v14_outcome_only',
        'low_spatial_opportunity_weight': 0.0,
        'low_candidate_priority_weight': 0.0,
        'low_fcs_alternative_penalty': 0.0,
        'low_mcs_alternative_penalty': 0.0,
        'low_mcs_competition_importance': 0.0,
        'quasi_attraction_weight': 0.0,
        'iev_attraction_weight': 0.0,
        'manual_spatial_reward_terms': {
            'candidate_urgency_attraction_priority': False,
            'spatial_desirability_difference': False,
            'attraction_competition_reward': False,
            'spatial_wait_opportunity_penalty': False,
        },
        'retained_low_reward_terms': [
            'actual_mcs_success_with_delayed_responsibility_credit',
            'attributed_controllable_failure',
            'realised_mcs_profit',
            'movement_energy_cost',
            'nonspatial_voluntary_wait_persistence_cost',
            'safety_events',
        ],
        'reward_experiment_change': (
            'only_reward_changed_from_v15; signed graph, simulation, '
            'candidate actions, PPO, fixed_40kwh_high_rule_unchanged'
        ),
    })
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--fixed-high-threshold' not in arguments:
        arguments.append('--fixed-high-threshold')
    if '--output-dir' not in arguments:
        arguments.extend(['--output-dir', str(_output_dir(arguments))])
    sys.argv = [sys.argv[0], *arguments]

    # The base collector constructs this class for the legacy priority scalar.
    # Rebinding it guarantees that scalar is zero before an event is recorded.
    base_train.RewardBuilder = V14OutcomeRewardBuilder
    base_train.MultiAgentEnv = V15OutcomeRewardEnv
    base_train.MCSMAPPOAgent = v15_train.TrainV15Agent
    base_train.collect_episode = collect_episode_with_v15_outcome_metrics
    base_train.ppo_update = v15_train.ppo_update_with_v15_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_v15_outcome_metadata():
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
        args.graph_layers = int(v15_train.GRAPH_LAYERS)
        args.graph_hidden_dim = int(v15_train.GRAPH_HIDDEN_DIM)
        args.graph_attention_heads = int(v15_train.ATTENTION_HEADS)
        args.graph_positive_relations = ['mcs_ev', 'fcs_ev']
        args.graph_negative_relations = [
            'mcs_mcs', 'mcs_fcs', 'fcs_fcs', 'ev_ev',
        ]
        args.graph_communication_range_km = float(v15_train.COMM_RANGE)
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
        args.graph_feature_schema = v15_train.FEATURE_SCHEMA
        args.local_soft_occupancy = False
        args.local_sequential_reservation = False
        args.action_postprocessing = False
        args.additional_graph_reward_shaping = False
        args.reward_design = 'v14_outcome_only_no_manual_spatial_shaping'
        args.simulator_changes = 'none'
        args.candidate_action_set = 'unchanged_v14_v10_onlylow'
        args.patrol_enabled = False
        return args

    base_train.parse_args = parse_args_with_v15_outcome_metadata
    base_train.main()
    _write_metadata(_output_dir(arguments))


if __name__ == '__main__':
    main()
