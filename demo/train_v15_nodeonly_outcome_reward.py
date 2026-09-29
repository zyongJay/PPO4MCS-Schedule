"""Train the v15 node-only control with the v14 outcome-only reward."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import train as base_train
import train_v15 as v15_train
from reward_v14 import V14OutcomeRewardBuilder
from train_v15_nodeonly import TrainV15NodeOnlyAgent
from train_v15_outcome_reward import (
    V15OutcomeRewardEnv,
    _write_metadata as _write_outcome_metadata,
    collect_episode_with_v15_outcome_metrics,
)


def _output_dir(arguments) -> Path:
    for index, value in enumerate(arguments):
        if value == '--output-dir' and index + 1 < len(arguments):
            return Path(arguments[index + 1])
    return (
        Path(__file__).resolve().parent.parent
        / 'training_results_v15_nodeonly_outcome_reward'
    )


def _write_metadata(output_dir: Path) -> None:
    _write_outcome_metadata(output_dir)
    path = output_dir / 'training_config.json'
    if not path.is_file():
        return
    config = json.loads(path.read_text(encoding='utf-8'))
    config.update({
        'graph_model': 'v15_canonical_global_signed_nodeonly_outcome_reward',
        'ablation_reference': 'v15_outcome_reward_signed_graph',
        'ablation_variable': 'all_positive_and_negative_messages_disabled',
        'graph_relations_constructed_for_audit': [
            'mcs_ev_positive', 'fcs_ev_positive',
            'mcs_mcs_negative', 'mcs_fcs_negative',
            'fcs_fcs_negative', 'ev_ev_negative',
        ],
        'graph_relations_policy_visible': False,
        'graph_message_passing': (
            'disabled_zero_all_packed_signed_edges_before_every_encoder_forward'
        ),
        'node_feature_schema_identical_to_v15_outcome_reward': True,
        'candidate_action_set_identical_to_v15_outcome_reward': True,
        'actor_critic_head_identical_to_v15_outcome_reward': True,
        'parameter_shapes_identical_to_v15_outcome_reward': True,
        'reward_experiment_change': (
            'same outcome-only reward as v15_outcome_reward; only all '
            'inter-node signed graph messages are disabled'
        ),
        'training_diagnostic_expectation': (
            'edge_ablation_counterfactual_metrics_must_be_exactly_zero'
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

    base_train.RewardBuilder = V14OutcomeRewardBuilder
    base_train.MultiAgentEnv = V15OutcomeRewardEnv
    base_train.MCSMAPPOAgent = TrainV15NodeOnlyAgent
    base_train.collect_episode = collect_episode_with_v15_outcome_metrics
    base_train.ppo_update = v15_train.ppo_update_with_v15_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_nodeonly_outcome_metadata():
        args = original_parse_args()
        args.graph_model = 'v15_canonical_global_signed_nodeonly_outcome_reward'
        args.ablation_reference = 'v15_outcome_reward_signed_graph'
        args.ablation_variable = 'all_positive_and_negative_messages_disabled'
        args.graph_scope = 'same_v15_10_mcs_5_fcs_300_ev_node_slots'
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
        args.graph_layers = int(v15_train.GRAPH_LAYERS)
        args.graph_hidden_dim = int(v15_train.GRAPH_HIDDEN_DIM)
        args.graph_attention_heads = int(v15_train.ATTENTION_HEADS)
        args.graph_communication_range_km = float(v15_train.COMM_RANGE)
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
        args.graph_feature_schema = v15_train.FEATURE_SCHEMA
        args.local_soft_occupancy = False
        args.local_sequential_reservation = False
        args.action_postprocessing = False
        args.additional_graph_reward_shaping = False
        args.reward_design = 'v14_outcome_only_no_manual_spatial_shaping'
        args.simulator_changes = 'none'
        args.patrol_enabled = False
        args.v15_graph_state_dim = int(v15_train.V15_LAYOUT.graph_dim)
        args.v15_policy_self_dim = int(v15_train.V15_SELF_DIM)
        args.v15_policy_candidate_dim = int(v15_train.V15_CANDIDATE_DIM)
        args.training_diagnostic_expectation = (
            'edge_ablation_counterfactual_metrics_must_be_exactly_zero'
        )
        return args

    base_train.parse_args = parse_args_with_nodeonly_outcome_metadata
    base_train.main()
    _write_metadata(_output_dir(arguments))


if __name__ == '__main__':
    main()
