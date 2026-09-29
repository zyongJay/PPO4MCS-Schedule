"""Train the node-only v14 ablation with the outcome-only v14 reward.

This is the companion control for ``train_v14_outcome_reward.py``.  It keeps
the same simulator, fixed 40 kWh High rule, node states, candidate actions,
PPO settings and outcome-only reward, while disabling every inter-node message
in both the Low Actor and Critic.  It therefore isolates the contribution of
v14 relation aggregation without reintroducing manual spatial reward shaping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import train as base_train
import train_v14 as v14_train
from config import COMM_RANGE
from observation_v14 import FEATURE_SCHEMA, V14_CANDIDATE_DIM, V14_LAYOUT, V14_SELF_DIM
from reward_v14 import V14OutcomeRewardBuilder
from train_v14_nodeonly import TrainV14NodeOnlyAgent
from train_v14_outcome_reward import V14OutcomeRewardEnv, _write_reward_metadata


class V14NodeOnlyOutcomeRewardEnv(V14OutcomeRewardEnv):
    """Separate instance registry for the node-only outcome-reward run."""

    instances = {}


def collect_episode_with_nodeonly_outcome_metrics(agent, args, episode):
    result = v14_train._collect_episode(agent, args, episode)
    environment = V14NodeOnlyOutcomeRewardEnv.instances.pop(
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
        / 'training_results_v14_nodeonly_outcome_reward'
    )


def _write_metadata(output_dir: Path) -> None:
    _write_reward_metadata(output_dir)
    path = output_dir / 'training_config.json'
    if not path.is_file():
        return
    config = json.loads(path.read_text(encoding='utf-8'))
    config.update({
        'graph_model': 'v14_canonical_global_nodeonly_outcome_reward',
        'ablation_reference': 'v14_outcome_reward_global_graph',
        'ablation_variable': 'all_inter_node_messages_disabled',
        'graph_relations_constructed_for_audit': [
            'mcs_ev_attraction', 'fcs_ev_attraction',
            'mcs_mcs_competition', 'mcs_fcs_competition',
            'fcs_fcs_competition',
        ],
        'graph_relations_policy_visible': False,
        'graph_message_passing': (
            'disabled_zero_all_relation_matrices_before_every_encoder_forward'
        ),
        'node_feature_schema_identical_to_v14_outcome_reward': True,
        'candidate_action_set_identical_to_v14_outcome_reward': True,
        'actor_critic_head_identical_to_v14_outcome_reward': True,
        'parameter_shapes_identical_to_v14_outcome_reward': True,
        'reward_experiment_change': (
            'same outcome-only reward as v14_outcome_reward; only all '
            'inter-node graph messages are disabled'
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

    # Prevent base_train.collect_episode from constructing the legacy
    # hand-crafted candidate-priority reward scalar.
    base_train.RewardBuilder = V14OutcomeRewardBuilder
    base_train.MultiAgentEnv = V14NodeOnlyOutcomeRewardEnv
    base_train.MCSMAPPOAgent = TrainV14NodeOnlyAgent
    base_train.collect_episode = collect_episode_with_nodeonly_outcome_metrics
    base_train.ppo_update = v14_train.ppo_update_with_v14_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_nodeonly_outcome_metadata():
        args = original_parse_args()
        args.graph_model = 'v14_canonical_global_nodeonly_outcome_reward'
        args.ablation_reference = 'v14_outcome_reward_global_graph'
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
        args.graph_layers = int(v14_train.GRAPH_LAYERS)
        args.graph_hidden_dim = int(v14_train.GRAPH_HIDDEN_DIM)
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
        args.node_feature_schema_identical_to_v14_outcome_reward = True
        args.candidate_action_set_identical_to_v14_outcome_reward = True
        args.actor_critic_head_identical_to_v14_outcome_reward = True
        args.parameter_shapes_identical_to_v14_outcome_reward = True
        args.local_soft_occupancy = False
        args.local_sequential_reservation = False
        args.action_postprocessing = False
        args.additional_graph_reward_shaping = False
        args.reward_design = 'v14_outcome_only_no_manual_spatial_shaping'
        args.simulator_changes = 'none'
        args.patrol_enabled = False
        args.v14_graph_state_dim = int(V14_LAYOUT.graph_dim)
        args.v14_policy_self_dim = int(V14_SELF_DIM)
        args.v14_policy_candidate_dim = int(V14_CANDIDATE_DIM)
        args.training_diagnostic_expectation = (
            'edge_ablation_counterfactual_metrics_must_be_exactly_zero'
        )
        return args

    base_train.parse_args = parse_args_with_nodeonly_outcome_metadata
    base_train.main()
    _write_metadata(_output_dir(arguments))


if __name__ == '__main__':
    main()
