"""Train v14 with outcome-only Low rewards and unchanged graph architecture.

This is intentionally a new entry point: existing v14 checkpoints and the
original v10-onlylow reward remain reproducible.  The only experimental change
is the Low reward supplied by ``reward_v14.V14OutcomeRewardBuilder``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import train as base_train
import train_v14 as v14_train
from reward_v14 import V14OutcomeRewardBuilder


class V14OutcomeRewardEnv(v14_train.V14MultiAgentEnv):
    """v14 observations/simulator, with a separate outcome-only reward."""

    instances = {}

    def __init__(self, seed=42):
        super().__init__(seed)
        self.world.reward_builder = V14OutcomeRewardBuilder()


def collect_episode_with_outcome_reward_metrics(agent, args, episode):
    result = v14_train._collect_episode(agent, args, episode)
    environment = V14OutcomeRewardEnv.instances.pop(
        int(args.seed + episode - 1)
    )
    result[0].update(environment.metrics())
    return result


def _write_reward_metadata(output_dir: Path) -> None:
    path = output_dir / 'training_config.json'
    if not path.is_file():
        return
    config = json.loads(path.read_text(encoding='utf-8'))
    config.update({
        'reward_design': 'v14_outcome_only_no_manual_spatial_shaping',
        'low_reward_design_version': 'v14_outcome_only',
        # Override inherited v10 metadata values so they cannot be mistaken
        # for active reward coefficients in this experiment.
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
            'only_reward_changed_from_v14; graph, simulation, candidate '
            'actions, PPO, fixed_40kwh_high_rule_unchanged'
        ),
    })
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')


def _output_dir(arguments) -> Path:
    for index, value in enumerate(arguments):
        if value == '--output-dir' and index + 1 < len(arguments):
            return Path(arguments[index + 1])
    return Path(__file__).resolve().parent.parent / 'training_results_v14_outcome_reward'


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--fixed-high-threshold' not in arguments:
        arguments.append('--fixed-high-threshold')
    if '--output-dir' not in arguments:
        arguments.extend(['--output-dir', str(_output_dir(arguments))])
    sys.argv = [sys.argv[0], *arguments]

    # ``train.collect_episode`` normally calls this class directly to compute
    # the legacy candidate-priority scalar.  Rebinding it to the v14 builder
    # makes that scalar exactly zero before it reaches the event record.
    base_train.RewardBuilder = V14OutcomeRewardBuilder
    base_train.MultiAgentEnv = V14OutcomeRewardEnv
    base_train.MCSMAPPOAgent = v14_train.TrainV14Agent
    base_train.collect_episode = collect_episode_with_outcome_reward_metrics
    base_train.ppo_update = v14_train.ppo_update_with_v14_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_outcome_reward_metadata():
        args = original_parse_args()
        # Reuse every architecture/diagnostic label of v14, then make the
        # reward treatment explicit in the artifact.
        args.graph_model = 'canonical_global_heterogeneous_relation_graph'
        args.graph_scope = 'all_10_mcs_5_fcs_300_ev_fixed_node_slots'
        args.graph_message_passing = 'local_relation_weighted_mean'
        args.graph_layers = int(v14_train.GRAPH_LAYERS)
        args.graph_hidden_dim = int(v14_train.GRAPH_HIDDEN_DIM)
        args.graph_feature_schema = v14_train.FEATURE_SCHEMA
        args.graph_global_pooling = False
        args.graph_regional_tokens = False
        args.actor_global_readout = False
        args.actor_legacy_manual_candidate_features = False
        args.local_soft_occupancy = False
        args.local_sequential_reservation = False
        args.action_postprocessing = False
        args.additional_graph_reward_shaping = False
        args.reward_design = 'v14_outcome_only_no_manual_spatial_shaping'
        args.simulator_changes = 'none'
        return args

    base_train.parse_args = parse_args_with_outcome_reward_metadata
    base_train.main()
    _write_reward_metadata(_output_dir(arguments))


if __name__ == '__main__':
    main()
