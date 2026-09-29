"""Train v14: canonical global graph + local aggregation + shared Low MAPPO.

The simulator, matching, rewards, fixed 40 kWh High rule, Low action set,
rollout semantics, PPO hyperparameters and checkpoint selection are inherited
from v10-onlylow.  V14 changes only the Low policy/value representation:

* one canonical MCS/FCS/EV graph is built per environment step;
* relation-aware message passing is local to attraction/competition edges;
* Actor scores each MCS's unchanged Top-K candidates from its own and the
  candidate EV's node embeddings;
* no global readout, regional token, soft occupancy, reservation or action
  correction is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import train as base_train
from config import (
    COMM_RANGE,
    MCS_BATTERY_CAPACITY,
    MCS_RECHARGE_THRESHOLD,
)
from core import MCS, euclidean_distance
from environment import MultiAgentEnv
from observation_v14 import (
    FEATURE_SCHEMA,
    ObservationBuilderV14,
    V14_CANDIDATE_DIM,
    V14_LAYOUT,
    V14_SELF_DIM,
)
from v14_global_graph import V14GlobalGraphMAPPOAgent


# Capacity-matched against the v10-onlylow Low networks.  With hidden_dim=128
# this gives v14 37,201 Low Actor parameters versus v10-onlylow's 34,945;
# using 64 graph channels would inflate the Actor to 245,185 parameters and
# confound graph-representation gains with raw model capacity.
GRAPH_HIDDEN_DIM = 16
GRAPH_LAYERS = 2


class TrainV14Agent(V14GlobalGraphMAPPOAgent):
    """Constructor-compatible adapter for the unchanged v10 trainer."""

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


class V14MultiAgentEnv(MultiAgentEnv):
    """Observation-only adapter; the physical simulator is unchanged."""

    instances = {}

    def __init__(self, seed=42):
        super().__init__(seed)
        self.seed = int(seed)
        self.world.obs_builder = ObservationBuilderV14(self.world)
        self.target_conflict_numerator = 0
        self.target_conflict_denominator = 0
        self.spatial_conflict_numerator = 0
        self.spatial_conflict_denominator = 0
        self.__class__.instances[self.seed] = self

    def _record_actions(self, action_n):
        target_ids = [
            int(action.get('low_candidate_id', -1))
            for agent, action in zip(list(self.world.agents), action_n)
            if (
                isinstance(agent, MCS)
                and action.get('requested_mode') == 'Serve'
                and int(action.get('low_candidate_id', -1)) >= 0
            )
        ]
        counts = {
            target_id: target_ids.count(target_id)
            for target_id in set(target_ids)
        }
        self.target_conflict_numerator += sum(
            counts[target_id] > 1 for target_id in target_ids
        )
        self.target_conflict_denominator += len(target_ids)

    def _record_spatial_conflicts(self):
        active = [
            mcs for mcs in self.world.MCSs
            if not (mcs.is_broken or mcs.is_energy_stranded)
        ]
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                self.spatial_conflict_numerator += int(
                    euclidean_distance(*left.pos, *right.pos) / 1000.0
                    <= 3.0
                )
                self.spatial_conflict_denominator += 1

    def step(self, action_n):
        self._record_actions(action_n)
        result = super().step(action_n)
        self._record_spatial_conflicts()
        return result

    def metrics(self):
        result = self.world.obs_builder.graph_builder.metrics()
        result.update({
            'target_conflict_rate': (
                self.target_conflict_numerator
                / max(self.target_conflict_denominator, 1)
            ),
            # With no reservation or post-processing, sampled and executed
            # Low targets are identical by construction.
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
            'v14_graph_state_dim': int(V14_LAYOUT.graph_dim),
            'v14_policy_self_dim': int(V14_SELF_DIM),
            'v14_policy_candidate_dim': int(V14_CANDIDATE_DIM),
        })
        return result


_collect_episode = base_train.collect_episode


def collect_episode_with_v14_metrics(agent, args, episode):
    result = _collect_episode(agent, args, episode)
    seed = int(args.seed + episode - 1)
    environment = V14MultiAgentEnv.instances.pop(seed)
    result[0].update(environment.metrics())
    return result


_ppo_update = base_train.ppo_update


def ppo_update_with_v14_metrics(*args, **kwargs):
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
                / 'training_results_v14'
            ),
        ])
    sys.argv = [sys.argv[0], *arguments]
    base_train.MultiAgentEnv = V14MultiAgentEnv
    base_train.MCSMAPPOAgent = TrainV14Agent
    base_train.collect_episode = collect_episode_with_v14_metrics
    base_train.ppo_update = ppo_update_with_v14_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_v14_metadata():
        args = original_parse_args()
        args.graph_model = 'canonical_global_heterogeneous_relation_graph'
        args.graph_scope = 'all_10_mcs_5_fcs_300_ev_fixed_node_slots'
        args.graph_message_passing = 'local_relation_weighted_mean'
        args.graph_layers = int(GRAPH_LAYERS)
        args.graph_hidden_dim = int(GRAPH_HIDDEN_DIM)
        args.graph_relations = [
            'mcs_ev_attraction', 'fcs_ev_attraction',
            'mcs_mcs_competition', 'mcs_fcs_competition',
            'fcs_fcs_competition',
        ]
        args.graph_communication_range_km = float(COMM_RANGE)
        args.graph_smooth_radius_weight = 'half_cosine_compact_support'
        args.graph_global_pooling = False
        args.graph_regional_tokens = False
        args.actor_global_readout = False
        args.actor_visibility = (
            'own_mcs_and_candidate_ev_embeddings_after_local_messages'
        )
        args.actor_legacy_manual_candidate_features = False
        args.critic_global_state = 'unchanged_v10_20_dimensional_state'
        args.graph_actor_critic_share_encoder = False
        args.graph_auxiliary_loss = False
        args.graph_training_diagnostics = [
            'gradient_norm_and_finiteness',
            'parameter_update_norm_and_changed_fraction',
            'same_type_node_embedding_dispersion',
            'edge_ablation_actor_js_top1_surrogate_advantage_alignment',
            'edge_ablation_critic_value_mse_explained_variance_gain',
        ]
        args.graph_counterfactual_ablation = (
            'zero_all_relation_matrices_keep_all_node_and_candidate_features'
        )
        args.graph_counterfactual_max_samples_per_update = 256
        args.graph_feature_schema = FEATURE_SCHEMA
        args.local_soft_occupancy = False
        args.local_sequential_reservation = False
        args.action_postprocessing = False
        args.additional_graph_reward_shaping = False
        args.reward_design = 'unchanged_from_v10_onlylow'
        args.simulator_changes = 'none'
        args.patrol_enabled = False
        return args

    base_train.parse_args = parse_args_with_v14_metadata
    base_train.main()


if __name__ == '__main__':
    main()
