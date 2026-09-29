"""Train the frozen first-version B3 Competition Hypergraph model.

This entry point reuses train.py's v10-onlylow PPO, option boundaries, reward,
matcher and checkpoint selection verbatim.  The environment adapter only adds
sampling-time graph snapshots to observations/global state and audit metrics.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import train as base_train
from competition_hypergraph import (
    CompetitionHypergraphBuilder, CompetitionHypergraphConfig,
    V12_CANDIDATE_DIM, SOFT_OCCUPANCY_OFFSET,
)
from core import MCS, euclidean_distance
from config import (
    MCS_BATTERY_CAPACITY, MCS_CRITIC_STATE_DIM, MCS_GLOBAL_STATE_DIM,
    MCS_RECHARGE_THRESHOLD,
)
from environment import MultiAgentEnv
from v12_hypergraph import V12CompetitionHypergraphAgent


HG_CONFIG = CompetitionHypergraphConfig(
    r_merge_km=1.0, jaccard_eta=0.5, hidden_dim=64,
)


class TrainV12Agent(V12CompetitionHypergraphAgent):
    """Constructor-compatible adapter for train.py's Low-only branch."""
    def __init__(
        self, high_state_dim, low_self_dim, low_candidate_dim,
        critic_state_dim, hidden_dim=128, actor_lr=3e-4,
        critic_lr=5e-4, low_actor_lr=None, low_critic_lr=None,
        device='cpu',
    ):
        del low_candidate_dim, actor_lr, critic_lr
        original_global_state_dim = int(critic_state_dim - high_state_dim)
        super().__init__(
            low_self_dim, original_global_state_dim, hidden_dim=hidden_dim,
            hg_hidden_dim=HG_CONFIG.hidden_dim,
            low_actor_lr=(low_actor_lr or 3e-4),
            low_critic_lr=(low_critic_lr or 5e-4),
            recharge_threshold_kwh=float(MCS_RECHARGE_THRESHOLD),
            battery_capacity_kwh=float(MCS_BATTERY_CAPACITY),
            r_merge_km=HG_CONFIG.r_merge_km,
            jaccard_eta=HG_CONFIG.jaccard_eta, device=device,
        )


class V12MultiAgentEnv(MultiAgentEnv):
    instances = {}

    def __init__(self, seed=42):
        super().__init__(seed)
        self.seed = int(seed)
        self.builder = CompetitionHypergraphBuilder(HG_CONFIG)
        self.graph_rows = []
        self.target_conflict_numerator = 0
        self.target_conflict_denominator = 0
        self.cluster_conflict_numerator = 0
        self.spatial_conflict_numerator = 0
        self.spatial_conflict_denominator = 0
        self.raw_top1_conflict_numerator = 0
        self.raw_top1_conflict_denominator = 0
        self.raw_sample_conflict_numerator = 0
        self.raw_sample_conflict_denominator = 0
        self.reservation_decision_count = 0
        self.reservation_reassignment_count = 0
        self.reservation_forced_stay_count = 0
        self.reservation_masked_candidate_count = 0
        self.selected_soft_occupancy_sum = 0.0
        self.raw_soft_occupancy_sum = 0.0
        self.alternate_desirability_gap_sum = 0.0
        self.alternate_distance_gap_sum = 0.0
        self.alternate_urgency_gap_sum = 0.0
        self._current_observation_by_agent = {}
        self._global_graph = None
        self._base_get_global_state = self.world.get_global_state
        self.world.get_global_state = self._get_augmented_global_state
        base_obs_mcs = self.world.obs_builder.obs_mcs

        def fixed_width_obs_mcs(mcs, all_fcss=None):
            observation = base_obs_mcs(mcs, all_fcss)
            return self.builder.augment_unseen_observation(observation)

        # This adapter changes only the policy observation width.  Simulator
        # state transitions, matching and reward code are not touched.
        self.world.obs_builder.obs_mcs = fixed_width_obs_mcs
        self.__class__.instances[self.seed] = self

    def _augment(self, observations):
        agents = list(self.world.agents)
        by_agent = {
            agent: observations[index]
            for index, agent in enumerate(agents) if index < len(observations)
        }
        started = time.perf_counter()
        graph, row = self.builder.build(self.world, by_agent)
        row['hg_build_time_ms'] = (time.perf_counter() - started) * 1000.0
        self.graph_rows.append(row)
        self._global_graph = graph
        self._current_observation_by_agent = by_agent
        return observations

    def _get_augmented_global_state(self):
        base = np.asarray(self._base_get_global_state(), np.float32)
        if self._global_graph is None:
            raise RuntimeError('v12 global graph requested before observation augmentation')
        return np.concatenate((base, self._global_graph)).astype(np.float32)

    def reset(self):
        return self._augment(super().reset())

    def _record_actions(self, action_n):
        target_ids, representative_ids = [], []
        raw_top1_ids, raw_sample_ids = [], []
        for agent, action in zip(list(self.world.agents), action_n):
            if not isinstance(agent, MCS) or action.get('requested_mode') != 'Serve':
                continue
            candidate_id = int(action.get('low_candidate_id', -1))
            observation = self._current_observation_by_agent[agent]
            candidate_ids = np.asarray(observation['candidate_ids'], np.int64)
            if candidate_id >= 0:
                target_ids.append(candidate_id)
                index = int(np.flatnonzero(candidate_ids == candidate_id)[0])
                representative_ids.append(int(
                    observation['competition_representative_ids'][index]
                ))

            if 'reservation_raw_top1_id' not in observation:
                continue
            self.reservation_decision_count += 1
            raw_top1_id = int(observation['reservation_raw_top1_id'])
            raw_sample_id = int(observation['reservation_raw_sample_id'])
            if raw_top1_id >= 0:
                raw_top1_ids.append(raw_top1_id)
            if raw_sample_id >= 0:
                raw_sample_ids.append(raw_sample_id)
            reassigned = bool(observation.get('reservation_reassigned', False))
            self.reservation_reassignment_count += int(reassigned)
            self.reservation_forced_stay_count += int(
                observation.get('reservation_forced_stay', False)
            )
            self.reservation_masked_candidate_count += int(
                observation.get('reservation_masked_count', 0)
            )
            candidates = np.asarray(observation['low_candidates'], np.float32)

            def candidate_index(qid):
                matches = np.flatnonzero(candidate_ids == int(qid))
                return int(matches[0]) if matches.size else None

            selected_index = candidate_index(candidate_id)
            raw_index = candidate_index(raw_sample_id)
            if selected_index is not None:
                self.selected_soft_occupancy_sum += float(
                    candidates[selected_index, SOFT_OCCUPANCY_OFFSET]
                )
            if raw_index is not None:
                self.raw_soft_occupancy_sum += float(
                    candidates[raw_index, SOFT_OCCUPANCY_OFFSET]
                )
            if reassigned and selected_index is not None and raw_index is not None:
                desirability = np.asarray(
                    observation.get(
                        'candidate_desirabilities',
                        np.zeros(len(candidate_ids), np.float32),
                    ), np.float32,
                )
                urgency = np.asarray(
                    observation.get(
                        'candidate_urgencies',
                        np.zeros(len(candidate_ids), np.float32),
                    ), np.float32,
                )
                self.alternate_desirability_gap_sum += float(
                    desirability[selected_index] - desirability[raw_index]
                )
                self.alternate_distance_gap_sum += float(
                    candidates[selected_index, 1] - candidates[raw_index, 1]
                )
                self.alternate_urgency_gap_sum += float(
                    urgency[selected_index] - urgency[raw_index]
                )

        def conflict_counts(values):
            counts = {value: values.count(value) for value in set(values)}
            return sum(counts[value] > 1 for value in values), len(values)

        raw_top1_numerator, raw_top1_denominator = conflict_counts(raw_top1_ids)
        raw_sample_numerator, raw_sample_denominator = conflict_counts(raw_sample_ids)
        self.raw_top1_conflict_numerator += raw_top1_numerator
        self.raw_top1_conflict_denominator += raw_top1_denominator
        self.raw_sample_conflict_numerator += raw_sample_numerator
        self.raw_sample_conflict_denominator += raw_sample_denominator
        self.target_conflict_denominator += len(target_ids)
        counts = {value: target_ids.count(value) for value in set(target_ids)}
        self.target_conflict_numerator += sum(
            counts[value] > 1 for value in target_ids
        )
        rep_counts = {
            value: representative_ids.count(value)
            for value in set(representative_ids) if value >= 0
        }
        self.cluster_conflict_numerator += sum(
            value >= 0 and rep_counts.get(value, 0) > 1
            for value in representative_ids
        )

    def _record_spatial_conflicts(self):
        active = [
            mcs for mcs in self.world.MCSs
            if not (mcs.is_broken or mcs.is_energy_stranded)
        ]
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                self.spatial_conflict_numerator += int(
                    euclidean_distance(*left.pos, *right.pos) / 1000.0 <= 3.0
                )
                self.spatial_conflict_denominator += 1

    def step(self, action_n):
        self._record_actions(action_n)
        new_obs, old_obs, rewards, dones = super().step(action_n)
        self._record_spatial_conflicts()
        return self._augment(new_obs), old_obs, rewards, dones

    def metrics(self):
        result = {}
        if self.graph_rows:
            for key in self.graph_rows[0]:
                values = np.asarray([row[key] for row in self.graph_rows], np.float64)
                result[f'{key}_episode_mean'] = float(values.mean())
                result[f'{key}_episode_max'] = float(values.max())
        result.update({
            'target_conflict_rate': self.target_conflict_numerator / max(self.target_conflict_denominator, 1),
            'cluster_target_conflict_rate': self.cluster_conflict_numerator / max(self.target_conflict_denominator, 1),
            'spatial_conflict_rate': self.spatial_conflict_numerator / max(self.spatial_conflict_denominator, 1),
            'raw_top1_target_conflict_rate': self.raw_top1_conflict_numerator / max(self.raw_top1_conflict_denominator, 1),
            'raw_sample_target_conflict_rate': self.raw_sample_conflict_numerator / max(self.raw_sample_conflict_denominator, 1),
            'reservation_reassignment_rate': self.reservation_reassignment_count / max(self.reservation_decision_count, 1),
            'stay_due_to_reservation_rate': self.reservation_forced_stay_count / max(self.reservation_decision_count, 1),
            'reservation_masked_candidates_per_decision': self.reservation_masked_candidate_count / max(self.reservation_decision_count, 1),
            'selected_soft_occupancy_mean': self.selected_soft_occupancy_sum / max(self.reservation_decision_count, 1),
            'raw_soft_occupancy_mean': self.raw_soft_occupancy_sum / max(self.reservation_decision_count, 1),
            'alternate_desirability_gap_mean': self.alternate_desirability_gap_sum / max(self.reservation_reassignment_count, 1),
            'alternate_distance_ratio_gap_mean': self.alternate_distance_gap_sum / max(self.reservation_reassignment_count, 1),
            'alternate_urgency_gap_mean': self.alternate_urgency_gap_sum / max(self.reservation_reassignment_count, 1),
            'reservation_decision_count': self.reservation_decision_count,
            'reservation_reassignment_count': self.reservation_reassignment_count,
            'reservation_forced_stay_count': self.reservation_forced_stay_count,
            'v12_policy_candidate_dim': int(V12_CANDIDATE_DIM),
        })
        return result


_collect_episode = base_train.collect_episode


def collect_episode_with_hg_metrics(agent, args, episode):
    result = _collect_episode(agent, args, episode)
    seed = int(args.seed + episode - 1)
    environment = V12MultiAgentEnv.instances.pop(seed)
    result[0].update(environment.metrics())
    return result


def critic_input_with_original_state(global_state, high_state):
    """High bookkeeping stays fixed-width although Low Critic sees global HG."""
    value = np.concatenate((
        np.asarray(global_state, np.float32)[:MCS_GLOBAL_STATE_DIM],
        np.asarray(high_state, np.float32),
    )).astype(np.float32)
    if value.shape != (MCS_CRITIC_STATE_DIM,):
        raise RuntimeError(f'unexpected High bookkeeping shape: {value.shape}')
    return value


_ppo_update = base_train.ppo_update


def ppo_update_with_hg_metrics(*args, **kwargs):
    metrics = _ppo_update(*args, **kwargs)
    agent = args[0]
    metrics.update(agent.pop_training_diagnostics())
    return metrics


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--fixed-high-threshold' not in arguments:
        arguments.append('--fixed-high-threshold')
    if '--output-dir' not in arguments:
        arguments.extend([
            '--output-dir', str(Path(__file__).resolve().parent.parent / 'training_results_v12')
        ])
    sys.argv = [sys.argv[0], *arguments]
    base_train.MultiAgentEnv = V12MultiAgentEnv
    base_train.MCSMAPPOAgent = TrainV12Agent
    base_train.collect_episode = collect_episode_with_hg_metrics
    base_train.critic_input = critic_input_with_original_state
    base_train.ppo_update = ppo_update_with_hg_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_v12_metadata():
        args = original_parse_args()
        args.hg_model = 'B3_competition_hypergraph'
        args.hg_r_merge_km = HG_CONFIG.r_merge_km
        args.hg_jaccard_eta = HG_CONFIG.jaccard_eta
        args.hg_hidden_dim = HG_CONFIG.hidden_dim
        args.hg_actor_global_visibility = False
        args.hg_actor_critic_share_encoder = False
        args.hg_auxiliary_loss = False
        args.competition_reward_shaping = False
        args.hg_local_soft_occupancy = True
        args.hg_local_sequential_reservation = True
        args.hg_reservation_scope = 'exact_quasi_within_actor_local_visibility'
        args.hg_soft_occupancy_definition = '1-product(1-local_competitor_probability)'
        args.hg_feature_schema_version = 2
        args.hg_aggregation = 'relation_aware_gated_sum'
        args.patrol_enabled = False
        return args

    base_train.parse_args = parse_args_with_v12_metadata
    base_train.main()


if __name__ == '__main__':
    main()
