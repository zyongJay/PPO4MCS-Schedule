"""v12 fixed-threshold Low MAPPO with the B3 Competition Hypergraph."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from competition_hypergraph import (
    CompetitionActor, CompetitionCritic, CompetitionHypergraphBuilder,
    CompetitionHypergraphConfig, FEATURE_SCHEMA, V12_CANDIDATE_DIM,
    V12_GLOBAL_GRAPH_DIM, SOFT_OCCUPANCY_OFFSET, RESERVED_OFFSET,
)
from network import MCS_ACTION_NAMES
from observation import MCS_STAY_CANDIDATE_ID


class V12CompetitionHypergraphAgent:
    CHECKPOINT_VERSION = 2
    ARCHITECTURE = 'v12_b3_local_reservation_competition_hypergraph_low_mappo'
    # v12 uses the optional coordination layer below.  The v11 ablation
    # overrides this flag while retaining the same heterogeneous-hypergraph
    # actor and critic encoders.
    use_local_reservation = True

    def __init__(
        self, low_self_dim, original_global_state_dim, hidden_dim=128,
        hg_hidden_dim=64, low_actor_lr=3e-4, low_critic_lr=5e-4,
        recharge_threshold_kwh=40.0, battery_capacity_kwh=300.0,
        r_merge_km=1.0, jaccard_eta=0.5, device='cpu',
    ):
        self.device = torch.device(device)
        self.hidden_dim = int(hidden_dim)
        self.hg_hidden_dim = int(hg_hidden_dim)
        self.low_self_dim = int(low_self_dim)
        self.low_candidate_dim = int(V12_CANDIDATE_DIM)
        self.original_global_state_dim = int(original_global_state_dim)
        self.global_state_dim = self.original_global_state_dim + V12_GLOBAL_GRAPH_DIM
        self.recharge_threshold_kwh = float(recharge_threshold_kwh)
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.hg_config = CompetitionHypergraphConfig(
            r_merge_km=float(r_merge_km), jaccard_eta=float(jaccard_eta),
            hidden_dim=int(hg_hidden_dim),
        )
        self.graph_builder = CompetitionHypergraphBuilder(self.hg_config)
        self.low_actor = CompetitionActor(
            low_self_dim, hidden_dim, hg_hidden_dim
        ).to(self.device)
        self.low_critic = CompetitionCritic(
            original_global_state_dim, low_self_dim, hidden_dim, hg_hidden_dim
        ).to(self.device)
        self.low_optimizer = torch.optim.Adam(
            self.low_actor.parameters(), lr=low_actor_lr
        )
        self.low_critic_optimizer = torch.optim.Adam(
            self.low_critic.parameters(), lr=low_critic_lr
        )
        self.evaluation_low_candidate_dim = self.low_candidate_dim
        self.evaluation_inference_semantics = self.ARCHITECTURE
        self.deterministic_low_actions = False
        self._episode_graph_rows = []
        self._target_conflict_numerator = 0
        self._target_conflict_denominator = 0
        self._cluster_conflict_numerator = 0
        self._spatial_conflict_numerator = 0
        self._spatial_conflict_denominator = 0

    def train(self):
        self.low_actor.train()
        self.low_critic.train()

    def eval(self):
        self.low_actor.eval()
        self.low_critic.eval()

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def begin_episode(self):
        self._episode_graph_rows = []
        self._target_conflict_numerator = 0
        self._target_conflict_denominator = 0
        self._cluster_conflict_numerator = 0
        self._spatial_conflict_numerator = 0
        self._spatial_conflict_denominator = 0

    def prepare_hypergraph_step(
        self, world, observation_by_agent, global_state, record_statistics=True,
    ):
        started = time.perf_counter()
        global_graph, statistics = self.graph_builder.build(
            world, observation_by_agent
        )
        statistics['hg_build_time_ms'] = (
            time.perf_counter() - started
        ) * 1000.0
        if record_statistics:
            self._episode_graph_rows.append(statistics)
        augmented = np.concatenate((
            np.asarray(global_state, np.float32), global_graph,
        )).astype(np.float32)
        return augmented

    def augment_unseen_observation(self, observation):
        return self.graph_builder.augment_unseen_observation(observation)

    def record_selected_targets(self, mcs_actions):
        target_ids = [
            int(item['candidate_id']) for item in mcs_actions
            if int(item.get('candidate_id', -1)) >= 0
        ]
        reps = [
            int(item.get('representative_id', -1)) for item in mcs_actions
            if int(item.get('candidate_id', -1)) >= 0
        ]
        self._target_conflict_denominator += len(target_ids)
        counts = {value: target_ids.count(value) for value in set(target_ids)}
        self._target_conflict_numerator += sum(counts[value] > 1 for value in target_ids)
        rep_counts = {value: reps.count(value) for value in set(reps) if value >= 0}
        self._cluster_conflict_numerator += sum(
            value >= 0 and rep_counts.get(value, 0) > 1 for value in reps
        )

    def record_spatial_conflict(self, mcss, radius_km=3.0):
        active = [mcs for mcs in mcss if not (mcs.is_broken or mcs.is_energy_stranded)]
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                from core import euclidean_distance
                self._spatial_conflict_numerator += int(
                    euclidean_distance(*left.pos, *right.pos) / 1000.0 <= radius_km
                )
                self._spatial_conflict_denominator += 1

    def episode_hypergraph_metrics(self):
        rows = self._episode_graph_rows
        result = {}
        if rows:
            for key in rows[0]:
                values = [float(row[key]) for row in rows]
                result[f'{key}_episode_mean'] = float(np.mean(values))
                result[f'{key}_episode_max'] = float(np.max(values))
        result.update({
            'target_conflict_rate': self._target_conflict_numerator / max(self._target_conflict_denominator, 1),
            'cluster_target_conflict_rate': self._cluster_conflict_numerator / max(self._target_conflict_denominator, 1),
            'spatial_conflict_rate': self._spatial_conflict_numerator / max(self._spatial_conflict_denominator, 1),
        })
        return result

    def _threshold_action(self, observation):
        serve, recharge = np.asarray(observation['high_action_mask'], dtype=bool)
        if not serve and recharge:
            return 1
        remain_kwh = float(observation['high_state'][0]) * self.battery_capacity_kwh
        if recharge and remain_kwh < self.recharge_threshold_kwh:
            return 1
        if serve:
            return 0
        if recharge:
            return 1
        raise RuntimeError('MCS has no legal Serve/Recharge mode')

    @torch.no_grad()
    def select_high_actions_batch(self, observations):
        indices = [self._threshold_action(item) for item in observations]
        return [
            {'mode': MCS_ACTION_NAMES[index], 'high_action': index, 'high_log_prob': 0.0}
            for index in indices
        ]

    @torch.no_grad()
    def get_high_values_batch(self, critic_states):
        count = 1 if np.asarray(critic_states).ndim == 1 else len(critic_states)
        return np.zeros(count, dtype=np.float32)

    @torch.no_grad()
    def select_low_actions_batch(self, observations):
        if not observations:
            return []
        candidate_arrays = [
            np.asarray(item['low_candidates'], np.float32).copy()
            for item in observations
        ]
        for array in candidate_arrays:
            array[:, SOFT_OCCUPANCY_OFFSET] = 0.0
            array[:, RESERVED_OFFSET] = 0.0
        original_masks = np.stack([
            np.asarray(item['low_candidate_mask'], bool).copy()
            for item in observations
        ])
        candidate_ids = [
            np.asarray(item['candidate_ids'], np.int64) for item in observations
        ]
        states = self._tensor(np.stack([
            item['low_self_state'] for item in observations
        ]))
        original_mask_tensor = self._tensor(original_masks, dtype=torch.bool)

        if not self.use_local_reservation:
            # v11 ablation: use only the HG-augmented MAPPO policy.  Both
            # control features stay zero, the legal-action mask is unchanged,
            # and no candidate is reserved or reassigned in this step.
            candidates = self._tensor(np.stack(candidate_arrays))
            logits = self.low_actor.logits(states, candidates)
            masked_logits = logits.masked_fill(~original_mask_tensor, -1e9)
            if self.deterministic_low_actions:
                selected = masked_logits.argmax(dim=-1)
            else:
                gumbel_uniform = torch.rand_like(masked_logits).clamp_(
                    1e-7, 1.0 - 1e-7
                )
                selected = (masked_logits - torch.log(
                    -torch.log(gumbel_uniform)
                )).argmax(dim=-1)
            distribution = torch.distributions.Categorical(logits=masked_logits)
            log_probs = distribution.log_prob(selected).detach().cpu().numpy()
            selected = selected.detach().cpu().numpy().astype(np.int64)
            raw_top1 = masked_logits.argmax(dim=-1).detach().cpu().numpy()
            candidate_ids = [
                np.asarray(item['candidate_ids'], np.int64)
                for item in observations
            ]
            for row, observation in enumerate(observations):
                choice = int(selected[row])
                observation['low_candidates'] = candidate_arrays[row]
                observation['obs_tgt'] = candidate_arrays[row]
                observation['low_candidate_mask'] = original_masks[row]
                observation['reservation_raw_sample_id'] = int(
                    candidate_ids[row][choice]
                )
                observation['reservation_raw_top1_id'] = int(
                    candidate_ids[row][int(raw_top1[row])]
                )
                observation['reservation_selected_id'] = int(
                    candidate_ids[row][choice]
                )
                observation['reservation_reassigned'] = False
                observation['reservation_forced_stay'] = False
                observation['reservation_masked_count'] = 0
                observation['reservation_priority_rank'] = -1
            return [
                {'low_action': int(selected[row]), 'low_log_prob': float(log_probs[row])}
                for row in range(len(observations))
            ]

        # First pass: estimate every active local competitor's probability of
        # selecting each exact quasi without any current-step reservations.
        initial_candidates = self._tensor(np.stack(candidate_arrays))
        initial_logits = self.low_actor.logits(states, initial_candidates)
        initial_probs = torch.softmax(
            initial_logits.masked_fill(~original_mask_tensor, -1e9), dim=-1
        ).detach().cpu().numpy()
        mcs_ids = [int(item.get('v12_mcs_id', index)) for index, item in enumerate(observations)]
        row_by_mcs_id = {mcs_id: row for row, mcs_id in enumerate(mcs_ids)}
        candidate_index_by_qid = [
            {int(qid): index for index, qid in enumerate(ids) if int(qid) >= 0}
            for ids in candidate_ids
        ]
        for row, observation in enumerate(observations):
            members = np.asarray(
                observation.get(
                    'competition_local_member_ids',
                    np.full((len(candidate_ids[row]), 1), -1, np.int64),
                ),
                np.int64,
            )
            for index, qid in enumerate(candidate_ids[row]):
                if qid < 0 or not original_masks[row, index]:
                    continue
                no_claim_probability = 1.0
                for competitor_id in members[index]:
                    competitor_row = row_by_mcs_id.get(int(competitor_id))
                    if competitor_row is None:
                        continue
                    competitor_index = candidate_index_by_qid[competitor_row].get(int(qid))
                    if (
                        competitor_index is None
                        or not original_masks[competitor_row, competitor_index]
                    ):
                        continue
                    no_claim_probability *= 1.0 - float(
                        initial_probs[competitor_row, competitor_index]
                    )
                candidate_arrays[row][index, SOFT_OCCUPANCY_OFFSET] = (
                    1.0 - no_claim_probability
                )

        # The second pass is the actual policy.  A single set of Gumbel draws
        # couples the pre-reservation sample and final sample, so a reported
        # reassignment is caused by reservation rather than fresh randomness.
        soft_candidates = self._tensor(np.stack(candidate_arrays))
        soft_logits = self.low_actor.logits(states, soft_candidates)
        masked_soft_logits = soft_logits.masked_fill(~original_mask_tensor, -1e9)
        if self.deterministic_low_actions:
            gumbel = torch.zeros_like(masked_soft_logits)
        else:
            uniform = torch.rand_like(masked_soft_logits).clamp_(1e-7, 1.0 - 1e-7)
            gumbel = -torch.log(-torch.log(uniform))
        perturbed_logits = masked_soft_logits + gumbel
        raw_sample = perturbed_logits.argmax(dim=-1).detach().cpu().numpy()
        raw_top1 = masked_soft_logits.argmax(dim=-1).detach().cpu().numpy()

        def priority(row):
            ids = candidate_ids[row]
            dispatch = original_masks[row] & (ids >= 0)
            urgencies = np.asarray(
                observations[row].get('candidate_urgencies', np.zeros_like(ids, dtype=float)),
                np.float32,
            )
            maximum_urgency = float(np.max(urgencies[dispatch])) if np.any(dispatch) else -1.0
            minimum_distance = float(
                np.min(candidate_arrays[row][dispatch, 1])
            ) if np.any(dispatch) else float('inf')
            return (-maximum_urgency, minimum_distance, mcs_ids[row])

        reservations = {}
        selected = np.empty(len(observations), np.int64)
        log_probs = np.empty(len(observations), np.float32)
        for row in sorted(range(len(observations)), key=priority):
            effective_mask = original_masks[row].copy()
            member_matrix = np.asarray(
                observations[row].get(
                    'competition_local_member_ids',
                    np.full((len(candidate_ids[row]), 1), -1, np.int64),
                ),
                np.int64,
            )
            reserved_indices = []
            for index, qid in enumerate(candidate_ids[row]):
                reserver = reservations.get(int(qid))
                if (
                    qid >= 0 and reserver is not None
                    and reserver in set(int(value) for value in member_matrix[index] if value >= 0)
                ):
                    effective_mask[index] = False
                    candidate_arrays[row][index, RESERVED_OFFSET] = 1.0
                    reserved_indices.append(index)
            if not np.any(effective_mask):
                stay_indices = np.flatnonzero(
                    candidate_ids[row] == MCS_STAY_CANDIDATE_ID
                )
                if stay_indices.size == 0:
                    raise RuntimeError('Reservation removed every action and no Stay exists')
                effective_mask[int(stay_indices[0])] = True

            final_mask = self._tensor(effective_mask[None, ...], dtype=torch.bool)
            # Candidate scoring is row-local.  Reservation only masks occupied
            # targets, so logits already computed for the unreserved candidates
            # can be reused exactly.
            final_logits = soft_logits[row]
            final_masked_logits = final_logits.masked_fill(~final_mask[0], -1e9)
            choice = int((final_masked_logits + gumbel[row]).argmax().item())
            distribution = torch.distributions.Categorical(logits=final_masked_logits)
            selected[row] = choice
            log_probs[row] = float(distribution.log_prob(
                torch.as_tensor(choice, device=self.device)
            ).item())
            selected_qid = int(candidate_ids[row][choice])
            if selected_qid >= 0:
                reservations[selected_qid] = mcs_ids[row]

            raw_index = int(raw_sample[row])
            raw_top1_index = int(raw_top1[row])
            raw_qid = int(candidate_ids[row][raw_index])
            selected_is_reservation_stay = bool(
                raw_qid >= 0
                and selected_qid == MCS_STAY_CANDIDATE_ID
                and raw_index in reserved_indices
            )
            observation = observations[row]
            observation['low_candidates'] = candidate_arrays[row]
            observation['obs_tgt'] = candidate_arrays[row]
            observation['low_candidate_mask'] = effective_mask
            observation['reservation_raw_sample_id'] = raw_qid
            observation['reservation_raw_top1_id'] = int(
                candidate_ids[row][raw_top1_index]
            )
            observation['reservation_selected_id'] = selected_qid
            observation['reservation_reassigned'] = bool(raw_index != choice)
            observation['reservation_forced_stay'] = selected_is_reservation_stay
            observation['reservation_masked_count'] = len(reserved_indices)
            observation['reservation_priority_rank'] = int(
                sorted(range(len(observations)), key=priority).index(row)
            )

        return [
            {'low_action': int(selected[i]), 'low_log_prob': float(log_probs[i])}
            for i in range(len(observations))
        ]

    @torch.no_grad()
    def get_low_values_batch(self, global_states, self_states, candidates, masks):
        candidate_array = np.asarray(candidates, np.float32)
        if candidate_array.shape[-1] != self.low_candidate_dim:
            if candidate_array.shape[-1] != 6:
                raise RuntimeError(
                    'unexpected Competition HG bootstrap candidate dimension'
                )
            padding = [(0, 0)] * candidate_array.ndim
            padding[-1] = (0, self.low_candidate_dim - 6)
            candidate_array = np.pad(candidate_array, padding)
            candidate_array[..., 4:6] = 0.0
        return self.low_values(
            self._tensor(global_states), self._tensor(self_states),
            self._tensor(candidate_array),
            self._tensor(masks, dtype=torch.bool),
        ).detach().cpu().numpy()

    def evaluate_low(self, self_states, candidates, masks, actions):
        distribution = self.low_actor.distribution(
            self_states.to(self.device), candidates.to(self.device), masks.to(self.device)
        )
        actions = actions.to(self.device)
        return distribution.log_prob(actions), distribution.entropy()

    def low_values(self, global_states, self_states, candidates, masks):
        return self.low_critic(
            global_states.to(self.device), self_states.to(self.device),
            candidates.to(self.device), masks.to(self.device),
        )

    @staticmethod
    def _gradient_norm(module):
        squared = 0.0
        finite = True
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            finite = finite and bool(torch.isfinite(gradient).all().item())
            squared += float(torch.sum(gradient * gradient).item())
        return squared ** 0.5, finite

    def pop_training_diagnostics(self):
        actor_grad, actor_grad_finite = self._gradient_norm(
            self.low_actor.hg_encoder
        )
        critic_grad, critic_grad_finite = self._gradient_norm(
            self.low_critic.hg_encoder
        )
        result = {}
        result.update(self.low_actor.hg_encoder.pop_diagnostics('actor_hg'))
        result.update(self.low_actor.pop_logit_diagnostics())
        result.update(self.low_critic.hg_encoder.pop_diagnostics('critic_hg'))
        result.update({
            'actor_hg_grad_norm': float(actor_grad),
            'critic_hg_grad_norm': float(critic_grad),
            'actor_hg_grad_finite': int(actor_grad_finite),
            'critic_hg_grad_finite': int(critic_grad_finite),
            'actor_hg_parameter_count': int(sum(
                parameter.numel()
                for parameter in self.low_actor.hg_encoder.parameters()
            )),
            'critic_hg_parameter_count': int(sum(
                parameter.numel()
                for parameter in self.low_critic.hg_encoder.parameters()
            )),
        })
        return result

    def save(self, path, metadata=None):
        torch.save({
            'checkpoint_version': self.CHECKPOINT_VERSION,
            'architecture': self.ARCHITECTURE,
            'hidden_dim': self.hidden_dim,
            'hg_hidden_dim': self.hg_hidden_dim,
            'low_self_dim': self.low_self_dim,
            'low_candidate_dim': self.low_candidate_dim,
            'original_global_state_dim': self.original_global_state_dim,
            'global_state_dim': self.global_state_dim,
            'recharge_threshold_kwh': self.recharge_threshold_kwh,
            'battery_capacity_kwh': self.battery_capacity_kwh,
            'hg_config': asdict(self.hg_config),
            'feature_schema': FEATURE_SCHEMA,
            'low_actor': self.low_actor.state_dict(),
            'low_critic': self.low_critic.state_dict(),
            'low_optimizer': self.low_optimizer.state_dict(),
            'low_critic_optimizer': self.low_critic_optimizer.state_dict(),
            'metadata': metadata or {},
        }, Path(path))

    def load(self, path, load_optimizers=False):
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        if checkpoint.get('architecture') != self.ARCHITECTURE:
            raise ValueError('checkpoint is not v12 B3 Competition Hypergraph')
        if checkpoint.get('feature_schema') != FEATURE_SCHEMA:
            raise ValueError('v12 checkpoint feature schema mismatch')
        if int(checkpoint['low_candidate_dim']) != self.low_candidate_dim:
            raise ValueError('v12 checkpoint candidate dimension mismatch')
        self.low_actor.load_state_dict(checkpoint['low_actor'])
        self.low_critic.load_state_dict(checkpoint['low_critic'])
        if load_optimizers:
            self.low_optimizer.load_state_dict(checkpoint['low_optimizer'])
            self.low_critic_optimizer.load_state_dict(
                checkpoint['low_critic_optimizer']
            )
        return checkpoint.get('metadata', {})

    @classmethod
    def from_checkpoint(cls, path, device='cpu', load_optimizers=False):
        checkpoint = torch.load(Path(path), map_location='cpu', weights_only=False)
        if checkpoint.get('architecture') != cls.ARCHITECTURE:
            raise ValueError('checkpoint is not v12 B3 Competition Hypergraph')
        if checkpoint.get('feature_schema') != FEATURE_SCHEMA:
            raise ValueError('v12 checkpoint feature schema mismatch')
        config = checkpoint['hg_config']
        agent = cls(
            checkpoint['low_self_dim'], checkpoint['original_global_state_dim'],
            checkpoint['hidden_dim'], checkpoint['hg_hidden_dim'],
            recharge_threshold_kwh=checkpoint['recharge_threshold_kwh'],
            battery_capacity_kwh=checkpoint['battery_capacity_kwh'],
            r_merge_km=config['r_merge_km'], jaccard_eta=config['jaccard_eta'],
            device=device,
        )
        agent.low_actor.load_state_dict(checkpoint['low_actor'])
        agent.low_critic.load_state_dict(checkpoint['low_critic'])
        if load_optimizers:
            agent.low_optimizer.load_state_dict(checkpoint['low_optimizer'])
            agent.low_critic_optimizer.load_state_dict(checkpoint['low_critic_optimizer'])
        return agent, checkpoint.get('metadata', {})


class V11CompetitionHypergraphAgent(V12CompetitionHypergraphAgent):
    """B3 heterogeneous-hypergraph MAPPO without occupancy or reservation."""

    ARCHITECTURE = 'v11_b3_competition_hypergraph_low_mappo_no_reservation'
    use_local_reservation = False
