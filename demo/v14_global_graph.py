"""Fixed-threshold Low MAPPO agent for the v14 canonical global graph."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from config import MCS_BATTERY_CAPACITY, MCS_RECHARGE_THRESHOLD
from global_graph_v14 import V14GlobalGraphActor, V14GlobalGraphCritic
from network import MCS_ACTION_NAMES
from observation_v14 import (
    FEATURE_SCHEMA,
    V14_CANDIDATE_DIM,
    V14_LAYOUT,
    V14_SELF_DIM,
)


class V14GlobalGraphMAPPOAgent:
    """Shared Low Actor, centralized Low Critic, and no action correction."""

    CHECKPOINT_VERSION = 1
    ARCHITECTURE = 'v14_canonical_global_graph_local_aggregation_low_mappo'

    def __init__(
        self,
        original_global_state_dim: int,
        hidden_dim: int = 128,
        graph_hidden_dim: int = 16,
        graph_layers: int = 2,
        low_actor_lr: float = 3e-4,
        low_critic_lr: float = 5e-4,
        recharge_threshold_kwh: float = MCS_RECHARGE_THRESHOLD,
        battery_capacity_kwh: float = MCS_BATTERY_CAPACITY,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)
        self.hidden_dim = int(hidden_dim)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.graph_layers = int(graph_layers)
        self.original_global_state_dim = int(original_global_state_dim)
        self.low_self_dim = int(V14_SELF_DIM)
        self.low_candidate_dim = int(V14_CANDIDATE_DIM)
        self.recharge_threshold_kwh = float(recharge_threshold_kwh)
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.low_actor = V14GlobalGraphActor(
            hidden_dim, graph_hidden_dim, graph_layers
        ).to(self.device)
        self.low_critic = V14GlobalGraphCritic(
            original_global_state_dim,
            hidden_dim,
            graph_hidden_dim,
            graph_layers,
        ).to(self.device)
        self.low_optimizer = torch.optim.Adam(
            self.low_actor.parameters(), lr=low_actor_lr
        )
        self.low_critic_optimizer = torch.optim.Adam(
            self.low_critic.parameters(), lr=low_critic_lr
        )
        self.deterministic_low_actions = False

    def train(self):
        self.low_actor.train()
        self.low_critic.train()

    def eval(self):
        self.low_actor.eval()
        self.low_critic.eval()

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _threshold_action(self, observation):
        serve, recharge = np.asarray(
            observation['high_action_mask'], dtype=bool
        )
        if not serve and recharge:
            return 1
        remain_kwh = (
            float(observation['high_state'][0])
            * self.battery_capacity_kwh
        )
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
        return [{
            'mode': MCS_ACTION_NAMES[index],
            'high_action': int(index),
            'high_log_prob': 0.0,
        } for index in indices]

    @torch.no_grad()
    def get_high_values_batch(self, critic_states):
        array = np.asarray(critic_states)
        count = 1 if array.ndim == 1 else len(array)
        return np.zeros(count, np.float32)

    @torch.no_grad()
    def select_low_actions_batch(self, observations):
        if not observations:
            return []
        self_states = self._tensor(np.stack([
            item['low_self_state'] for item in observations
        ]))
        candidates = self._tensor(np.stack([
            item['low_candidates'] for item in observations
        ]))
        masks = self._tensor(np.stack([
            item['low_candidate_mask'] for item in observations
        ]), dtype=torch.bool)
        distribution = self.low_actor.distribution(
            self_states, candidates, masks
        )
        actions = (
            distribution.probs.argmax(dim=-1)
            if self.deterministic_low_actions
            else distribution.sample()
        )
        log_probs = distribution.log_prob(actions)
        action_values = actions.detach().cpu().numpy()
        log_prob_values = log_probs.detach().cpu().numpy()
        return [{
            'low_action': int(action_values[index]),
            'low_log_prob': float(log_prob_values[index]),
        } for index in range(len(observations))]

    @torch.no_grad()
    def get_low_values_batch(
        self, global_states, self_states, candidates, masks
    ):
        return self.low_values(
            self._tensor(global_states),
            self._tensor(self_states),
            self._tensor(candidates),
            self._tensor(masks, dtype=torch.bool),
        ).detach().cpu().numpy()

    def evaluate_low(self, self_states, candidates, masks, actions):
        distribution = self.low_actor.distribution(
            self_states.to(self.device),
            candidates.to(self.device),
            masks.to(self.device),
        )
        actions = actions.to(self.device)
        return distribution.log_prob(actions), distribution.entropy()

    def low_values(self, global_states, self_states, candidates, masks):
        return self.low_critic(
            global_states.to(self.device),
            self_states.to(self.device),
            candidates.to(self.device),
            masks.to(self.device),
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

    def snapshot_graph_parameters(self):
        """Small CPU snapshot used to prove an optimizer update occurred."""
        return {
            'actor': [
                parameter.detach().cpu().clone()
                for parameter in self.low_actor.graph_encoder.parameters()
            ],
            'critic': [
                parameter.detach().cpu().clone()
                for parameter in self.low_critic.graph_encoder.parameters()
            ],
        }

    @staticmethod
    def _parameter_update_metrics(module, before, prefix):
        delta_squared = 0.0
        parameter_squared = 0.0
        changed = 0
        total = 0
        finite = True
        for parameter, old_value in zip(module.parameters(), before):
            current = parameter.detach().cpu()
            delta = current - old_value
            finite = finite and bool(torch.isfinite(current).all().item())
            delta_squared += float(torch.sum(delta * delta).item())
            parameter_squared += float(torch.sum(current * current).item())
            changed += int(torch.count_nonzero(delta).item())
            total += int(delta.numel())
        delta_norm = delta_squared ** 0.5
        parameter_norm = parameter_squared ** 0.5
        return {
            f'{prefix}_parameter_update_norm': float(delta_norm),
            f'{prefix}_relative_parameter_update_norm': float(
                delta_norm / max(parameter_norm, 1e-12)
            ),
            f'{prefix}_parameter_changed_fraction': float(
                changed / max(total, 1)
            ),
            f'{prefix}_parameter_finite': int(finite),
        }

    def graph_parameter_update_diagnostics(self, snapshot):
        result = self._parameter_update_metrics(
            self.low_actor.graph_encoder,
            snapshot['actor'],
            'actor_graph',
        )
        result.update(self._parameter_update_metrics(
            self.low_critic.graph_encoder,
            snapshot['critic'],
            'critic_graph',
        ))
        return result

    @staticmethod
    def _edge_ablated_states(self_states):
        # Keep current-MCS addressing and every primitive node feature, but
        # remove all five relation matrices.  This isolates message passing
        # from type-specific node encoders and the unchanged candidate fields.
        edge_start = 1 + (
            V14_LAYOUT.mcs_features
            + V14_LAYOUT.fcs_features
            + V14_LAYOUT.ev_features
        )
        ablated = self_states.clone()
        ablated[:, edge_start:] = 0.0
        return ablated

    @torch.no_grad()
    def graph_counterfactual_diagnostics(
        self,
        low_buffer,
        clip_ratio: float,
        max_samples: int = 256,
    ):
        """Compare the trained model with the same samples but no graph edges.

        Positive surrogate/MSE gains indicate that edge messages help the
        current PPO objective and value fit respectively.  They are online
        diagnostics, not a replacement for held-out v10-onlylow evaluation.
        """
        transitions = list(low_buffer.transitions)
        if not transitions:
            return {'graph_counterfactual_sample_count': 0}
        sample_count = min(len(transitions), int(max_samples))
        indices = np.linspace(
            0, len(transitions) - 1, sample_count, dtype=np.int64
        )
        selected = [transitions[int(index)] for index in indices]
        self_states = self._tensor(np.stack([
            item.low_self_state for item in selected
        ]))
        candidates = self._tensor(np.stack([
            item.low_candidates for item in selected
        ]))
        masks = self._tensor(np.stack([
            item.low_mask for item in selected
        ]), dtype=torch.bool)
        actions = self._tensor(
            np.asarray([item.low_action for item in selected]),
            dtype=torch.long,
        )
        old_log_probs = self._tensor(np.asarray([
            item.low_log_prob for item in selected
        ]))
        advantages = self._tensor(np.asarray([
            item.advantage for item in selected
        ]))
        returns = self._tensor(np.asarray([
            item.return_target for item in selected
        ]))
        global_states = self._tensor(np.stack([
            item.global_state for item in selected
        ]))
        ablated_states = self._edge_ablated_states(self_states)

        full_distribution = self.low_actor.distribution(
            self_states, candidates, masks
        )
        ablated_distribution = self.low_actor.distribution(
            ablated_states, candidates, masks
        )
        full_prob = full_distribution.probs
        ablated_prob = ablated_distribution.probs
        midpoint = 0.5 * (full_prob + ablated_prob)
        epsilon = torch.finfo(full_prob.dtype).eps
        js_divergence = 0.5 * (
            (full_prob * (
                full_prob.clamp_min(epsilon).log()
                - midpoint.clamp_min(epsilon).log()
            )).sum(dim=-1)
            + (ablated_prob * (
                ablated_prob.clamp_min(epsilon).log()
                - midpoint.clamp_min(epsilon).log()
            )).sum(dim=-1)
        )
        full_log_prob = full_distribution.log_prob(actions)
        ablated_log_prob = ablated_distribution.log_prob(actions)
        normalized_advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        def clipped_surrogate(log_prob):
            ratio = torch.exp(log_prob - old_log_probs)
            unclipped = ratio * normalized_advantages
            clipped = torch.clamp(
                ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * normalized_advantages
            return torch.minimum(unclipped, clipped).mean()

        full_surrogate = clipped_surrogate(full_log_prob)
        ablated_surrogate = clipped_surrogate(ablated_log_prob)
        full_values = self.low_values(
            global_states, self_states, candidates, masks
        )
        ablated_values = self.low_values(
            global_states, ablated_states, candidates, masks
        )
        full_mse = torch.mean((full_values - returns) ** 2)
        ablated_mse = torch.mean((ablated_values - returns) ** 2)
        target_variance = torch.var(returns, unbiased=False)

        def explained_variance(predictions):
            if float(target_variance.item()) < 1e-12:
                return torch.zeros((), device=returns.device)
            return 1.0 - torch.var(
                returns - predictions, unbiased=False
            ) / target_variance

        return {
            'graph_counterfactual_sample_count': int(sample_count),
            'actor_graph_edge_ablation_js_divergence': float(
                js_divergence.mean().item()
            ),
            'actor_graph_edge_ablation_top1_flip_rate': float((
                full_prob.argmax(dim=-1) != ablated_prob.argmax(dim=-1)
            ).float().mean().item()),
            'actor_graph_edge_selected_probability_gain': float((
                full_prob.gather(1, actions.unsqueeze(1)).squeeze(1)
                - ablated_prob.gather(1, actions.unsqueeze(1)).squeeze(1)
            ).mean().item()),
            'actor_graph_edge_advantage_alignment': float((
                normalized_advantages
                * (full_log_prob - ablated_log_prob)
            ).mean().item()),
            'actor_graph_edge_surrogate_gain': float(
                (full_surrogate - ablated_surrogate).item()
            ),
            'critic_graph_edge_value_mae': float(
                torch.mean(torch.abs(full_values - ablated_values)).item()
            ),
            'critic_graph_edge_mse_gain': float(
                (ablated_mse - full_mse).item()
            ),
            'critic_graph_edge_explained_variance_gain': float((
                explained_variance(full_values)
                - explained_variance(ablated_values)
            ).item()),
        }

    def pop_training_diagnostics(self):
        actor_grad, actor_finite = self._gradient_norm(
            self.low_actor.graph_encoder
        )
        critic_grad, critic_finite = self._gradient_norm(
            self.low_critic.graph_encoder
        )
        result = self.low_actor.pop_diagnostics()
        result.update(self.low_critic.graph_encoder.pop_diagnostics(
            'critic_graph'
        ))
        result.update({
            'actor_graph_grad_norm': float(actor_grad),
            'critic_graph_grad_norm': float(critic_grad),
            'actor_graph_grad_finite': int(actor_finite),
            'critic_graph_grad_finite': int(critic_finite),
            'actor_graph_parameter_count': int(sum(
                parameter.numel()
                for parameter in self.low_actor.graph_encoder.parameters()
            )),
            'critic_graph_parameter_count': int(sum(
                parameter.numel()
                for parameter in self.low_critic.graph_encoder.parameters()
            )),
            'v14_graph_state_dim': int(V14_LAYOUT.graph_dim),
            'v14_policy_self_dim': int(V14_SELF_DIM),
            'v14_policy_candidate_dim': int(V14_CANDIDATE_DIM),
        })
        return result

    def save(self, path, metadata=None):
        torch.save({
            'checkpoint_version': self.CHECKPOINT_VERSION,
            'architecture': self.ARCHITECTURE,
            'hidden_dim': self.hidden_dim,
            'graph_hidden_dim': self.graph_hidden_dim,
            'graph_layers': self.graph_layers,
            'original_global_state_dim': self.original_global_state_dim,
            'low_self_dim': self.low_self_dim,
            'low_candidate_dim': self.low_candidate_dim,
            'recharge_threshold_kwh': self.recharge_threshold_kwh,
            'battery_capacity_kwh': self.battery_capacity_kwh,
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
        self._validate_checkpoint(checkpoint)
        self.low_actor.load_state_dict(checkpoint['low_actor'])
        self.low_critic.load_state_dict(checkpoint['low_critic'])
        if load_optimizers:
            self.low_optimizer.load_state_dict(checkpoint['low_optimizer'])
            self.low_critic_optimizer.load_state_dict(
                checkpoint['low_critic_optimizer']
            )
        return checkpoint.get('metadata', {})

    @classmethod
    def _validate_checkpoint(cls, checkpoint):
        if checkpoint.get('architecture') != cls.ARCHITECTURE:
            raise ValueError('checkpoint is not a v14 canonical global graph')
        if checkpoint.get('feature_schema') != FEATURE_SCHEMA:
            raise ValueError('v14 checkpoint feature schema mismatch')
        if int(checkpoint['low_self_dim']) != int(V14_SELF_DIM):
            raise ValueError('v14 checkpoint self-state width mismatch')
        if int(checkpoint['low_candidate_dim']) != int(V14_CANDIDATE_DIM):
            raise ValueError('v14 checkpoint candidate width mismatch')

    @classmethod
    def from_checkpoint(cls, path, device='cpu', load_optimizers=False):
        checkpoint = torch.load(
            Path(path), map_location='cpu', weights_only=False
        )
        cls._validate_checkpoint(checkpoint)
        agent = cls(
            checkpoint['original_global_state_dim'],
            checkpoint['hidden_dim'],
            checkpoint['graph_hidden_dim'],
            checkpoint['graph_layers'],
            recharge_threshold_kwh=checkpoint[
                'recharge_threshold_kwh'
            ],
            battery_capacity_kwh=checkpoint['battery_capacity_kwh'],
            device=device,
        )
        agent.low_actor.load_state_dict(checkpoint['low_actor'])
        agent.low_critic.load_state_dict(checkpoint['low_critic'])
        if load_optimizers:
            agent.low_optimizer.load_state_dict(checkpoint['low_optimizer'])
            agent.low_critic_optimizer.load_state_dict(
                checkpoint['low_critic_optimizer']
            )
        return agent, checkpoint.get('metadata', {})
