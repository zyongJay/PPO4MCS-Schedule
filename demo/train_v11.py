"""Train the v11 heterogeneous-hypergraph MAPPO reservation ablation.

v11 keeps v12's local Actor hypergraph, global Critic hypergraph, fixed 40 kWh
High rule, PPO setup, simulation, matcher and rewards.  It deliberately does
not use local soft occupancy or sequential target reservation: Low MAPPO acts
once from its unmodified legal candidate mask.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import train as base_train
from config import (
    MCS_BATTERY_CAPACITY, MCS_CRITIC_STATE_DIM, MCS_GLOBAL_STATE_DIM,
    MCS_RECHARGE_THRESHOLD,
)
from competition_hypergraph import V12_CANDIDATE_DIM
from train_v12 import HG_CONFIG, critic_input_with_original_state
from train_v12 import ppo_update_with_hg_metrics
from train_v12 import V12MultiAgentEnv
from v12_hypergraph import V11CompetitionHypergraphAgent


class TrainV11Agent(V11CompetitionHypergraphAgent):
    """Constructor-compatible v11 adapter for train.py's Low-only branch."""

    def __init__(
        self, high_state_dim, low_self_dim, low_candidate_dim,
        critic_state_dim, hidden_dim=128, actor_lr=3e-4,
        critic_lr=5e-4, low_actor_lr=None, low_critic_lr=None,
        device='cpu',
    ):
        del low_candidate_dim, actor_lr, critic_lr
        super().__init__(
            low_self_dim,
            int(critic_state_dim - high_state_dim),
            hidden_dim=hidden_dim,
            hg_hidden_dim=HG_CONFIG.hidden_dim,
            low_actor_lr=(low_actor_lr or 3e-4),
            low_critic_lr=(low_critic_lr or 5e-4),
            recharge_threshold_kwh=float(MCS_RECHARGE_THRESHOLD),
            battery_capacity_kwh=float(MCS_BATTERY_CAPACITY),
            r_merge_km=HG_CONFIG.r_merge_km,
            jaccard_eta=HG_CONFIG.jaccard_eta,
            device=device,
        )


class V11MultiAgentEnv(V12MultiAgentEnv):
    """The v12 observation adapter with separately labelled v11 metrics."""

    instances = {}

    def metrics(self):
        result = super().metrics()
        result['v11_policy_candidate_dim'] = result.pop(
            'v12_policy_candidate_dim'
        )
        return result


_collect_episode = base_train.collect_episode


def collect_episode_with_hg_metrics(agent, args, episode):
    result = _collect_episode(agent, args, episode)
    environment = V11MultiAgentEnv.instances.pop(int(args.seed + episode - 1))
    result[0].update(environment.metrics())
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--fixed-high-threshold' not in arguments:
        arguments.append('--fixed-high-threshold')
    if '--output-dir' not in arguments:
        arguments.extend([
            '--output-dir',
            str(Path(__file__).resolve().parent.parent / 'training_results_v11'),
        ])
    sys.argv = [sys.argv[0], *arguments]
    base_train.MultiAgentEnv = V11MultiAgentEnv
    base_train.MCSMAPPOAgent = TrainV11Agent
    base_train.collect_episode = collect_episode_with_hg_metrics
    base_train.critic_input = critic_input_with_original_state
    base_train.ppo_update = ppo_update_with_hg_metrics
    original_parse_args = base_train.parse_args

    def parse_args_with_v11_metadata():
        args = original_parse_args()
        args.hg_model = 'B3_competition_hypergraph_no_reservation'
        args.hg_r_merge_km = HG_CONFIG.r_merge_km
        args.hg_jaccard_eta = HG_CONFIG.jaccard_eta
        args.hg_hidden_dim = HG_CONFIG.hidden_dim
        args.hg_actor_global_visibility = False
        args.hg_actor_critic_share_encoder = False
        args.hg_auxiliary_loss = False
        args.competition_reward_shaping = False
        args.hg_local_soft_occupancy = False
        args.hg_local_sequential_reservation = False
        args.hg_reservation_scope = 'disabled'
        args.hg_soft_occupancy_definition = 'disabled'
        args.hg_feature_schema_version = 2
        args.hg_aggregation = 'relation_aware_gated_sum'
        args.patrol_enabled = False
        return args

    base_train.parse_args = parse_args_with_v11_metadata
    base_train.main()


if __name__ == '__main__':
    main()
