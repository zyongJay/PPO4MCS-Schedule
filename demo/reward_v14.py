"""Outcome-based reward for the v14 graph-representation experiment.

This module deliberately leaves :mod:`reward` unchanged.  It removes the
hand-designed *spatial* teaching signals from the Low policy reward so that
the GNN must obtain demand and competition information from node states and
edges instead:

* no hand-crafted attraction / competition / desirability difference reward;
* no hand-crafted candidate priority made from urgency and local IEV density;
* no spatial-opportunity term in the voluntary-wait penalty.

The retained Low signal is limited to observable consequences of a decision:
actual MCS service success, attributed controllable failure, realised MCS
profit, movement energy, safety events, and non-spatial persistence of a
voluntary stay.  Counterfactual responsibility remains solely as delayed
credit assignment for an actually failed/successful service; it is not a
candidate-ranking feature or an attraction/competition reward.
"""
from __future__ import annotations

import numpy as np

from config import MAX_MOVE_PER_STEP, POWER_UNIT
from reward import (
    EPSILON,
    FORCED_WAIT_TIME_PENALTY,
    LOW_EVENT_REWARD_WEIGHT,
    LOW_STEP_REWARD_WEIGHT,
    SERVE_MOVE_PENALTY,
    WAIT_BASE_PENALTY,
    WAIT_STREAK_NORMALIZER,
    WAIT_STREAK_PENALTY,
    RewardBuilder,
)


class V14OutcomeRewardBuilder(RewardBuilder):
    """v14 reward with all manually designed spatial terms disabled."""

    @staticmethod
    def compute_low_candidate_priority_reward(*args, **kwargs) -> float:
        """Disable the v10 hand-crafted urgency/attraction ranking signal."""
        del args, kwargs
        return 0.0

    @staticmethod
    def compute_mcs_spatial_features(mcs, all_evs, all_mcss, all_fcss):
        """Provide a zero-valued compatibility schema; never score space."""
        del mcs, all_evs, all_mcss, all_fcss
        return {
            'attraction': 0.0,
            'immediate_iev_demand': 0.0,
            'mcs_competition': 0.0,
            'fcs_competition': 0.0,
            'competition': 0.0,
            'potential': 0.0,
        }

    @staticmethod
    def compute_low_spatial_features(mcs, all_evs, all_fcss):
        """Provide a zero-valued compatibility schema; never score space."""
        del mcs, all_evs, all_fcss
        return {
            'attraction': 0.0,
            'immediate_iev_attraction': 0.0,
            'mcs_competition': 0.0,
            'fcs_competition': 0.0,
            'desirability': 0.0,
        }

    @staticmethod
    def compute_best_available_serve_potential(mcs, all_mcss, all_fcss):
        """Remove the hand-crafted spatial opportunity from wait shaping."""
        del mcs, all_mcss, all_fcss
        return 0.0

    @staticmethod
    def _nonspatial_wait_penalty(event: dict) -> tuple[float, float, float, float]:
        """Return wait total, base, streak and forced-time components.

        The old opportunity component depended on a manually constructed best
        candidate desirability.  A voluntary stay still has a small temporal
        cost, but it does not depend on what hand-crafted score another
        location receives.
        """
        low_stay_selected = bool(event.get('low_stay_selected', False))
        is_wait = str(event.get('requested_mode', '')) == 'Wait'
        forced_wait = bool(event.get('forced_wait', False)) and (
            low_stay_selected or is_wait
        )
        voluntary_wait = bool(event.get('voluntary_wait', False)) and (
            low_stay_selected or is_wait
        )
        duration = max(int(event.get('wait_duration_steps', 1)), 0)
        streak = max(int(event.get('consecutive_voluntary_wait_steps', 0)), 0)

        if forced_wait:
            forced_time = -FORCED_WAIT_TIME_PENALTY * duration
            return float(forced_time), 0.0, 0.0, float(forced_time)
        if not voluntary_wait:
            return 0.0, 0.0, 0.0, 0.0

        duration = max(duration, 1)
        base = -WAIT_BASE_PENALTY * duration
        streak_penalty = -WAIT_STREAK_PENALTY * min(
            streak / max(float(WAIT_STREAK_NORMALIZER), 1.0), 1.0
        ) * duration
        return float(base + streak_penalty), float(base), float(streak_penalty), 0.0

    def compute_mcs_reward(self, mcs, event):
        """Keep outcome rewards while replacing Low spatial shaping by zero."""
        components = super().compute_mcs_reward(mcs, event)

        is_serve = str(event.get('requested_mode', '')) == 'Serve'
        low_stay_selected = bool(event.get('low_stay_selected', False))
        wait_step, wait_base, wait_streak, wait_forced_time = (
            self._nonspatial_wait_penalty(event)
        )

        movement_cost = 0.0
        if is_serve and not low_stay_selected:
            movement_energy = max(float(event.get('movement_energy_kwh', 0.0)), 0.0)
            max_move_energy = max(
                (float(MAX_MOVE_PER_STEP) / 1000.0) * float(POWER_UNIT),
                EPSILON,
            )
            movement_ratio = float(np.clip(
                movement_energy / max_move_energy, 0.0, 1.0
            ))
            movement_cost = -SERVE_MOVE_PENALTY * movement_ratio

        # A Low decision is emitted only while serving.  Its immediate term is
        # now either real movement energy or a non-spatial voluntary-stay cost.
        if is_serve:
            low_step = wait_step if low_stay_selected else movement_cost
        else:
            low_step = 0.0

        low_event = float(components['low_event'])
        low_total = float(
            LOW_STEP_REWARD_WEIGHT * low_step
            + LOW_EVENT_REWARD_WEIGHT * low_event
        )

        components.update({
            'serve_step': float(movement_cost),
            'wait_step': float(wait_step),
            'wait_base': float(wait_base),
            'wait_opportunity': 0.0,
            'wait_streak': float(wait_streak),
            'wait_forced_time': float(wait_forced_time),
            'movement': float(movement_cost),
            'wait': float(wait_step),
            'low_step': float(low_step),
            'low_total': float(low_total),
            'low_step_weighted': float(LOW_STEP_REWARD_WEIGHT * low_step),
            'low_event_weighted': float(LOW_EVENT_REWARD_WEIGHT * low_event),
            # Explicit zeroes make the training log auditable.
            'position_gain': 0.0,
            'resource_gap_improvement': 0.0,
            'low_spatial_opportunity_gain': 0.0,
            'low_candidate_priority_reward': 0.0,
            'low_attraction_reward': 0.0,
            'low_immediate_iev_attraction': 0.0,
            'low_attraction_metric': 0.0,
            'low_mcs_competition_metric': 0.0,
            'low_fcs_competition_metric': 0.0,
            'urgency_coverage': 0.0,
            'mcs_cluster_penalty': 0.0,
            'fcs_redundancy_penalty': 0.0,
            'serve_attraction': 0.0,
            'serve_competition': 0.0,
            'serve_potential_improvement': 0.0,
            'best_available_serve_potential': 0.0,
        })
        components['step_total'] = float(
            components['high_step'] + low_step
        )
        components['event_total'] = float(
            components['high_event'] + low_event
        )
        components['total'] = float(components['high_total'] + low_total)
        return {name: float(value) for name, value in components.items()}
