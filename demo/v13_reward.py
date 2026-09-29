"""v13 system-profit reward adapter.

This module intentionally leaves ``reward.py`` unchanged.  It replaces only
the MCS-profit event term at runtime with a signed, per-MCS share of the
increment in total provider profit (all MCS plus all FCS).
"""
from __future__ import annotations

import numpy as np

from reward import (
    HIGH_EVENT_REWARD_WEIGHT,
    HIGH_PROFIT_EVENT_WEIGHT,
    LOW_EVENT_REWARD_WEIGHT,
    LOW_PROFIT_EVENT_WEIGHT,
    MCS_PROFIT_REFERENCE,
    RewardBuilder,
)


SYSTEM_TOTAL_PROFIT_REFERENCE_PER_MCS = float(MCS_PROFIT_REFERENCE)


class SystemProfitRewardBuilder(RewardBuilder):
    """Replace v10's individual-MCS profit event with system-profit feedback."""

    def compute_mcs_reward(self, mcs, event):
        components = super().compute_mcs_reward(mcs, event)

        # ``system_total_profit_share_delta`` is populated by V13MultiAgentEnv
        # after matching has completed and before the world assigns rewards.
        # Signed normalization makes a system-level loss a negative learning
        # signal instead of silently treating it as zero profit.
        system_delta = float(event.get('system_total_profit_delta', 0.0))
        system_share = float(event.get(
            'system_total_profit_share_delta', 0.0
        ))
        normalized_system_share = float(np.clip(
            system_share / SYSTEM_TOTAL_PROFIT_REFERENCE_PER_MCS,
            -1.0,
            1.0,
        ))
        has_low_responsibility = int(
            event.get('low_decision_id', -1)
        ) >= 0

        old_high_profit_event = float(components['high_profit_event'])
        old_low_profit_event = float(components['low_profit_event'])
        new_high_profit_event = (
            HIGH_PROFIT_EVENT_WEIGHT * normalized_system_share
        )
        new_low_profit_event = (
            LOW_PROFIT_EVENT_WEIGHT * normalized_system_share
            if has_low_responsibility else 0.0
        )

        # Preserve all safety, success, spatial, wait, and recharge terms from
        # v10.  Only the profit contribution is replaced.
        high_profit_delta = new_high_profit_event - old_high_profit_event
        low_profit_delta = new_low_profit_event - old_low_profit_event
        components['high_event'] = float(
            components['high_event'] + high_profit_delta
        )
        components['low_event'] = float(
            components['low_event'] + low_profit_delta
        )
        components['high_profit_event'] = float(new_high_profit_event)
        components['low_profit_event'] = float(new_low_profit_event)
        components['profit_delta'] = float(system_share)
        components['normalized_profit'] = float(normalized_system_share)
        components['system_total_profit_delta'] = float(system_delta)
        components['system_total_profit_share_delta'] = float(system_share)
        components['normalized_system_total_profit_share'] = float(
            normalized_system_share
        )
        components['high_total'] = float(
            components['high_total']
            + HIGH_EVENT_REWARD_WEIGHT * high_profit_delta
        )
        components['high_current_option_total'] = float(
            components['high_total']
            - components['high_attributed_success_weighted']
        )
        components['low_total'] = float(
            components['low_total']
            + LOW_EVENT_REWARD_WEIGHT * low_profit_delta
        )
        components['event_total'] = float(
            components['high_event'] + components['low_event']
        )
        components['total'] = float(
            components['high_total'] + components['low_total']
        )
        return components
