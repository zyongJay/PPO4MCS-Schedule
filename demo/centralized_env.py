"""Centralized fixed-step MCS dispatch environment.

This module is intentionally separate from :mod:`environment` and
:mod:`world`.  It keeps the existing EV/MCS/FCS entities, charging progress
and Hungarian matching, but replaces the old ``near_quasi`` MCS action with a
global fixed-grid dispatch action.

At the beginning of every physical step one centralized policy schedules all
eligible MCSs.  Dispatching, existing task progress and EV movement are then
advanced for one fixed interval, and immediate EV/provider matching is run at
the end of the interval.  A dispatching/charging/recharging MCS is a task and
does not receive a new policy action until that task completes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config import (
    AREA_LAT_MAX,
    AREA_LAT_MIN,
    AREA_LON_MAX,
    AREA_LON_MIN,
    CHARGE_PRICE,
    MAX_MOVE_PER_STEP,
    MAX_STEPS_PER_EPISODE,
    MCS_BATTERY_CAPACITY,
    MCS_RECHARGE_THRESHOLD,
    NUM_MCS,
    PG_PRICE,
    POWER_UNIT,
    RC_PRICE,
    STEP_DURATION_MIN,
)
from core import euclidean_distance
from matching import RechargeMatcher
from observation_v14 import GlobalGraphSnapshotBuilder
from world import World


@dataclass(frozen=True)
class CentralizedRewardConfig:
    """Outcome-only global reward; no hand-crafted spatial teaching signal."""

    success: float = 1.0
    failure: float = 1.0
    realised_profit: float = 0.01
    dispatch_cost: float = 0.01


def build_dispatch_points(rows: int, columns: int) -> np.ndarray:
    """Return fixed geometric cell centres in stable row-major order."""
    if rows <= 0 or columns <= 0:
        raise ValueError('dispatch grid rows and columns must be positive')
    lon_width = (AREA_LON_MAX - AREA_LON_MIN) / float(columns)
    lat_height = (AREA_LAT_MAX - AREA_LAT_MIN) / float(rows)
    points = []
    for row in range(rows):
        for column in range(columns):
            points.append([
                AREA_LON_MIN + (column + 0.5) * lon_width,
                AREA_LAT_MIN + (row + 0.5) * lat_height,
            ])
    return np.asarray(points, dtype=np.float64)


class CentralizedDispatchEnv:
    """One centralized-agent environment with persistent MCS dispatch tasks."""

    def __init__(
        self,
        seed: int = 42,
        grid_rows: int = 4,
        grid_columns: int = 4,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        recharge_threshold_kwh: float = MCS_RECHARGE_THRESHOLD,
        reward_config: CentralizedRewardConfig | None = None,
        verbose: bool = False,
    ):
        self.seed = int(seed)
        self.world = World(self.seed, verbose=verbose)
        self.dispatch_points = build_dispatch_points(grid_rows, grid_columns)
        self.dispatch_point_count = int(len(self.dispatch_points))
        self.stay_action = self.dispatch_point_count
        self.action_count = self.dispatch_point_count + 1
        self.max_steps = int(max_steps)
        self.recharge_threshold_kwh = float(recharge_threshold_kwh)
        self.reward_config = reward_config or CentralizedRewardConfig()
        self.recharge_matcher = RechargeMatcher()
        self.graph_builder = GlobalGraphSnapshotBuilder(self.world)
        self.pending_service_profit: Dict[int, float] = {}
        self.last_reward_components: Dict[str, float] = {}
        self.last_step_metrics: Dict[str, float] = {}
        self._map_diagonal_m = max(euclidean_distance(
            AREA_LON_MIN,
            AREA_LAT_MIN,
            AREA_LON_MAX,
            AREA_LAT_MAX,
        ), 1.0)
        self._initialise_centralized_fields()

    def _initialise_centralized_fields(self) -> None:
        for mcs in self.world.MCSs:
            mcs.centralized_is_dispatching = False
            mcs.centralized_dispatch_target_id = -1
            mcs.centralized_dispatch_target_pos = None
            mcs.centralized_dispatch_start_step = -1
            mcs.centralized_recharge_pending = False

    def reset(self) -> Dict[str, np.ndarray]:
        # Deliberately do not call match_and_get_neibor here.  The first
        # episode step must follow the same schedule -> progress -> match order
        # as every later step, so no reset-time matching is allowed.
        self.world.reset_world()
        self.world.agents.clear()
        self.world.last_agents.clear()
        self.pending_service_profit.clear()
        self.last_reward_components = {}
        self.last_step_metrics = {}
        self._initialise_centralized_fields()
        self.graph_builder = GlobalGraphSnapshotBuilder(self.world)
        return self.observe()

    @staticmethod
    def _is_dispatching(mcs) -> bool:
        return bool(getattr(mcs, 'centralized_is_dispatching', False))

    def _eligible_mask(self) -> np.ndarray:
        result = np.zeros(NUM_MCS, dtype=bool)
        for index, mcs in enumerate(self.world.MCSs[:NUM_MCS]):
            low_energy = float(mcs.remain) < self.recharge_threshold_kwh
            result[index] = bool(
                mcs.is_idle
                and not self._is_dispatching(mcs)
                and not mcs.is_recharging
                and not mcs.is_broken
                and not getattr(mcs, 'is_energy_stranded', False)
                and not low_energy
                and not getattr(mcs, 'centralized_recharge_pending', False)
            )
        return result

    def _nearest_fcs_energy(self, position: Iterable[float]) -> float:
        if not self.world.FCSs:
            return float('inf')
        return min(
            euclidean_distance(*position, *fcs.pos) / 1000.0 * POWER_UNIT
            for fcs in self.world.FCSs
        )

    def _pair_features_and_action_mask(
        self, eligible: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pair = np.zeros(
            (NUM_MCS, self.action_count, 2), dtype=np.float32
        )
        mask = np.zeros((NUM_MCS, self.action_count), dtype=bool)
        for mcs_index, mcs in enumerate(self.world.MCSs[:NUM_MCS]):
            if not eligible[mcs_index]:
                continue
            for point_index, point in enumerate(self.dispatch_points):
                distance_m = euclidean_distance(*mcs.pos, *point)
                move_energy = distance_m / 1000.0 * POWER_UNIT
                reserve = (
                    self._nearest_fcs_energy(point)
                    + self.recharge_threshold_kwh
                )
                margin = float(mcs.remain) - move_energy - reserve
                pair[mcs_index, point_index] = [
                    np.clip(distance_m / self._map_diagonal_m, 0.0, 1.0),
                    np.clip(margin / MCS_BATTERY_CAPACITY, -1.0, 1.0),
                ]
                mask[mcs_index, point_index] = bool(margin > 0.0)
            # Stay is always a legal policy action for an otherwise eligible
            # MCS.  Service safety remains enforced by ImmediateMatcher.
            stay_margin = (
                float(mcs.remain)
                - self._nearest_fcs_energy(mcs.pos)
                - self.recharge_threshold_kwh
            )
            pair[mcs_index, self.stay_action] = [
                0.0,
                np.clip(stay_margin / MCS_BATTERY_CAPACITY, -1.0, 1.0),
            ]
            mask[mcs_index, self.stay_action] = True
        return pair, mask

    def observe(self) -> Dict[str, np.ndarray]:
        """Build exactly one canonical graph for the current physical step."""
        graph, _mcs_index, _ev_index = self.graph_builder.build()
        eligible = self._eligible_mask()
        pair, action_mask = self._pair_features_and_action_mask(eligible)
        points = self.dispatch_points.astype(np.float32).copy()
        points[:, 0] = (
            points[:, 0] - AREA_LON_MIN
        ) / max(AREA_LON_MAX - AREA_LON_MIN, 1e-8)
        points[:, 1] = (
            points[:, 1] - AREA_LAT_MIN
        ) / max(AREA_LAT_MAX - AREA_LAT_MIN, 1e-8)
        return {
            'graph': graph.copy(),
            'eligible_mask': eligible,
            'action_mask': action_mask,
            'pair_features': pair,
            'dispatch_points': points,
            'step': np.asarray(self.world.current_step, dtype=np.int64),
        }

    def _begin_dispatch(self, mcs, point_index: int) -> None:
        target = self.dispatch_points[int(point_index)].tolist()
        mcs.centralized_dispatch_target_id = int(point_index)
        mcs.centralized_dispatch_target_pos = list(target)
        mcs.centralized_dispatch_start_step = int(self.world.current_step)
        mcs.current_target = None
        mcs.current_target_id = int(point_index)
        mcs.current_target_type = 'DISPATCH'
        mcs.current_target_pos = list(target)
        distance_m = euclidean_distance(*mcs.pos, *target)
        mcs.centralized_is_dispatching = bool(distance_m > 1e-6)
        mcs.is_idle = not mcs.centralized_is_dispatching
        mcs.is_arrive = not mcs.centralized_is_dispatching

    @staticmethod
    def _clear_dispatch(mcs, arrived: bool) -> None:
        mcs.centralized_is_dispatching = False
        mcs.centralized_dispatch_target_id = -1
        mcs.centralized_dispatch_target_pos = None
        mcs.centralized_dispatch_start_step = -1
        mcs.current_target = None
        mcs.current_target_id = -1
        mcs.current_target_type = ''
        mcs.current_target_pos = None
        mcs.is_arrive = False
        if arrived and not mcs.is_broken and not mcs.is_energy_stranded:
            mcs.is_idle = True

    @staticmethod
    def _waypoint(mcs) -> List[float] | None:
        target = getattr(mcs, 'centralized_dispatch_target_pos', None)
        if target is None:
            return None
        distance_m = euclidean_distance(*mcs.pos, *target)
        if distance_m <= MAX_MOVE_PER_STEP:
            return list(target)
        ratio = float(MAX_MOVE_PER_STEP) / max(distance_m, 1e-8)
        return [
            float(mcs.pos[0]) + ratio * (float(target[0]) - float(mcs.pos[0])),
            float(mcs.pos[1]) + ratio * (float(target[1]) - float(mcs.pos[1])),
        ]

    def _start_forced_recharge(self) -> List[Dict]:
        candidates = []
        for mcs in self.world.MCSs:
            pending = bool(getattr(mcs, 'centralized_recharge_pending', False))
            needs_recharge = bool(
                float(mcs.remain) < self.recharge_threshold_kwh
                and not self._is_dispatching(mcs)
                and not mcs.is_recharging
                and mcs.current_target is None
                and not mcs.is_broken
                and not getattr(mcs, 'is_energy_stranded', False)
            )
            if not (pending or needs_recharge):
                continue
            # RechargeMatcher accepts only idle MCSs.  This temporary state is
            # internal to the forced recharge transaction; a failed match is
            # restored to a non-serviceable pending task below.
            mcs.is_idle = True
            candidates.append(mcs)
        results = self.recharge_matcher.match_all(
            candidates, self.world.FCSs
        )
        by_id = {int(item['mcs_id']): item for item in results}
        for mcs in candidates:
            success = bool(by_id.get(int(mcs.id), {}).get('success', False))
            mcs.centralized_recharge_pending = not success
            if not success:
                mcs.is_idle = False
        return results

    def _policy_actions(self, actions: np.ndarray, observation: Dict) -> None:
        eligible = np.asarray(observation['eligible_mask'], dtype=bool)
        masks = np.asarray(observation['action_mask'], dtype=bool)
        if actions.shape != (NUM_MCS,):
            raise ValueError(
                f'joint action shape {actions.shape} != {(NUM_MCS,)}'
            )
        for index, mcs in enumerate(self.world.MCSs[:NUM_MCS]):
            if not eligible[index]:
                if int(actions[index]) != -1:
                    raise ValueError('task MCS must use implicit action -1')
                continue
            action = int(actions[index])
            if action < 0 or action >= self.action_count or not masks[index, action]:
                raise ValueError(f'illegal centralized action mcs={mcs.id}: {action}')
            option_id = int(self.world.current_step * 10_000 + mcs.id)
            mcs.active_high_option_id = option_id
            mcs.active_serve_option_id = option_id
            mcs.active_low_decision_id = option_id
            mcs.active_high_mode = 'CentralizedDispatch'
            mcs.active_serve_has_matched = False
            mcs.active_low_candidate_id = action
            mcs.active_low_started_step = int(self.world.current_step)
            if action == self.stay_action:
                self._clear_dispatch(mcs, arrived=True)
            else:
                self._begin_dispatch(mcs, action)

    def _world_actions(self) -> List[Dict]:
        result: List[Dict] = []
        for mcs in self.world.MCSs:
            if self._is_dispatching(mcs):
                result.append({
                    'mode': 'Serve',
                    'requested_mode': 'Serve',
                    'target_pos': self._waypoint(mcs),
                    'high_action_mask': [True, True],
                    'low_decision_id': int(mcs.active_low_decision_id),
                    'serve_option_id': int(mcs.active_serve_option_id),
                    'high_option_id': int(mcs.active_high_option_id),
                    'low_candidate_id': int(
                        mcs.centralized_dispatch_target_id
                    ),
                    'low_stay_selected': False,
                    'low_forced_stay': False,
                    'has_quasi_candidate': True,
                })
            elif mcs.is_idle and not mcs.is_recharging:
                # An eligible Stay remains available to the end-of-step
                # matcher, but starts a fresh centralized decision record.
                result.append({
                    'mode': 'Serve',
                    'requested_mode': 'Serve',
                    'target_pos': list(mcs.pos),
                    'high_action_mask': [True, True],
                    'low_decision_id': int(mcs.active_low_decision_id),
                    'serve_option_id': int(mcs.active_serve_option_id),
                    'high_option_id': int(mcs.active_high_option_id),
                    'low_candidate_id': self.stay_action,
                    'low_stay_selected': True,
                    'low_forced_stay': False,
                    'has_quasi_candidate': False,
                })
            else:
                result.append({'target_pos': None})
        return result

    def _finish_dispatch_movements(self) -> int:
        arrivals = 0
        for mcs in self.world.MCSs:
            if not self._is_dispatching(mcs):
                continue
            target = mcs.centralized_dispatch_target_pos
            if mcs.is_broken or mcs.is_energy_stranded:
                self._clear_dispatch(mcs, arrived=False)
                continue
            if target is not None and euclidean_distance(*mcs.pos, *target) <= 1.0:
                arrivals += 1
                self._clear_dispatch(mcs, arrived=True)
        return arrivals

    def _register_new_matches(self) -> None:
        mcs_by_id = {int(mcs.id): mcs for mcs in self.world.MCSs}
        for result in self.world.last_immediate_results:
            if not result.get('success'):
                continue
            ev_id = int(result['ev_id'])
            charge_power = float(result.get('charge_power', 0.0))
            if result.get('provider_type') == 'MCS':
                mcs = mcs_by_id[int(result['provider_id'])]
                charge_pos = list(mcs.current_target_pos or mcs.pos)
                movement_energy = (
                    euclidean_distance(*mcs.pos, *charge_pos)
                    / 1000.0 * POWER_UNIT
                )
                profit = (
                    charge_power * CHARGE_PRICE
                    - (charge_power + movement_energy) * RC_PRICE
                )
            else:
                profit = charge_power * (CHARGE_PRICE - PG_PRICE)
            self.pending_service_profit[ev_id] = float(profit)

    def _completed_service_ids(self, in_progress: Dict[int, bool]) -> List[int]:
        completed = []
        for ev in self.world.EVs:
            if (
                in_progress.get(int(ev.id), False)
                and ev.is_charged
                and float(ev.charge_time_remain_min) <= 1e-8
            ):
                completed.append(int(ev.id))
        return completed

    def step(self, actions: np.ndarray):
        """Advance one fixed-duration schedule -> progress -> match step."""
        observation = self.observe()
        actions = np.asarray(actions, dtype=np.int64)
        self._policy_actions(actions, observation)

        in_progress = {
            int(ev.id): bool(
                ev.is_charged and float(ev.charge_time_remain_min) > 1e-8
            )
            for ev in self.world.EVs
        }
        # ``World.update`` advances a quasi EV along its route, where it then
        # calls ``EV.set_charge``.  An EV which was already an IEV does not
        # travel, so it would otherwise never receive the subsequent
        # ``set_charge`` calls that advance its wait-time / timeout state.
        # Snapshot before physical progress: an EV that *becomes* IEV in this
        # interval must start at wait=0 and only begin waiting next interval.
        awaiting_iev_ids = {
            int(ev.id) for ev in self.world.EVs if ev.is_iev
        }
        failed_before = {int(ev.id): bool(ev.fail_charge) for ev in self.world.EVs}
        positions_before = {
            int(mcs.id): list(mcs.pos) for mcs in self.world.MCSs
        }

        recharge_results = self._start_forced_recharge()
        dispatching_this_step = {
            int(mcs.id)
            for mcs in self.world.MCSs
            if self._is_dispatching(mcs)
        }
        # World.update expects actions aligned with world.agents.  The
        # centralized environment deliberately supplies every MCS; EV motion
        # and all existing charging tasks remain handled by World.update.
        self.world.agents = list(self.world.MCSs)
        self.world.update(self._world_actions())
        arrivals = self._finish_dispatch_movements()

        # Centralized-only IEV deadline progression.  Keep this outside the
        # legacy World implementation so existing MAPPO experiments retain
        # their original dynamics.  It is deliberately before matching: an
        # EV whose deadline expires during this physical interval cannot be
        # rescued by a match made at the end of the same interval.
        waiting_iev_advanced = 0
        timeout_failure_count = 0
        for ev in self.world.EVs:
            if int(ev.id) not in awaiting_iev_ids or not ev.is_iev:
                continue
            was_failed = bool(ev.fail_charge)
            ev.set_charge()
            waiting_iev_advanced += 1
            timeout_failure_count += int(not was_failed and ev.fail_charge)

        completed_ids = self._completed_service_ids(in_progress)
        new_failures = sum(
            bool(ev.fail_charge) and not failed_before[int(ev.id)]
            for ev in self.world.EVs
        )
        realised_profit = sum(
            self.pending_service_profit.pop(ev_id, 0.0)
            for ev_id in completed_ids
        )
        dispatch_distance_km = sum(
            euclidean_distance(
                *positions_before[int(mcs.id)], *mcs.pos
            ) / 1000.0
            for mcs in self.world.MCSs
            if int(mcs.id) in dispatching_this_step
        )
        dispatch_cost = dispatch_distance_km * POWER_UNIT * RC_PRICE

        # Matching is intentionally last.  A new match creates a task for the
        # next state; it is not counted as a completed service in this reward.
        self.world.step_finish()
        self.world.match_and_get_neibor()
        self._register_new_matches()

        cfg = self.reward_config
        reward_components = {
            'completed_service_reward': cfg.success * len(completed_ids),
            'failure_penalty': -cfg.failure * new_failures,
            'realised_profit_reward': cfg.realised_profit * realised_profit,
            'dispatch_cost_penalty': -cfg.dispatch_cost * dispatch_cost,
        }
        reward = float(sum(reward_components.values()))
        self.last_reward_components = {
            **{key: float(value) for key, value in reward_components.items()},
            'total': reward,
        }
        done = bool(
            self.world.current_step >= min(
                self.max_steps, MAX_STEPS_PER_EPISODE
            )
        )
        info = self.metrics()
        info.update({
            'dispatch_arrival_count_step': int(arrivals),
            'completed_service_count_step': int(len(completed_ids)),
            'new_failure_count_step': int(new_failures),
            'realised_service_profit_step': float(realised_profit),
            'dispatch_distance_km_step': float(dispatch_distance_km),
            'forced_recharge_request_count_step': int(len(recharge_results)),
            'iev_waiting_advanced_count_step': int(waiting_iev_advanced),
            'iev_timeout_failure_count_step': int(timeout_failure_count),
            **self.last_reward_components,
        })
        self.last_step_metrics = dict(info)
        return self.observe(), reward, done, info

    def metrics(self) -> Dict[str, float]:
        completed = sum(
            ev.is_charged and float(ev.charge_time_remain_min) <= 1e-8
            for ev in self.world.EVs
        )
        matched_or_completed = sum(ev.is_charged for ev in self.world.EVs)
        failures = sum(ev.fail_charge for ev in self.world.EVs)
        moving = sum(self._is_dispatching(mcs) for mcs in self.world.MCSs)
        charging = sum(
            mcs.is_task and str(mcs.current_target_type).upper() == 'IEV'
            for mcs in self.world.MCSs
        )
        recharging = sum(mcs.is_recharging for mcs in self.world.MCSs)
        return {
            'step': int(self.world.current_step),
            'completed_ev_count': int(completed),
            'matched_or_completed_ev_count': int(matched_or_completed),
            'failed_ev_count': int(failures),
            'completed_ev_ratio': float(completed / max(len(self.world.EVs), 1)),
            'moving_mcs_count': int(moving),
            'charging_mcs_count': int(charging),
            'recharging_mcs_count': int(recharging),
            'eligible_mcs_count': int(self._eligible_mask().sum()),
            'total_mcs_profit': float(sum(
                mcs.total_profit for mcs in self.world.MCSs
            )),
            'total_fcs_profit': float(sum(
                fcs.total_profit for fcs in self.world.FCSs
            )),
        }
