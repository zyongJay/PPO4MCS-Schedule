"""使用固定场景对比 v10 分层策略、随机 Low、完整 Random 和阈值 High。

实验 A：v10 High Actor + v10 Low Actor，High/Low 均使用确定性贪心动作。
实验 B：v10 High Actor + Random Low，High 使用同一个确定性策略，底层仅在
        Low Actor 当前可见的合法 quasi 候选中均    匀随机选择。
实验 C：完整采用 test.py 中的 RandomDecisionPolicy，包括补电规则、随机选取
        周围 quasi 和无候选时 Wait 的规则。
实验 D：保持与 A 完全相同的 Option 生命周期、合法动作域和 v10 Low Actor，
        仅将 learned High Actor 替换为 40 kWh 固定阈值规则。Serve 因绝对
        安全约束不可用时仍强制 Recharge；阈值只在 High Option 边界判断。

默认读取 ``training_results_v10/best_model.pt``，并在种子 1001--1050
对应的 50 个固定场景上进行配对测试。结果输出到项目根目录的
``test_actor_results``。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

import world as world_module
from config import (
    MAX_CONSECUTIVE_NO_CANDIDATE_LOW_REPLANS,
    MAX_SERVE_OPTION_STEPS,
    MAX_STEPS_PER_EPISODE,
    MCS_RECHARGE_THRESHOLD,
    TRACK_DATA_PATH,
)
from core import EV, MCS
from environment import MultiAgentEnv
from matching import RechargeMatcher
from network import MCSMAPPOAgent, MCS_ACTION_NAMES
from observation import MCS_STAY_CANDIDATE_ID
from test import (
    RandomDecisionPolicy,
    build_summary,
    iev_track_action,
    load_rl_agent,
    mean_attribute,
    remember_charge_providers,
    resolve_device,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v10' / 'best_model.pt'
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_actor_results'
DEFAULT_ACTOR_SEEDS = tuple(range(1001, 1051))
DEFAULT_ONLYLOW_CHECKPOINT = (
    PROJECT_DIR / 'training_results_v10_onlylow' / 'best_model.pt'
)

# config.py 中的轨迹路径以 demo 目录为基准。转为绝对路径，避免启动目录影响。
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())

EXPERIMENT_A = 'A_v10_high_v10_low'
EXPERIMENT_B = 'B_v10_high_random_low'
EXPERIMENT_C = 'C_full_random'
EXPERIMENT_D = 'D_threshold40_option_high_v10_low'

# Low Actor 候选特征在 observation.py 中的定义。
LOW_FEATURE_NAMES = (
    'urgency_demand_ratio',
    'distance_ratio',
    'attraction',
    'immediate_iev_attraction',
    'mcs_competition',
    'fcs_competition',
)


def select_threshold_high_action(mcs: MCS, observation: Dict) -> int:
    """在 High Option 边界返回阈值基线动作索引。

    D 与 A 共用同一个两动作安全可行域。只有 Serve/Recharge 同时合法时，
    才由严格的 ``remain < 40 kWh`` 阈值替代 learned High Actor；若 Serve
    已被绝对安全约束屏蔽，则无条件 Recharge。这样 A-D 的可控差异只有
    High 决策器，而不是 Option 生命周期、Low Actor 或安全约束。
    """
    high_mask = np.asarray(
        observation.get('high_action_mask', []), dtype=bool
    ).reshape(-1)
    if high_mask.size < 2:
        raise ValueError('阈值 High 缺少 Serve/Recharge 动作掩码')

    serve_allowed = bool(high_mask[0])
    recharge_allowed = bool(high_mask[1])
    if not serve_allowed and recharge_allowed:
        return 1
    if (
        serve_allowed
        and recharge_allowed
        and float(mcs.remain) < float(MCS_RECHARGE_THRESHOLD)
    ):
        return 1
    if serve_allowed:
        return 0
    if recharge_allowed:
        return 1
    raise RuntimeError('High 动作掩码中 Serve/Recharge 均不可用')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='在固定场景中对比四组 High/Low/Random 消融策略'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help='默认使用 v10 按充电率选出的 best_model.pt',
    )
    parser.add_argument(
        '--onlylow-checkpoint',
        type=Path,
        default=DEFAULT_ONLYLOW_CHECKPOINT,
        help='v10_onlylow 的 Low Actor checkpoint（High 使用固定阈值）',
    )
    parser.add_argument(
        '--three-policy',
        action='store_true',
        help='仅比较 v10、v10_onlylow（阈值 High）和完整 Random',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(DEFAULT_ACTOR_SEEDS),
        help='四组实验共同使用的固定场景种子',
    )
    parser.add_argument(
        '--max-steps', type=int, default=MAX_STEPS_PER_EPISODE
    )
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto'
    )
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument(
        '--random-low-seed-offset',
        type=int,
        default=100_000,
        help='实验 B 的随机 Low 种子偏移，便于完全复现实验',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help=(
            '将新场景追加到 output-dir 中已有的 actor_scenarios.csv，'
            '并基于合并后的全部场景重算汇总与配对比较'
        ),
    )
    parser.add_argument(
        '--no-save', action='store_true', help='仅打印结果，不保存 CSV/JSON'
    )
    return parser.parse_args()


class CheckpointLowAblationPolicy:
    """使用 checkpoint Low Actor，并允许切换 learned/threshold High。"""

    def __init__(
        self,
        agent: MCSMAPPOAgent,
        low_mode: str,
        random_seed: int,
        high_mode: str = 'learned',
    ):
        if low_mode not in ('learned', 'random'):
            raise ValueError(f'不支持的 low_mode: {low_mode}')
        if high_mode not in ('learned', 'threshold'):
            raise ValueError(f'不支持的 high_mode: {high_mode}')
        self.agent = agent
        self.low_mode = low_mode
        self.high_mode = high_mode
        self.rng = random.Random(random_seed)
        self.recharge_matcher = RechargeMatcher()
        self.active_high_options: Dict[int, Dict] = {}

        # 统计 High Actor 的原始请求动作，不把 Recharge 匹配失败后的 Wait
        # 误记为 High Actor 主动选择 Wait。
        self.high_action_counts = {name: 0 for name in MCS_ACTION_NAMES}
        self.low_selected_ranks: List[int] = []
        self.low_selected_features: List[np.ndarray] = []

    def synchronize(self) -> None:
        if self.agent.device.type == 'cuda':
            torch.cuda.synchronize(self.agent.device)

    @torch.no_grad()
    def _select_mcs_actions(
        self,
        mcss: Sequence[MCS],
        observations: Sequence[Dict],
        select_low: bool = True,
    ) -> List[Dict]:
        if not observations:
            return []
        if len(mcss) != len(observations):
            raise ValueError('MCS 与 observation 数量不一致')

        if self.high_mode == 'learned':
            high_states = self.agent._tensor(np.stack([
                obs['high_state'] for obs in observations
            ]))
            high_masks = self.agent._tensor(np.stack([
                obs['high_action_mask'] for obs in observations
            ]), dtype=torch.bool)
            high_distribution = self.agent.high_actor.distribution(
                high_states, high_masks
            )
            high_indices = (
                high_distribution.probs.argmax(dim=-1)
                .detach().cpu().numpy().astype(int)
            )
        else:
            # D 只替换 Option 边界处的 High 决策器；Option 生命周期、Low
            # Actor 和安全可行域与 A 完全一致。
            high_indices = np.asarray([
                select_threshold_high_action(mcs, observation)
                for mcs, observation in zip(mcss, observations)
            ], dtype=int)

        results = []
        for action_index in high_indices:
            mode = MCS_ACTION_NAMES[int(action_index)]
            self.high_action_counts[mode] += 1
            results.append({'mode': mode, 'low_action': -1})

        serve_indices = np.flatnonzero(high_indices == 0)
        if not serve_indices.size or not select_low:
            return results
        selected_low_indices = self._select_low_actions([
            observations[int(index)] for index in serve_indices
        ])
        for batch_index, observation_index in enumerate(serve_indices):
            results[int(observation_index)]['low_action'] = int(
                selected_low_indices[batch_index]
            )
        return results

    @torch.no_grad()
    def _select_low_actions(
        self,
        observations: Sequence[Dict],
    ) -> List[int]:
        if not observations:
            return []
        if self.low_mode == 'learned':
            low_self_states = self.agent._tensor(np.stack([
                observation['low_self_state']
                for observation in observations
            ]))
            low_candidates = self.agent._tensor(np.stack([
                observation['low_candidates']
                for observation in observations
            ]))
            low_masks = self.agent._tensor(np.stack([
                observation['low_candidate_mask']
                for observation in observations
            ]), dtype=torch.bool)
            low_distribution = self.agent.low_actor.distribution(
                low_self_states, low_candidates, low_masks
            )
            selected_low_indices = (
                low_distribution.probs.argmax(dim=-1)
                .detach().cpu().numpy().astype(int)
            )
        else:
            # 为保证消融实验只改变“如何选 quasi”，随机策略与 Low Actor
            # 使用完全相同的合法候选集合，而不是扩大到观测范围外的 EV。
            selected_low_indices = []
            for observation in observations:
                valid_indices = np.flatnonzero(
                    observation['low_candidate_mask']
                )
                if not valid_indices.size:
                    raise RuntimeError(
                        'High Actor 选择 Serve，但 Low Actor 没有合法 quasi 候选'
                    )
                selected_low_indices.append(
                    self.rng.choice(valid_indices.tolist())
                )
            selected_low_indices = np.asarray(
                selected_low_indices, dtype=int
            )

        results: List[int] = []
        for batch_index, observation in enumerate(observations):
            low_index = int(selected_low_indices[batch_index])
            if not bool(observation['low_candidate_mask'][low_index]):
                raise RuntimeError('底层策略选择了非法 quasi 候选')
            self.low_selected_ranks.append(low_index + 1)
            self.low_selected_features.append(
                np.asarray(
                    observation['low_candidates'][low_index],
                    dtype=np.float64,
                ).copy()
            )
            results.append(low_index)
        return results

    def build_actions(
        self,
        env: MultiAgentEnv,
        acting_agents: Sequence,
        observations: Sequence[Dict],
    ) -> List[Dict]:
        observation_by_agent = {
            actor: observations[index]
            for index, actor in enumerate(acting_agents)
            if index < len(observations)
        }
        acting_mcss = [
            actor for actor in acting_agents if isinstance(actor, MCS)
        ]
        mcs_by_id = {mcs.id: mcs for mcs in env.world.MCSs}
        for mcs_id, state in list(self.active_high_options.items()):
            mcs = mcs_by_id[mcs_id]
            if state['mode'] == 'Serve' and mcs.is_task:
                state['task_started'] = True
            completed = bool(
                state['mode'] == 'Serve'
                and state['task_started'] and mcs.is_idle
            ) or bool(
                state['mode'] == 'Recharge'
                and state['matched'] and mcs.is_idle
            )
            abnormal = bool(
                state['mode'] == 'Serve'
                and (
                    state['duration_steps'] >= MAX_SERVE_OPTION_STEPS
                    or state['no_candidate_replans']
                    >= MAX_CONSECUTIVE_NO_CANDIDATE_LOW_REPLANS
                )
            )
            if completed or abnormal or mcs.is_broken or mcs.is_energy_stranded:
                self.active_high_options.pop(mcs_id, None)

        for mcs in acting_mcss:
            state = self.active_high_options.get(mcs.id)
            if (
                state is not None and state['mode'] == 'Serve'
                and bool(
                    observation_by_agent[mcs].get(
                        'high_recharge_only', False
                    )
                )
            ):
                self.active_high_options.pop(mcs.id)

        boundary_mcss = [
            mcs for mcs in acting_mcss
            if mcs.id not in self.active_high_options
        ]
        boundary_observations = [
            observation_by_agent[mcs] for mcs in boundary_mcss
        ]
        boundary_actions = self._select_mcs_actions(
            boundary_mcss, boundary_observations, select_low=False
        )
        for mcs, action in zip(boundary_mcss, boundary_actions):
            self.active_high_options[mcs.id] = {
                'mode': action['mode'],
                'duration_steps': 0,
                'no_candidate_replans': 0,
                'task_started': False,
                'matched': False,
            }

        active_serve_mcss = [
            mcs for mcs in acting_mcss
            if self.active_high_options[mcs.id]['mode'] == 'Serve'
        ]
        low_actions = self._select_low_actions([
            observation_by_agent[mcs] for mcs in active_serve_mcss
        ])
        low_action_by_id = {
            mcs.id: low_actions[index]
            for index, mcs in enumerate(active_serve_mcss)
        }
        recharge_requests = [
            mcs for mcs in acting_mcss
            if self.active_high_options[mcs.id]['mode'] == 'Recharge'
        ]
        recharge_results = self.recharge_matcher.match_all(
            recharge_requests, env.world.FCSs
        )
        recharge_matched_ids = {
            int(result['mcs_id'])
            for result in recharge_results if result.get('success')
        }
        ev_by_id = {int(ev.id): ev for ev in env.world.EVs}

        action_n = []
        for actor in acting_agents:
            if isinstance(actor, EV):
                action_n.append(iev_track_action(actor))
                continue

            observation = observation_by_agent[actor]
            state = self.active_high_options[actor.id]
            state['duration_steps'] += 1
            common = {
                # 显式传回决策时的 mask，使 reward.py 能准确区分 forced Wait
                # 与 voluntary Wait。
                'high_action_mask': (
                    observation['high_action_mask'].tolist()
                ),
            }
            if state['mode'] == 'Recharge':
                matched = actor.id in recharge_matched_ids
                state['matched'] = bool(state['matched'] or matched)
                action_n.append({
                    **common,
                    'mode': 'Recharge' if matched else 'Wait',
                    'requested_mode': 'Recharge',
                    'recharge_matched': matched,
                    'target_pos': (
                        list(actor.current_target_pos)
                        if matched else list(actor.pos)
                    ),
                })
            else:
                low_action = low_action_by_id[actor.id]
                candidate_id = int(
                    observation['candidate_ids'][low_action]
                )
                low_stay_selected = candidate_id == MCS_STAY_CANDIDATE_ID
                has_quasi_candidate = bool(
                    observation.get('quasi_candidate_count', 0) > 0
                )
                state['no_candidate_replans'] = (
                    0 if has_quasi_candidate
                    else state['no_candidate_replans'] + 1
                )
                target = ev_by_id.get(candidate_id)
                if (
                    not low_stay_selected
                    and (target is None or not target.is_quasi)
                ):
                    raise RuntimeError(
                        f'候选 EV {candidate_id} 已不是合法 quasi'
                    )
                action_n.append({
                    **common,
                    'mode': 'Serve',
                    'requested_mode': 'Serve',
                    'recharge_matched': False,
                    'target_pos': (
                        list(actor.pos)
                        if low_stay_selected else list(target.pos)
                    ),
                    'target_ev_id': candidate_id,
                    'low_candidate_id': candidate_id,
                    'low_stay_selected': low_stay_selected,
                    'low_forced_stay': bool(
                        low_stay_selected and not has_quasi_candidate
                    ),
                    'has_quasi_candidate': has_quasi_candidate,
                })
        return action_n

    def diagnostic_metrics(self) -> Dict[str, float | int]:
        total_high = sum(self.high_action_counts.values())
        metrics: Dict[str, float | int] = {
            'high_action_total_count': int(total_high),
            'high_serve_count': int(self.high_action_counts['Serve']),
            'high_recharge_count': int(self.high_action_counts['Recharge']),
            'high_wait_count': int(self.high_action_counts.get('Wait', 0)),
            'high_serve_ratio': (
                self.high_action_counts['Serve'] / total_high
                if total_high else 0.0
            ),
            'high_recharge_ratio': (
                self.high_action_counts['Recharge'] / total_high
                if total_high else 0.0
            ),
            'high_wait_ratio': (
                self.high_action_counts.get('Wait', 0) / total_high
                if total_high else 0.0
            ),
            'low_selection_count': len(self.low_selected_ranks),
            'avg_low_selected_candidate_rank': (
                float(np.mean(self.low_selected_ranks))
                if self.low_selected_ranks else 0.0
            ),
        }
        if self.low_selected_features:
            feature_means = np.mean(
                np.stack(self.low_selected_features), axis=0
            )
        else:
            feature_means = np.zeros(len(LOW_FEATURE_NAMES), dtype=float)
        for name, value in zip(LOW_FEATURE_NAMES, feature_means):
            metrics[f'avg_low_selected_{name}'] = float(value)
        return metrics


class InstrumentedRandomDecisionPolicy(RandomDecisionPolicy):
    """原样执行 test.py Random 策略，只额外记录动作数量。

    这里不改写 RandomDecisionPolicy 的任何决策规则。C 组统计列中的
    ``high_*`` 表示随机策略产生的顶层 Serve/Recharge/Wait 请求，不能理解为
    High Actor 的神经网络输出。
    """

    def __init__(self, seed: int):
        super().__init__(seed)
        # 这是旧随机基线而非High Actor，仍允许其产生显式Wait。
        self.high_action_counts = {
            'Serve': 0, 'Recharge': 0, 'Wait': 0
        }

    def build_actions(
        self,
        env: MultiAgentEnv,
        acting_agents: Sequence,
        observations: Sequence[Dict],
    ) -> List[Dict]:
        action_n = super().build_actions(env, acting_agents, observations)
        for actor, action in zip(acting_agents, action_n):
            if not isinstance(actor, MCS):
                continue
            # Recharge 抢位失败时执行模式是 Wait，但策略请求仍是 Recharge。
            requested_mode = action.get('requested_mode', action['mode'])
            if requested_mode not in self.high_action_counts:
                raise RuntimeError(
                    f'Random 策略返回未知动作: {requested_mode}'
                )
            self.high_action_counts[requested_mode] += 1
        return action_n

    def synchronize(self) -> None:
        """完整 Random 策略不执行异步神经网络推理。"""

    def diagnostic_metrics(self) -> Dict[str, float | int]:
        total = sum(self.high_action_counts.values())
        metrics: Dict[str, float | int] = {
            'high_action_total_count': int(total),
            'high_serve_count': int(self.high_action_counts['Serve']),
            'high_recharge_count': int(
                self.high_action_counts['Recharge']
            ),
            'high_wait_count': int(self.high_action_counts.get('Wait', 0)),
            'high_serve_ratio': (
                self.high_action_counts['Serve'] / total if total else 0.0
            ),
            'high_recharge_ratio': (
                self.high_action_counts['Recharge'] / total
                if total else 0.0
            ),
            'high_wait_ratio': (
                self.high_action_counts.get('Wait', 0) / total if total else 0.0
            ),
            # C 组没有 Low Actor 的候选索引和 checkpoint 候选特征；用 NaN
            # 明确标记
            # “不适用”，防止汇总时被误当作 0。
            'low_selection_count': np.nan,
            'avg_low_selected_candidate_rank': np.nan,
        }
        for name in LOW_FEATURE_NAMES:
            metrics[f'avg_low_selected_{name}'] = np.nan
        return metrics


def evaluate_scenario(
    experiment_name: str,
    policy_mode: str,
    scenario_seed: int,
    random_low_seed: int,
    max_steps: int,
    checkpoint_path: Path,
    checkpoint_metadata: Dict,
    agent: MCSMAPPOAgent,
) -> Dict:
    """运行一个固定场景，返回业务指标、顶层动作比例和选位特征。"""
    random.seed(scenario_seed)
    np.random.seed(scenario_seed)
    torch.manual_seed(scenario_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(scenario_seed)

    env = MultiAgentEnv(scenario_seed)
    env.world.verbose = False
    observations = env.reset()
    if policy_mode == 'full_random':
        # 与 test.py 的 build_decision_policy(..., seed=scenario_seed) 一致。
        policy = InstrumentedRandomDecisionPolicy(scenario_seed)
    elif policy_mode in (
        'learned', 'random_low', 'threshold_high_learned_low'
    ):
        low_mode = 'learned' if policy_mode == 'learned' else 'random'
        if policy_mode == 'threshold_high_learned_low':
            low_mode = 'learned'
        high_mode = (
            'threshold'
            if policy_mode == 'threshold_high_learned_low'
            else 'learned'
        )
        policy = CheckpointLowAblationPolicy(
            agent, low_mode, random_low_seed, high_mode=high_mode
        )
    else:
        raise ValueError(f'未知实验策略模式: {policy_mode}')

    provider_by_ev_id: Dict[int, str] = {}
    remember_charge_providers(env.world.EVs, provider_by_ev_id)
    decision_times_ms: List[float] = []
    executed_steps = 0

    for _ in range(max_steps):
        acting_agents = list(env.world.agents)
        policy.synchronize()
        decision_start_ns = time.perf_counter_ns()
        action_n = policy.build_actions(
            env, acting_agents, observations
        )
        policy.synchronize()
        decision_times_ms.append(
            (time.perf_counter_ns() - decision_start_ns) / 1_000_000.0
        )
        if len(action_n) != len(acting_agents):
            raise RuntimeError(
                f'{experiment_name}: action 数与 acting_agents 数不一致'
            )

        observations, _, _, _ = env.step(action_n)
        executed_steps += 1
        remember_charge_providers(env.world.EVs, provider_by_ev_id)
        if env.world.get_done():
            break

    evs = list(env.world.EVs)
    mcss = list(env.world.MCSs)
    fcss = list(env.world.FCSs)
    successful_evs = [ev for ev in evs if ev.is_charged]
    failure_count = int(sum(ev.fail_charge for ev in evs))
    success_count = len(successful_evs)
    finished_count = success_count + failure_count
    unresolved_count = int(sum(
        not ev.is_normal and not ev.fail_charge and not ev.is_charged
        for ev in evs
    ))

    provider_counts = {'MCS': 0, 'FCS': 0}
    missing_provider_ids = []
    for ev in successful_evs:
        provider_type = provider_by_ev_id.get(int(ev.id), '')
        if provider_type in provider_counts:
            provider_counts[provider_type] += 1
        else:
            missing_provider_ids.append(int(ev.id))
    if missing_provider_ids:
        raise RuntimeError(
            '成功 EV 缺少 charge_provider_type 记录: '
            f'{missing_provider_ids[:10]}'
        )

    checkpoint_episode = checkpoint_metadata.get('episode', '')
    if not checkpoint_episode:
        match = re.search(r'(\d+)', checkpoint_path.stem)
        checkpoint_episode = int(match.group(1)) if match else ''

    success_denominator = max(success_count, 1)
    row = {
        'policy_name': experiment_name,
        'policy_type': policy_mode,
        'high_mode': (
            policy.high_mode
            if isinstance(policy, CheckpointLowAblationPolicy)
            else 'full_random'
        ),
        'low_mode': (
            policy.low_mode
            if isinstance(policy, CheckpointLowAblationPolicy)
            else 'not_applicable'
        ),
        'checkpoint_path': (
            str(checkpoint_path) if policy_mode != 'full_random' else ''
        ),
        'checkpoint_episode': (
            checkpoint_episode if policy_mode != 'full_random' else ''
        ),
        'scenario_seed': int(scenario_seed),
        'random_low_seed': int(random_low_seed),
        'steps': int(executed_steps),
        'ev_charge_success_ratio': (
            success_count / finished_count if finished_count else 0.0
        ),
        'ev_population_success_ratio': (
            success_count / len(evs) if evs else 0.0
        ),
        'ev_success_count': int(success_count),
        'ev_failure_count': int(failure_count),
        'ev_unresolved_count': int(unresolved_count),
        'avg_mcs_profit': mean_attribute(mcss, 'total_profit'),
        'avg_mcs_idle_time_min': mean_attribute(
            mcss, 'total_idle_time_min'
        ),
        'avg_fcs_profit': mean_attribute(fcss, 'total_profit'),
        'avg_fcs_idle_time_min': mean_attribute(
            fcss, 'total_idle_time_min'
        ),
        'avg_ev_extra_distance_km': mean_attribute(
            evs, 'total_extra_dist_km'
        ),
        'avg_ev_charging_delay_min': mean_attribute(
            evs, 'total_wait_time_min'
        ),
        'avg_decision_time_ms': float(np.mean(decision_times_ms)),
        'decision_time_std_ms': float(
            np.std(decision_times_ms, ddof=0)
        ),
        'decision_time_p95_ms': float(
            np.percentile(decision_times_ms, 95)
        ),
        'successful_ev_mcs_count': int(provider_counts['MCS']),
        'successful_ev_fcs_count': int(provider_counts['FCS']),
        'successful_ev_mcs_share': (
            provider_counts['MCS'] / success_denominator
        ),
        'successful_ev_fcs_share': (
            provider_counts['FCS'] / success_denominator
        ),
        'broken_mcs_count': int(sum(mcs.is_broken for mcs in mcss)),
        'energy_stranded_mcs_count': int(sum(
            mcs.is_energy_stranded for mcs in mcss
        )),
        'unavailable_mcs_count': int(sum(
            mcs.is_broken or mcs.is_energy_stranded for mcs in mcss
        )),
    }
    row.update(policy.diagnostic_metrics())
    return row


# 对业务指标指定优化方向；动作和候选特征只做描述性比较。
METRIC_DIRECTIONS = {
    'ev_charge_success_ratio': 'higher',
    'ev_population_success_ratio': 'higher',
    'ev_success_count': 'higher',
    'ev_failure_count': 'lower',
    'ev_unresolved_count': 'lower',
    'avg_mcs_profit': 'higher',
    'avg_mcs_idle_time_min': 'lower',
    'avg_fcs_profit': 'higher',
    'avg_fcs_idle_time_min': 'lower',
    'avg_ev_extra_distance_km': 'lower',
    'avg_ev_charging_delay_min': 'lower',
    'avg_decision_time_ms': 'lower',
    'successful_ev_mcs_count': 'higher',
    'successful_ev_fcs_count': 'higher',
    'successful_ev_mcs_share': 'higher',
    'broken_mcs_count': 'lower',
    'energy_stranded_mcs_count': 'lower',
    'unavailable_mcs_count': 'lower',
    'high_serve_ratio': 'descriptive',
    'high_recharge_ratio': 'descriptive',
    'high_wait_ratio': 'descriptive',
    'avg_low_selected_candidate_rank': 'descriptive',
    'avg_low_selected_urgency_demand_ratio': 'descriptive',
    'avg_low_selected_distance_ratio': 'descriptive',
    'avg_low_selected_attraction': 'descriptive',
    'avg_low_selected_immediate_iev_attraction': 'descriptive',
    'avg_low_selected_mcs_competition': 'descriptive',
    'avg_low_selected_fcs_competition': 'descriptive',
}

COMPARISON_PAIRS = (
    (EXPERIMENT_A, EXPERIMENT_B),
    (EXPERIMENT_A, EXPERIMENT_C),
    (EXPERIMENT_A, EXPERIMENT_D),
    (EXPERIMENT_B, EXPERIMENT_C),
    (EXPERIMENT_B, EXPERIMENT_D),
    (EXPERIMENT_C, EXPERIMENT_D),
)


def build_paired_comparison(
    scenarios: pd.DataFrame,
    comparison_pairs: Sequence[tuple[str, str]] = COMPARISON_PAIRS,
) -> pd.DataFrame:
    """对 A/B/C/D 两两配对，避免场景难度差异掩盖策略差异。"""
    rows = []
    pivot_tables = {
        metric: scenarios.pivot(
            index='scenario_seed', columns='policy_name', values=metric
        )
        for metric in METRIC_DIRECTIONS
    }
    for left_policy, right_policy in comparison_pairs:
        for metric, direction in METRIC_DIRECTIONS.items():
            pivot = pivot_tables[metric]
            if left_policy not in pivot or right_policy not in pivot:
                raise ValueError(
                    f'指标 {metric} 缺少 {left_policy}/{right_policy} 数据'
                )
            paired = pivot[[left_policy, right_policy]].dropna()
            # 完整 Random 策略没有 Low Actor 候选特征；这些不适用的比较跳过。
            if paired.empty:
                continue
            left_values = paired[left_policy].to_numpy(dtype=float)
            right_values = paired[right_policy].to_numpy(dtype=float)
            delta = left_values - right_values
            n = len(delta)
            delta_std = float(np.std(delta, ddof=1)) if n > 1 else 0.0
            # 默认 50 个样本使用 t(49)≈2.009，其余样本数使用 1.96 近似。
            critical = 2.009 if n == 50 else 1.96
            half_width = critical * delta_std / np.sqrt(n) if n else 0.0

            if direction == 'higher':
                signed_advantage = delta
            elif direction == 'lower':
                signed_advantage = -delta
            else:
                signed_advantage = None

            if signed_advantage is None:
                left_better = ties = right_better = np.nan
            else:
                left_better = int(np.sum(signed_advantage > 1e-12))
                ties = int(np.sum(np.abs(signed_advantage) <= 1e-12))
                right_better = int(np.sum(signed_advantage < -1e-12))

            rows.append({
                'comparison': f'{left_policy}_vs_{right_policy}',
                'left_policy': left_policy,
                'right_policy': right_policy,
                'metric': metric,
                'preferred_direction': direction,
                'paired_scenario_count': n,
                'left_mean': float(np.mean(left_values)),
                'right_mean': float(np.mean(right_values)),
                'mean_delta_left_minus_right': float(np.mean(delta)),
                'median_delta_left_minus_right': float(np.median(delta)),
                'delta_std': delta_std,
                'delta_95ci_low_approx': float(
                    np.mean(delta) - half_width
                ),
                'delta_95ci_high_approx': float(
                    np.mean(delta) + half_width
                ),
                'left_better_count': left_better,
                'tie_count': ties,
                'right_better_count': right_better,
            })
    return pd.DataFrame(rows)


def print_comparison(comparison: pd.DataFrame) -> None:
    key_metrics = [
        'ev_charge_success_ratio',
        'ev_success_count',
        'ev_failure_count',
        'avg_mcs_profit',
        'successful_ev_mcs_count',
        'avg_ev_extra_distance_km',
        'avg_ev_charging_delay_min',
        'high_serve_ratio',
        'high_recharge_ratio',
        'high_wait_ratio',
        'avg_low_selected_candidate_rank',
    ]
    print('\n策略两两配对汇总（delta = left - right）')
    for comparison_name, group in comparison.groupby(
        'comparison', sort=False
    ):
        printable = group[group['metric'].isin(key_metrics)][[
            'metric',
            'left_mean',
            'right_mean',
            'mean_delta_left_minus_right',
            'left_better_count',
            'right_better_count',
        ]]
        print(f'\n{comparison_name}')
        print(
            printable.to_string(
                index=False, float_format=lambda x: f'{x:.6f}'
            )
        )


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError('--max-steps 必须大于 0')
    if args.hidden_dim <= 0 or args.torch_threads <= 0:
        raise ValueError('--hidden-dim 和 --torch-threads 必须大于 0')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('--seeds 中不能包含重复种子')

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint 不存在: {checkpoint_path}')
    onlylow_checkpoint_path = args.onlylow_checkpoint.expanduser().resolve()
    if args.three_policy and not onlylow_checkpoint_path.is_file():
        raise FileNotFoundError(
            f'v10_onlylow checkpoint 不存在: {onlylow_checkpoint_path}'
        )

    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    agent, checkpoint_metadata = load_rl_agent(
        checkpoint_path, args.hidden_dim, device
    )
    onlylow_agent = None
    onlylow_checkpoint_metadata: Dict = {}
    if args.three_policy:
        onlylow_agent, onlylow_checkpoint_metadata = load_rl_agent(
            onlylow_checkpoint_path, args.hidden_dim, device
        )
    max_steps = min(args.max_steps, MAX_STEPS_PER_EPISODE)

    print(
        f'device={device} checkpoint={checkpoint_path} '
        f'paired_scenarios={len(args.seeds)} max_steps={max_steps}'
    )
    scenario_rows = []
    if args.three_policy:
        if onlylow_agent is None:
            raise RuntimeError('v10_onlylow agent 未正确加载')
        experiments = (
            ('v10', 'learned', checkpoint_path, checkpoint_metadata, agent),
            (
                'v10_onlylow',
                'threshold_high_learned_low',
                onlylow_checkpoint_path,
                onlylow_checkpoint_metadata,
                onlylow_agent,
            ),
            ('Random', 'full_random', checkpoint_path, checkpoint_metadata, agent),
        )
        comparison_pairs = (
            ('v10', 'v10_onlylow'),
            ('v10', 'Random'),
            ('v10_onlylow', 'Random'),
        )
    else:
        experiments = (
            (EXPERIMENT_A, 'learned', checkpoint_path, checkpoint_metadata, agent),
            (EXPERIMENT_B, 'random_low', checkpoint_path, checkpoint_metadata, agent),
            (EXPERIMENT_C, 'full_random', checkpoint_path, checkpoint_metadata, agent),
            (
                EXPERIMENT_D,
                'threshold_high_learned_low',
                checkpoint_path,
                checkpoint_metadata,
                agent,
            ),
        )
        comparison_pairs = COMPARISON_PAIRS
    # 种子在外层循环，使同一固定场景的策略结果相邻输出。
    for seed in args.seeds:
        for (
            experiment_name,
            policy_mode,
            experiment_checkpoint_path,
            experiment_checkpoint_metadata,
            experiment_agent,
        ) in experiments:
            random_low_seed = (
                seed if policy_mode == 'full_random'
                else seed + args.random_low_seed_offset
            )
            row = evaluate_scenario(
                experiment_name=experiment_name,
                policy_mode=policy_mode,
                scenario_seed=seed,
                random_low_seed=random_low_seed,
                max_steps=max_steps,
                checkpoint_path=experiment_checkpoint_path,
                checkpoint_metadata=experiment_checkpoint_metadata,
                agent=experiment_agent,
            )
            scenario_rows.append(row)
            print(
                f'{experiment_name} seed={seed} '
                f'success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_success={row["successful_ev_mcs_count"]} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f} '
                f'serve={row["high_serve_ratio"]:.3f} '
                f'wait={row["high_wait_ratio"]:.3f}'
            )

    scenario_table = pd.DataFrame(scenario_rows)
    output_dir = args.output_dir.expanduser().resolve()
    scenario_path = output_dir / 'actor_scenarios.csv'
    summary_path = output_dir / 'actor_summary.csv'
    comparison_path = output_dir / 'actor_paired_comparison.csv'
    config_path = output_dir / 'actor_config.json'

    if args.append:
        if args.no_save:
            raise ValueError('--append 不能与 --no-save 同时使用')
        if not scenario_path.is_file():
            raise FileNotFoundError(
                f'--append 要求已有逐场景结果: {scenario_path}'
            )
        existing_table = pd.read_csv(scenario_path)
        if list(existing_table.columns) != list(scenario_table.columns):
            raise ValueError('已有结果与新增结果的 CSV 列结构不一致')
        key_columns = ['policy_name', 'scenario_seed']
        existing_keys = existing_table[key_columns].astype(str).agg(
            '|'.join, axis=1
        )
        new_keys = scenario_table[key_columns].astype(str).agg(
            '|'.join, axis=1
        )
        duplicate_keys = sorted(set(existing_keys) & set(new_keys))
        if duplicate_keys:
            raise ValueError(
                '--append 检测到重复的策略/场景种子: '
                f'{duplicate_keys[:10]}'
            )
        scenario_table = pd.concat(
            [existing_table, scenario_table], ignore_index=True
        )

    summary_table = build_summary(scenario_table)
    comparison_table = build_paired_comparison(
        scenario_table, comparison_pairs
    )
    print_comparison(comparison_table)

    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        scenario_table.to_csv(scenario_path, index=False)
        summary_table.to_csv(summary_path, index=False)
        comparison_table.to_csv(comparison_path, index=False)
        config = {
            'evaluated_at': datetime.now().astimezone().isoformat(
                timespec='seconds'
            ),
            'checkpoint': str(checkpoint_path),
            'checkpoint_metadata': checkpoint_metadata,
            'seeds': sorted(
                scenario_table['scenario_seed'].astype(int).unique().tolist()
            ),
            'paired_scenario_count': int(
                scenario_table['scenario_seed'].nunique()
            ),
            'max_steps': max_steps,
            'hidden_dim': args.hidden_dim,
            'device': device,
            'torch_threads': args.torch_threads,
            'random_low_seed_offset': args.random_low_seed_offset,
            'experiment_a': (
                'checkpoint High Actor 贪心 + checkpoint Low Actor 贪心'
            ),
            'experiment_b': (
                'checkpoint High Actor 贪心 + 在同一合法候选集合中均匀随机选 quasi'
            ),
            'experiment_c': (
                '完整复用带 quasi 安全过滤的 test.py RandomDecisionPolicy；'
                '随机种子等于场景种子'
            ),
            'experiment_d': (
                '与A共用Option生命周期、合法动作域和checkpoint Low Actor；'
                '仅在High Option边界用严格remain<40 kWh阈值替换learned High，'
                'Serve被绝对安全约束屏蔽时强制Recharge'
            ),
            'experiment_a_vs_d_control': (
                'A-D的预期唯一策略差异是High决策器：'
                'A=checkpoint High贪心，D=Option边界40 kWh阈值；'
                '两者Low Actor、checkpoint、候选集合、安全掩码和Option终止规则一致'
            ),
            'quasi_safety_constraint': (
                'A/B/D 使用 observation.low_candidate_mask 中经过 '
                'MCS->quasi->最近FCS 能量过滤的候选；C 使用 '
                'RandomDecisionPolicy 的同一物理安全过滤（不应用紧急度Top-K）'
            ),
            'policy_name_note': (
                'A/B/D 的 policy_name 使用 v10 标签；实际模型和 episode '
                '以 checkpoint 与 checkpoint_metadata 字段为准'
            ),
            'low_candidate_feature_names': LOW_FEATURE_NAMES,
            'decision_time_scope': (
                '根据当前全部 observation 生成完整 action_n；不包含 env.step()'
            ),
            'ev_success_ratio_definition': (
                'success / (success + failure)'
            ),
            'paired_delta_definition': 'left_policy - right_policy',
        }
        if args.three_policy:
            config.update({
                'evaluation_mode': 'v10_v10_onlylow_random',
                'checkpoints': {
                    'v10': str(checkpoint_path),
                    'v10_onlylow': str(onlylow_checkpoint_path),
                },
                'checkpoint_metadata_by_policy': {
                    'v10': checkpoint_metadata,
                    'v10_onlylow': onlylow_checkpoint_metadata,
                },
                'policies': {
                    'v10': 'v10 checkpoint High/Low Actor 均使用确定性贪心',
                    'v10_onlylow': (
                        'v10_onlylow checkpoint Low Actor 使用确定性贪心；'
                        'High 在 Option 边界按严格 remain < 40 kWh 固定阈值决策'
                    ),
                    'Random': (
                        '复用 test.py RandomDecisionPolicy 与其物理安全过滤'
                    ),
                },
                'comparison_pairs': [list(pair) for pair in comparison_pairs],
            })
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        print(f'\n逐场景结果: {scenario_path}')
        print(f'策略汇总结果: {summary_path}')
        print(f'两两配对差异结果: {comparison_path}')
        print(f'评估配置: {config_path}')


if __name__ == '__main__':
    main()
