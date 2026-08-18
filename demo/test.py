"""在固定场景中统一评测多种 MCS 调度策略。

支持的策略：

1. 一个或多个指定 checkpoint 的 RL 策略；
2. 与 run_test.py 一致的随机调度策略；
3. 不调度策略：IEV 沿轨迹行驶，MCS 始终停在原地且不主动补电。

默认评测 V1（training_results_gpu）以及 V2--V5 的 Episode 300 checkpoint，
并同时评测随机调度和不调度两款基线。所有策略共享 20 个固定场景。

示例（显式指定新模型时，结果会追加/更新到已有结果文件）：

    python test.py \
        --checkpoints ../training_results_v6/model_episode_300.pt \
        --baselines \
        --seeds 1001 1002 1003 1004 1005

默认会把逐场景结果、跨场景汇总和评测配置保存到项目根目录下的
``test_results``。逐场景结果按“策略名称 + 场景种子”幂等追加，重复测试同一
模型与场景时会更新原记录而不生成重复行。使用 ``--no-save`` 可仅在终端显示结果。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

import world as world_module
from config import (
    MAX_STEPS_PER_EPISODE,
    MCS_CRITIC_STATE_DIM,
    MCS_FEAT_DIM_self,
    MCS_FEAT_DIM_tgt,
    MCS_HIGH_FEAT_DIM,
    MCS_RECHARGE_THRESHOLD,
    TRACK_DATA_PATH,
)
from core import EV, MCS
from environment import MultiAgentEnv
from matching import RechargeMatcher
from network import MCSMAPPOAgent, MCS_ACTION_NAMES
from observation import MCS_STAY_CANDIDATE_ID


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'test_results'
DEFAULT_SEEDS = tuple(range(1001, 1021))
# 第一版完整 CUDA 训练保存在 training_results_gpu；training_results 是更早的
# 预备训练目录，因此默认的五版模型采用 V1(CUDA) 与 V2--V5。
DEFAULT_CHECKPOINTS = (
    PROJECT_DIR / 'training_results_gpu' / 'model_episode_300.pt',
    PROJECT_DIR / 'training_results_v2' / 'model_episode_300.pt',
    PROJECT_DIR / 'training_results_v3' / 'model_episode_300.pt',
    PROJECT_DIR / 'training_results_v4' / 'model_episode_300.pt',
    PROJECT_DIR / 'training_results_v5' / 'model_episode_300.pt',
)

# config.py 中的轨迹路径以 demo 目录为基准。转成绝对路径后，从项目
# 根目录或 demo 目录启动本脚本都能读取相同的轨迹数据。
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


@dataclass(frozen=True)
class PolicySpec:
    """一项待评测策略的静态描述。"""

    policy_name: str
    policy_type: str
    checkpoint_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='在相同固定场景中评测 RL、随机和不调度策略'
    )
    parser.add_argument(
        '--checkpoints',
        type=Path,
        nargs='*',
        default=list(DEFAULT_CHECKPOINTS),
        help='一个或多个 RL checkpoint 路径；默认评测 V1--V5 的 Episode 300',
    )
    parser.add_argument(
        '--baselines',
        nargs='*',
        choices=('random', 'no_schedule'),
        default=['random', 'no_schedule'],
        help='要评测的基线；传入空列表可只评测 checkpoint',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(DEFAULT_SEEDS),
        help='所有策略共同使用的固定场景种子',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=MAX_STEPS_PER_EPISODE,
    )
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto'
    )
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存 CSV/JSON，只在终端输出汇总结果',
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定了 CUDA，但当前 PyTorch 无法使用 CUDA')
    if requested == 'cuda':
        return 'cuda'
    if requested == 'auto' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def checkpoint_policy_name(path: Path) -> str:
    """生成能够区分不同模型目录中同名 checkpoint 的名称。"""
    return f'rl:{path.parent.name}/{path.stem}'


def build_policy_specs(args: argparse.Namespace) -> List[PolicySpec]:
    specs: List[PolicySpec] = []
    seen_paths = set()
    seen_names = set()

    for raw_path in args.checkpoints:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'checkpoint 不存在: {path}')
        if path in seen_paths:
            continue
        seen_paths.add(path)
        name = checkpoint_policy_name(path)
        if name in seen_names:
            raise ValueError(f'策略名称重复，无法区分 checkpoint: {name}')
        seen_names.add(name)
        specs.append(PolicySpec(name, 'rl', path))

    for baseline in args.baselines:
        name = 'random' if baseline == 'random' else 'no_schedule'
        if name in seen_names:
            continue
        seen_names.add(name)
        specs.append(PolicySpec(name, baseline))

    if not specs:
        raise ValueError('至少需要指定一个 checkpoint 或一种 baseline')
    return specs


def iev_track_action(ev: EV) -> Dict:
    """IEV 沿当前轨迹向下一个轨迹点移动。"""
    if ev.track and ev.track_index + 1 < len(ev.track):
        point = ev.track[ev.track_index + 1]
        target = [float(point[0]), float(point[1])]
    else:
        target = [float(ev.pos[0]), float(ev.pos[1])]
    return {'target_pos': target}


class DecisionPolicy:
    """策略统一接口：根据当前全部 agent 和 obs 生成完整 action_n。"""

    def build_actions(
        self,
        env: MultiAgentEnv,
        acting_agents: Sequence,
        observations: Sequence[Dict],
    ) -> List[Dict]:
        raise NotImplementedError

    def synchronize(self) -> None:
        """等待异步推理完成；CPU/非 RL 策略无需处理。"""


class RLDecisionPolicy(DecisionPolicy):
    """使用 High/Low Actor 进行确定性贪心决策。"""

    def __init__(self, agent: MCSMAPPOAgent):
        self.agent = agent
        self.recharge_matcher = RechargeMatcher()

    def synchronize(self) -> None:
        if self.agent.device.type == 'cuda':
            torch.cuda.synchronize(self.agent.device)

    @torch.no_grad()
    def _greedy_mcs_actions(
        self, observations: Sequence[Dict]
    ) -> List[Dict]:
        if not observations:
            return []

        high_states = self.agent._tensor(np.stack([
            observation['high_state'] for observation in observations
        ]))
        high_masks = self.agent._tensor(np.stack([
            observation['high_action_mask'] for observation in observations
        ]), dtype=torch.bool)
        high_distribution = self.agent.high_actor.distribution(
            high_states, high_masks
        )
        high_actions = high_distribution.probs.argmax(dim=-1)
        high_indices = high_actions.detach().cpu().numpy().astype(int)

        results = [{
            'mode': MCS_ACTION_NAMES[action_index],
            'low_action': -1,
        } for action_index in high_indices]

        serve_indices = np.flatnonzero(high_indices == 0)
        if serve_indices.size:
            low_self_states = self.agent._tensor(np.stack([
                observations[index]['low_self_state']
                for index in serve_indices
            ]))
            low_candidates = self.agent._tensor(np.stack([
                observations[index]['low_candidates']
                for index in serve_indices
            ]))
            low_masks = self.agent._tensor(np.stack([
                observations[index]['low_candidate_mask']
                for index in serve_indices
            ]), dtype=torch.bool)
            low_distribution = self.agent.low_actor.distribution(
                low_self_states, low_candidates, low_masks
            )
            low_actions = (
                low_distribution.probs.argmax(dim=-1)
                .detach().cpu().numpy().astype(int)
            )
            for batch_index, observation_index in enumerate(serve_indices):
                results[int(observation_index)]['low_action'] = int(
                    low_actions[batch_index]
                )
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
        mcs_observations = [
            observation_by_agent[mcs] for mcs in acting_mcss
        ]
        selected_actions = self._greedy_mcs_actions(mcs_observations)
        selected_by_id = {
            mcs.id: (observation, action)
            for mcs, observation, action in zip(
                acting_mcss, mcs_observations, selected_actions
            )
        }

        recharge_requests = [
            mcs for mcs, action in zip(acting_mcss, selected_actions)
            if action['mode'] == 'Recharge'
        ]
        recharge_results = self.recharge_matcher.match_all(
            recharge_requests, env.world.FCSs
        )
        recharge_matched_ids = {
            int(result['mcs_id'])
            for result in recharge_results if result.get('success')
        }
        ev_by_id = {ev.id: ev for ev in env.world.EVs}

        action_n = []
        for actor in acting_agents:
            if isinstance(actor, EV):
                action_n.append(iev_track_action(actor))
                continue

            observation, action = selected_by_id[actor.id]
            if action['mode'] == 'Recharge':
                matched = actor.id in recharge_matched_ids
                action_n.append({
                    'mode': 'Recharge' if matched else 'Wait',
                    'requested_mode': 'Recharge',
                    'recharge_matched': matched,
                    'target_pos': (
                        list(actor.current_target_pos)
                        if matched else list(actor.pos)
                    ),
                })
            else:
                candidate_id = int(
                    observation['candidate_ids'][action['low_action']]
                )
                low_stay_selected = candidate_id == MCS_STAY_CANDIDATE_ID
                has_quasi_candidate = bool(
                    observation.get('quasi_candidate_count', 0) > 0
                )
                target_pos = (
                    list(actor.pos)
                    if low_stay_selected
                    else list(ev_by_id[candidate_id].pos)
                )
                action_n.append({
                    'mode': 'Serve',
                    'requested_mode': 'Serve',
                    'recharge_matched': False,
                    'high_action_mask': observation['high_action_mask'].tolist(),
                    'target_pos': target_pos,
                    'low_candidate_id': candidate_id,
                    'low_stay_selected': low_stay_selected,
                    'low_forced_stay': bool(
                        low_stay_selected and not has_quasi_candidate
                    ),
                    'has_quasi_candidate': has_quasi_candidate,
                })
        return action_n


class RandomDecisionPolicy(DecisionPolicy):
    """复现 run_test.py 的随机 MCS 调度规则。"""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.recharge_matcher = RechargeMatcher()

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

        # 只有低于补电阈值、空闲且当前存在合法 FCS 的 MCS 才申请补电。
        recharge_mcss = []
        for actor in acting_agents:
            if not isinstance(actor, MCS) or not actor.is_idle:
                continue
            if actor.remain >= MCS_RECHARGE_THRESHOLD:
                continue
            high_mask = observation_by_agent.get(actor, {}).get(
                'high_action_mask', []
            )
            if len(high_mask) > 1 and bool(high_mask[1]):
                recharge_mcss.append(actor)

        recharge_results = self.recharge_matcher.match_all(
            recharge_mcss, env.world.FCSs
        )
        recharge_request_ids = {mcs.id for mcs in recharge_mcss}
        recharge_matched_ids = {
            int(result['mcs_id'])
            for result in recharge_results if result.get('success')
        }

        action_n = []
        for actor in acting_agents:
            if isinstance(actor, EV):
                action_n.append(iev_track_action(actor))
            elif actor.id in recharge_matched_ids:
                action_n.append({
                    'mode': 'Recharge',
                    'requested_mode': 'Recharge',
                    'recharge_matched': True,
                    'target_pos': list(actor.current_target_pos),
                })
            elif actor.id in recharge_request_ids:
                # 补电请求未抢到充电位时，本 step 原地等待。
                action_n.append({
                    'mode': 'Wait',
                    'requested_mode': 'Recharge',
                    'recharge_matched': False,
                    'target_pos': list(actor.pos),
                })
            else:
                quasi_candidates = [
                    ev for ev in actor.near_quasi if ev.is_quasi
                ]
                if quasi_candidates:
                    target = self.rng.choice(quasi_candidates)
                    action_n.append({
                        'mode': 'Serve',
                        'requested_mode': 'Serve',
                        'recharge_matched': False,
                        'target_pos': list(target.pos),
                    })
                else:
                    action_n.append({
                        'mode': 'Wait',
                        'requested_mode': 'Wait',
                        'recharge_matched': False,
                        'target_pos': list(actor.pos),
                    })
        return action_n


class NoScheduleDecisionPolicy(DecisionPolicy):
    """IEV 正常行驶，MCS 不移动且不主动申请补电。"""

    def build_actions(
        self,
        env: MultiAgentEnv,
        acting_agents: Sequence,
        observations: Sequence[Dict],
    ) -> List[Dict]:
        del env, observations
        action_n = []
        for actor in acting_agents:
            if isinstance(actor, EV):
                action_n.append(iev_track_action(actor))
            else:
                action_n.append({
                    'mode': 'Wait',
                    'requested_mode': 'Wait',
                    'recharge_matched': False,
                    'target_pos': list(actor.pos),
                })
        return action_n


def load_rl_agent(
    checkpoint_path: Path,
    hidden_dim: int,
    device: str,
) -> tuple[MCSMAPPOAgent, Dict]:
    agent = MCSMAPPOAgent(
        high_state_dim=MCS_HIGH_FEAT_DIM,
        low_self_dim=MCS_FEAT_DIM_self,
        low_candidate_dim=MCS_FEAT_DIM_tgt,
        critic_state_dim=MCS_CRITIC_STATE_DIM,
        hidden_dim=hidden_dim,
        device=device,
    )
    metadata = agent.load(checkpoint_path)
    agent.eval()
    return agent, metadata


def build_decision_policy(
    spec: PolicySpec,
    seed: int,
    agent: MCSMAPPOAgent | None,
) -> DecisionPolicy:
    if spec.policy_type == 'rl':
        if agent is None:
            raise RuntimeError('RL 策略缺少已加载的 agent')
        return RLDecisionPolicy(agent)
    if spec.policy_type == 'random':
        return RandomDecisionPolicy(seed)
    if spec.policy_type == 'no_schedule':
        return NoScheduleDecisionPolicy()
    raise ValueError(f'未知策略类型: {spec.policy_type}')


def remember_charge_providers(
    evs: Iterable[EV],
    provider_by_ev_id: Dict[int, str],
) -> None:
    """在 EV 完成任务并清空 charge_provider_type 前保存充电方类型。"""
    for ev in evs:
        provider_type = str(getattr(ev, 'charge_provider_type', ''))
        if ev.is_charged and provider_type in ('MCS', 'FCS'):
            provider_by_ev_id.setdefault(int(ev.id), provider_type)


def mean_attribute(items: Sequence, attribute: str) -> float:
    if not items:
        return 0.0
    return float(np.mean([
        float(getattr(item, attribute)) for item in items
    ]))


def evaluate_scenario(
    spec: PolicySpec,
    scenario_seed: int,
    max_steps: int,
    agent: MCSMAPPOAgent | None,
    checkpoint_metadata: Dict,
) -> Dict:
    """在一个固定场景中运行一项策略并返回统一业务指标。"""
    random.seed(scenario_seed)
    np.random.seed(scenario_seed)
    torch.manual_seed(scenario_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(scenario_seed)

    env = MultiAgentEnv(scenario_seed)
    env.world.verbose = False
    observations = env.reset()
    policy = build_decision_policy(spec, scenario_seed, agent)

    provider_by_ev_id: Dict[int, str] = {}
    remember_charge_providers(env.world.EVs, provider_by_ev_id)
    decision_times_ms = []
    executed_steps = 0

    for _ in range(max_steps):
        acting_agents = list(env.world.agents)

        # CUDA 推理是异步的。计时前后同步，确保测到的是实际完成动作
        # 生成所需的时间，而不是仅测到 CUDA kernel 的提交时间。
        policy.synchronize()
        decision_start_ns = time.perf_counter_ns()
        action_n = policy.build_actions(
            env, acting_agents, observations
        )
        policy.synchronize()
        decision_elapsed_ms = (
            time.perf_counter_ns() - decision_start_ns
        ) / 1_000_000.0
        decision_times_ms.append(float(decision_elapsed_ms))

        if len(action_n) != len(acting_agents):
            raise RuntimeError(
                f'{spec.policy_name}: action_n 与 acting_agents 数量不一致'
            )

        # 仿真推进严格放在决策计时区间之外。
        observations, _, _, _ = env.step(action_n)
        executed_steps += 1
        remember_charge_providers(env.world.EVs, provider_by_ev_id)
        if env.world.get_done():
            break

    evs = list(env.world.EVs)
    mcss = list(env.world.MCSs)
    fcss = list(env.world.FCSs)
    # 当前仿真在匹配阶段已经检查充电任务的物理可行性；与 train.py
    # 保持一致，is_charged=True 即计为成功充电。
    successful_evs = [ev for ev in evs if ev.is_charged]
    failure_count = sum(ev.fail_charge for ev in evs)
    success_count = len(successful_evs)
    finished_count = success_count + failure_count
    unresolved_count = sum(
        not ev.is_normal
        and not ev.fail_charge
        and not ev.is_charged
        for ev in evs
    )

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

    success_denominator = max(success_count, 1)
    checkpoint_episode = checkpoint_metadata.get('episode', '')
    if not checkpoint_episode and spec.checkpoint_path is not None:
        match = re.search(r'(\d+)', spec.checkpoint_path.stem)
        checkpoint_episode = int(match.group(1)) if match else ''

    # FCS 当前实现中的实际字段名是 total_idle_time_min。保留显式名称，
    # 避免把分钟误解为 step 数或其他时间单位。
    row = {
        'policy_name': spec.policy_name,
        'policy_type': spec.policy_type,
        'checkpoint_path': (
            str(spec.checkpoint_path) if spec.checkpoint_path else ''
        ),
        'checkpoint_episode': checkpoint_episode,
        'scenario_seed': int(scenario_seed),
        'steps': int(executed_steps),
        # 与 train.py 一致：成功数 / (成功数 + 失败数)。另附全部 EV
        # 分母的比例，便于存在 unresolved EV 时检查口径差异。
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
        'decision_time_std_ms': float(np.std(decision_times_ms, ddof=0)),
        'decision_time_p95_ms': float(np.percentile(
            decision_times_ms, 95
        )),
        'successful_ev_mcs_count': int(provider_counts['MCS']),
        'successful_ev_fcs_count': int(provider_counts['FCS']),
        'successful_ev_mcs_share': (
            provider_counts['MCS'] / success_denominator
        ),
        'successful_ev_fcs_share': (
            provider_counts['FCS'] / success_denominator
        ),
        'broken_mcs_count': int(sum(mcs.is_broken for mcs in mcss)),
    }
    return row


def build_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    """按策略聚合场景均值、标准差和全局充电方占比。"""
    identity_columns = {
        'policy_name', 'policy_type', 'checkpoint_path',
        'checkpoint_episode', 'scenario_seed',
    }
    numeric_columns = [
        column for column in scenarios.columns
        if column not in identity_columns
        and pd.api.types.is_numeric_dtype(scenarios[column])
    ]

    summary_rows = []
    for policy_name, group in scenarios.groupby('policy_name', sort=False):
        first = group.iloc[0]
        row = {
            'policy_name': policy_name,
            'policy_type': first['policy_type'],
            'checkpoint_path': first['checkpoint_path'],
            'checkpoint_episode': first['checkpoint_episode'],
            'scenario_count': int(len(group)),
        }
        for column in numeric_columns:
            row[column] = float(group[column].mean())
            row[f'{column}_std_across_scenarios'] = float(
                group[column].std(ddof=0)
            )

        # 充电方占比使用所有场景成功 EV 数量汇总后再计算，避免对成功数
        # 不同的场景做简单比例平均。
        mcs_successes = float(group['successful_ev_mcs_count'].sum())
        fcs_successes = float(group['successful_ev_fcs_count'].sum())
        provider_total = mcs_successes + fcs_successes
        row['successful_ev_mcs_share'] = (
            mcs_successes / provider_total if provider_total else 0.0
        )
        row['successful_ev_fcs_share'] = (
            fcs_successes / provider_total if provider_total else 0.0
        )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def print_summary(summary: pd.DataFrame) -> None:
    columns = [
        'policy_name',
        'ev_charge_success_ratio',
        'avg_mcs_profit',
        'avg_mcs_idle_time_min',
        'avg_fcs_profit',
        'avg_fcs_idle_time_min',
        'avg_ev_extra_distance_km',
        'avg_ev_charging_delay_min',
        'avg_decision_time_ms',
        'successful_ev_mcs_share',
        'successful_ev_fcs_share',
    ]
    printable = summary[columns].copy()
    print('\n固定场景评测汇总（跨场景均值）')
    print(printable.to_string(index=False, float_format=lambda x: f'{x:.6f}'))


def merge_scenario_results(
    existing: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """按策略和场景种子幂等合并，支持以后直接追加新模型测试结果。"""
    if existing.empty:
        merged = current.copy()
    else:
        merged = pd.concat([existing, current], ignore_index=True, sort=False)

    merge_key = ['policy_name', 'scenario_seed']
    missing_key = [column for column in merge_key if column not in merged.columns]
    if missing_key:
        raise ValueError(f'逐场景结果缺少合并键: {missing_key}')

    # 当前测试记录放在后面，因此重复键保留本次重新评测的最新结果。
    merged = merged.drop_duplicates(subset=merge_key, keep='last')
    merged = merged.sort_values(merge_key, kind='stable').reset_index(drop=True)
    return merged


def append_config_history(config_path: Path, current_run: Dict) -> Dict:
    """保留历史测试配置，并追加记录本次评测参数。"""
    history = []
    if config_path.is_file():
        try:
            existing = json.loads(config_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f'无法读取已有评测配置: {config_path}') from exc

        if isinstance(existing, dict) and isinstance(
            existing.get('evaluation_runs'), list
        ):
            history.extend(existing['evaluation_runs'])
        elif isinstance(existing, dict) and existing:
            # 兼容旧版单次配置结构，首次追加时将其迁入历史列表。
            history.append(existing)

    history.append(current_run)
    return {
        'scenario_merge_key': ['policy_name', 'scenario_seed'],
        'duplicate_policy': 'keep_latest',
        'evaluation_runs': history,
    }


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError('--max-steps 必须大于 0')
    if args.hidden_dim <= 0 or args.torch_threads <= 0:
        raise ValueError('--hidden-dim 和 --torch-threads 必须大于 0')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('--seeds 中不能包含重复种子')

    specs = build_policy_specs(args)
    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)

    scenario_rows = []
    metadata_by_policy: Dict[str, Dict] = {}
    print(
        f'device={device} policies={len(specs)} '
        f'scenarios_per_policy={len(args.seeds)} max_steps={args.max_steps}'
    )

    for spec in specs:
        agent = None
        metadata: Dict = {}
        if spec.policy_type == 'rl':
            agent, metadata = load_rl_agent(
                spec.checkpoint_path, args.hidden_dim, device
            )
        metadata_by_policy[spec.policy_name] = metadata

        for seed in args.seeds:
            row = evaluate_scenario(
                spec=spec,
                scenario_seed=seed,
                max_steps=min(args.max_steps, MAX_STEPS_PER_EPISODE),
                agent=agent,
                checkpoint_metadata=metadata,
            )
            scenario_rows.append(row)
            print(
                f'{spec.policy_name} seed={seed} '
                f'success={row["ev_charge_success_ratio"]:.4f} '
                f'mcs_profit={row["avg_mcs_profit"]:.2f} '
                f'decision={row["avg_decision_time_ms"]:.3f}ms'
            )

        # 及时释放上一个 checkpoint 的模型和优化器显存。
        del agent
        if device == 'cuda':
            torch.cuda.empty_cache()

    current_scenario_table = pd.DataFrame(scenario_rows)
    current_summary_table = build_summary(current_scenario_table)
    print_summary(current_summary_table)

    if not args.no_save:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = output_dir / 'evaluation_scenarios.csv'
        summary_path = output_dir / 'evaluation_summary.csv'
        config_path = output_dir / 'evaluation_config.json'
        existing_scenarios = (
            pd.read_csv(scenario_path) if scenario_path.is_file()
            else pd.DataFrame()
        )
        scenario_table = merge_scenario_results(
            existing_scenarios, current_scenario_table
        )
        summary_table = build_summary(scenario_table)
        scenario_table.to_csv(scenario_path, index=False)
        summary_table.to_csv(summary_path, index=False)
        current_config = {
            'evaluated_at': datetime.now().astimezone().isoformat(
                timespec='seconds'
            ),
            'checkpoints': [
                str(spec.checkpoint_path)
                for spec in specs if spec.checkpoint_path is not None
            ],
            'baselines': [
                spec.policy_type for spec in specs
                if spec.policy_type != 'rl'
            ],
            'seeds': list(args.seeds),
            'max_steps': min(args.max_steps, MAX_STEPS_PER_EPISODE),
            'hidden_dim': args.hidden_dim,
            'device': device,
            'torch_threads': args.torch_threads,
            'rl_deterministic': True,
            'decision_time_scope': (
                '从全部当前 obs 生成完整 action_n；不包含 env.step()'
            ),
            'ev_success_ratio_definition': (
                'success / (success + failure)'
            ),
            'checkpoint_metadata': metadata_by_policy,
        }
        config_history = append_config_history(config_path, current_config)
        config_path.write_text(
            json.dumps(config_history, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'\n逐场景结果: {scenario_path}')
        print(f'汇总结果: {summary_path}')
        print(f'评测配置: {config_path}')


if __name__ == '__main__':
    main()
