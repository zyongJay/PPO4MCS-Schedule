"""Train v13: v10_onlylow with system-total-profit reward and checkpointing.

No simulation, matcher, or base reward source file is modified.  The runtime
adapter in ``v13_reward.py`` supplies the alternate profit term only to this
entry point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import train as base_train
from environment import MultiAgentEnv
from v13_reward import SystemProfitRewardBuilder


SYSTEM_PROFIT_METRIC = 'rolling40_success_then_system_total_profit'


class V13MultiAgentEnv(MultiAgentEnv):
    """Standard simulator with a v13-only reward builder and profit snapshot."""

    instances = {}

    def __init__(self, seed=42):
        super().__init__(seed)
        self.seed = int(seed)
        self.world.reward_builder = SystemProfitRewardBuilder()
        self.__class__.instances[self.seed] = self

    def system_total_profit(self) -> float:
        return float(
            sum(mcs.total_profit for mcs in self.world.MCSs)
            + sum(fcs.total_profit for fcs in self.world.FCSs)
        )

    def system_profit_metrics(self):
        total_mcs_profit = float(sum(
            mcs.total_profit for mcs in self.world.MCSs
        ))
        total_fcs_profit = float(sum(
            fcs.total_profit for fcs in self.world.FCSs
        ))
        return {
            'total_mcs_profit': total_mcs_profit,
            'total_fcs_profit': total_fcs_profit,
            'system_total_profit': total_mcs_profit + total_fcs_profit,
            'system_total_profit_definition': (
                'sum_mcs_total_profit_plus_sum_fcs_total_profit'
            ),
        }

    def step(self, action_n):
        # This is MultiAgentEnv.step with one non-simulation addition: snapshot
        # provider profit around the normal transition, then attach that delta
        # to reward events before World.mix_get_reward_n() is called.
        before_profit = self.system_total_profit()
        self.world.update(action_n)
        self.world.step_finish()
        self.world.match_and_get_neibor()
        after_profit = self.system_total_profit()
        system_delta = after_profit - before_profit
        share = system_delta / max(len(self.world.MCSs), 1)
        for event in self.world.mcs_step_events.values():
            event['system_total_profit_delta'] = float(system_delta)
            event['system_total_profit_share_delta'] = float(share)
        new_obs_n, old_obs_n, done_n = self.world.get_obs_n()
        reward_n = self.world.mix_get_reward_n()
        self.world.last_system_reward_event.update({
            'system_total_profit_delta': float(system_delta),
            'system_total_profit_share_delta': float(share),
        })
        return new_obs_n, old_obs_n, reward_n, done_n


_collect_episode = base_train.collect_episode


def collect_episode_with_system_profit(agent, args, episode):
    result = _collect_episode(agent, args, episode)
    environment = V13MultiAgentEnv.instances.pop(int(args.seed + episode - 1))
    result[0].update(environment.system_profit_metrics())
    return result


def checkpoint_selection_system_profit(episode_rows):
    """Secondary selection target after the unchanged success-rate filter."""
    if not episode_rows:
        raise ValueError('checkpoint selection lacks episode rows')
    return float(np.mean([
        float(row['system_total_profit']) for row in episode_rows
    ]))


def _rename_selection_columns(output_dir: Path):
    """Make v13 artifacts self-describing after reusing train.py's loop."""
    log_path = output_dir / 'training_log.csv'
    if log_path.is_file():
        table = pd.read_csv(log_path)
        table = table.rename(columns={
            'update_mcs_profit': 'update_system_total_profit',
            'checkpoint_selection_profit': (
                'checkpoint_selection_system_total_profit'
            ),
            'best_mcs_profit_within_success_tolerance': (
                'best_system_total_profit_within_success_tolerance'
            ),
        })
        table.to_csv(log_path, index=False)
        if 'system_total_profit' in table:
            figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
            values = table['system_total_profit'].rolling(
                window=20, min_periods=1
            ).mean()
            axis.plot(table['episode'], values, color='#264653')
            axis.set_xlabel('Episode')
            axis.set_ylabel('Provider total profit')
            axis.set_title('v13 System Total Profit (rolling 20)')
            axis.grid(alpha=0.25)
            figure.savefig(output_dir / 'system_profit_curve.png', dpi=160)
            plt.close(figure)

    config_path = output_dir / 'training_config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config.update({
            'low_reward_design_version': (
                'v13_system_total_profit_share_event'
            ),
            'profit_reward_definition': (
                'signed_step_delta(sum_mcs_total_profit_plus_sum_fcs_total_profit)'
                '/number_of_mcss'
            ),
            'checkpoint_selection_metric': SYSTEM_PROFIT_METRIC,
            'checkpoint_selection_secondary_objective': (
                'system_total_profit'
            ),
            'checkpoint_selection_system_profit_definition': (
                'sum_mcs_total_profit_plus_sum_fcs_total_profit'
            ),
        })
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    for checkpoint_path in output_dir.glob('*.pt'):
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        metadata = checkpoint.get('metadata')
        if not isinstance(metadata, dict):
            continue
        if 'selected_checkpoint_mcs_profit' in metadata:
            metadata['selected_checkpoint_system_total_profit'] = metadata.pop(
                'selected_checkpoint_mcs_profit'
            )
        if 'best_mcs_profit_within_success_tolerance' in metadata:
            metadata['best_system_total_profit_within_success_tolerance'] = (
                metadata.pop('best_mcs_profit_within_success_tolerance')
            )
        if 'checkpoint_selection_profit' in metadata:
            metadata['checkpoint_selection_system_total_profit'] = (
                metadata.pop('checkpoint_selection_profit')
            )
        metadata['checkpoint_selection_metric'] = SYSTEM_PROFIT_METRIC
        checkpoint['metadata'] = metadata
        torch.save(checkpoint, checkpoint_path)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--fixed-high-threshold' not in arguments:
        arguments.append('--fixed-high-threshold')
    if '--output-dir' not in arguments:
        arguments.extend([
            '--output-dir',
            str(Path(__file__).resolve().parent.parent / 'training_results_v13'),
        ])
    sys.argv = [sys.argv[0], *arguments]
    base_train.MultiAgentEnv = V13MultiAgentEnv
    base_train.collect_episode = collect_episode_with_system_profit
    base_train.checkpoint_selection_profit = checkpoint_selection_system_profit
    original_parse_args = base_train.parse_args

    def parse_args_with_v13_metadata():
        args = original_parse_args()
        args.profit_reward_objective = 'system_total_profit'
        args.profit_reward_definition = (
            'signed_provider_profit_delta_shared_equally_across_mcss'
        )
        args.checkpoint_selection_secondary_objective = 'system_total_profit'
        return args

    base_train.parse_args = parse_args_with_v13_metadata
    base_train.main()
    _rename_selection_columns(Path(args_output_dir(arguments)))


def args_output_dir(arguments):
    """Resolve the same output default/override used by main()."""
    for index, value in enumerate(arguments):
        if value == '--output-dir' and index + 1 < len(arguments):
            return arguments[index + 1]
    return Path(__file__).resolve().parent.parent / 'training_results_v13'


if __name__ == '__main__':
    main()
