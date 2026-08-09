"""Analyze checkpoint reward alignment and compare v3 against v2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluation correlation analysis')
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, default=None)
    return parser.parse_args()


def correlation_rows(summary: pd.DataFrame, version: str) -> list[dict]:
    targets = {
        'ev_success_rate': 'eval_ev_success_rate',
        'mcs_incremental_capacity_share': 'eval_mcs_incremental_capacity_share',
        'mcs_match_incremental_share': 'eval_mcs_match_incremental_share',
        'broken_mcs': 'eval_broken_mcs',
        'avg_mcs_profit': 'eval_avg_mcs_profit',
    }
    rows = []
    reward = summary['eval_avg_reward'].to_numpy()
    for label, column in targets.items():
        values = summary[column].to_numpy()
        pearson = pearsonr(reward, values)
        spearman = spearmanr(reward, values)
        rows.append({
            'version': version,
            'target': label,
            'target_column': column,
            'pearson_r': float(pearson.statistic),
            'pearson_p': float(pearson.pvalue),
            'spearman_rho': float(spearman.statistic),
            'spearman_p': float(spearman.pvalue),
            'checkpoint_count': len(summary),
        })
    return rows


def add_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked['reward_rank'] = ranked['eval_avg_reward'].rank(
        ascending=False, method='min'
    ).astype(int)
    ranked['success_rank'] = ranked['eval_ev_success_rate'].rank(
        ascending=False, method='min'
    ).astype(int)
    ranked['incremental_capacity_rank'] = ranked[
        'eval_mcs_incremental_capacity_share'
    ].rank(ascending=False, method='min').astype(int)
    ranked['profit_rank'] = ranked['eval_avg_mcs_profit'].rank(
        ascending=False, method='min'
    ).astype(int)
    ranked['broken_rank'] = ranked['eval_broken_mcs'].rank(
        ascending=True, method='min'
    ).astype(int)
    return ranked.sort_values('model_episode')


def paired_comparison(v3: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    joined = v3.merge(
        v2,
        on=['model_episode', 'scenario_seed'],
        suffixes=('_v3', '_v2'),
        validate='one_to_one',
    )
    metrics = [
        ('eval_ev_success_rate', 1),
        ('eval_avg_mcs_profit', 1),
        ('eval_broken_mcs', -1),
        ('eval_avg_reward', 1),
        ('eval_mcs_incremental_capacity_share', 1),
        ('eval_mcs_match_incremental_share', 1),
        ('eval_serve_action_rate', 0),
        ('eval_recharge_action_rate', 0),
        ('eval_wait_action_rate', 0),
    ]
    rows = []
    for model_episode, group in joined.groupby('model_episode', sort=True):
        for metric, direction in metrics:
            delta = group[f'{metric}_v3'] - group[f'{metric}_v2']
            standard_error = delta.std(ddof=1) / np.sqrt(len(delta))
            rows.append({
                'model_episode': int(model_episode),
                'metric': metric,
                'v3_mean': float(group[f'{metric}_v3'].mean()),
                'v2_mean': float(group[f'{metric}_v2'].mean()),
                'mean_delta_v3_minus_v2': float(delta.mean()),
                'paired_wins': int((delta * direction > 0).sum())
                if direction else np.nan,
                'paired_ties': int((delta == 0).sum()),
                'approx_95ci_low': float(delta.mean() - 2.262 * standard_error),
                'approx_95ci_high': float(delta.mean() + 2.262 * standard_error),
            })
    return pd.DataFrame(rows)


def plot_correlations(summary: pd.DataFrame, path: Path) -> None:
    panels = [
        ('eval_ev_success_rate', 'EV success rate'),
        ('eval_mcs_incremental_capacity_share', 'MCS incremental capacity share'),
        ('eval_broken_mcs', 'Broken MCS'),
        ('eval_avg_mcs_profit', 'Average MCS profit'),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (column, label) in zip(axes.flat, panels):
        x = summary['eval_avg_reward']
        y = summary[column]
        axis.scatter(x, y, s=55)
        if len(summary) >= 2:
            coefficients = np.polyfit(x, y, 1)
            line_x = np.linspace(x.min(), x.max(), 100)
            axis.plot(line_x, np.polyval(coefficients, line_x), linewidth=1.2)
        for _, row in summary.iterrows():
            axis.annotate(
                str(int(row['model_episode'])),
                (row['eval_avg_reward'], row[column]),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
            )
        rho = spearmanr(x, y).statistic
        pearson = pearsonr(x, y).statistic
        axis.set_title(f'{label}: Spearman={rho:.3f}, Pearson={pearson:.3f}')
        axis.set_xlabel('Evaluation average reward')
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_rankings(ranked: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    episode = ranked['model_episode']
    for column, label in [
        ('reward_rank', 'Reward rank'),
        ('success_rank', 'Success-rate rank'),
        ('incremental_capacity_rank', 'Incremental-capacity rank'),
        ('profit_rank', 'Profit rank'),
    ]:
        axis.plot(episode, ranked[column], marker='o', label=label)
    axis.invert_yaxis()
    axis.set_yticks(range(1, len(ranked) + 1))
    axis.set_xlabel('Checkpoint episode')
    axis.set_ylabel('Rank (1 is best)')
    axis.set_title('Checkpoint objective rankings')
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.resolve()
    summary = pd.read_csv(evaluation_dir / 'evaluation_summary.csv')
    scenarios = pd.read_csv(evaluation_dir / 'evaluation_scenarios.csv')
    ranked = add_ranks(summary)
    ranked.to_csv(evaluation_dir / 'checkpoint_rankings.csv', index=False)

    correlations = correlation_rows(summary, 'v3')
    baseline_summary = None
    if args.baseline_dir is not None:
        baseline_dir = args.baseline_dir.resolve()
        baseline_summary = pd.read_csv(baseline_dir / 'evaluation_summary.csv')
        baseline_scenarios = pd.read_csv(
            baseline_dir / 'evaluation_scenarios.csv'
        )
        correlations.extend(correlation_rows(baseline_summary, 'v2'))
        paired = paired_comparison(scenarios, baseline_scenarios)
        paired.to_csv(
            evaluation_dir / 'v2_v3_scenario_paired.csv', index=False
        )
        v2_ranked = add_ranks(baseline_summary)
        comparison = ranked.merge(
            v2_ranked,
            on='model_episode',
            suffixes=('_v3', '_v2'),
            validate='one_to_one',
        )
        comparison.to_csv(
            evaluation_dir / 'v2_v3_checkpoint_comparison.csv', index=False
        )

    correlation_table = pd.DataFrame(correlations)
    correlation_table.to_csv(
        evaluation_dir / 'reward_correlations.csv', index=False
    )
    plot_correlations(summary, evaluation_dir / 'reward_correlation_scatter.png')
    plot_rankings(ranked, evaluation_dir / 'checkpoint_rankings.png')

    reward_best = ranked.loc[ranked['reward_rank'].idxmin()]
    success_best = ranked.loc[ranked['success_rank'].idxmin()]
    success_rho = correlation_table.query(
        "version == 'v3' and target == 'ev_success_rate'"
    ).iloc[0]['spearman_rho']
    incremental_rho = correlation_table.query(
        "version == 'v3' and target == 'mcs_incremental_capacity_share'"
    ).iloc[0]['spearman_rho']
    criteria = pd.DataFrame([{
        'reward_success_spearman_gt_0_5': bool(success_rho > 0.5),
        'reward_incremental_capacity_positive': bool(incremental_rho > 0.0),
        'reward_best_success_in_top3': bool(reward_best['success_rank'] <= 3),
        'success_best_reward_in_top3': bool(success_best['reward_rank'] <= 3),
        'no_action_rate_above_0_9': bool(
            summary[[
                'eval_serve_action_rate',
                'eval_recharge_action_rate',
                'eval_wait_action_rate',
            ]].to_numpy().max() < 0.9
        ),
        'reward_best_episode': int(reward_best['model_episode']),
        'reward_best_success_rank': int(reward_best['success_rank']),
        'success_best_episode': int(success_best['model_episode']),
        'success_best_reward_rank': int(success_best['reward_rank']),
    }])
    criteria.to_csv(evaluation_dir / 'phase1_criteria.csv', index=False)
    print(correlation_table.to_string(index=False))
    print(ranked[[
        'model_episode', 'reward_rank', 'success_rank',
        'incremental_capacity_rank', 'profit_rank', 'broken_rank',
    ]].to_string(index=False))
    print(criteria.to_string(index=False))


if __name__ == '__main__':
    main()
