"""Export descriptive tables and figures from completed, frozen experiments.

No fitting, checkpoint selection, significance test or research report.
Stock averages are computed within seed first. Three seeds describe optimizer
variation; they do not provide independent market samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {'I': 'PPO', 'J': 'Fixed risk penalty', 'K': 'PPO-Lagrangian'}
COLORS = {'I': '#386cb0', 'J': '#e69f00', 'K': '#009e73'}


def export(root: Path, output: Path):
    if output.exists():
        raise ValueError('Choose a new output directory; frozen exports are not overwritten')
    study = root / 'global-risk-study/frozen_evaluation'
    freeze = json.loads((study / 'selection_freeze.json').read_text())
    completed = json.loads((study / 'completion.json').read_text())
    if not completed['completed'] or completed['runs'] != 27:
        raise ValueError('Incomplete frozen evaluation')
    metrics = pd.read_csv(study / 'metrics.csv', dtype={'symbol': str})
    keys = ['fold', 'condition', 'seed', 'split', 'symbol']
    if len(metrics) != 135 or metrics.duplicated(keys).any():
        raise ValueError('Expected 27 frozen models evaluated on five symbols each')
    sources = [study / 'metrics.csv', study / 'selection_freeze.json']
    stress_parts = []
    for condition in LABELS:
        for seed in (42, 43, 44):
            shard = root / 'stress-shards' / f'{condition}-seed{seed}'
            done = json.loads((shard / 'completion.json').read_text())
            part = pd.read_csv(shard / 'metrics.csv', dtype={'symbol': str})
            if (not done['completed'] or done['refitting'] or len(part) != 55
                    or set(part.condition) != {condition} or set(part.seed) != {seed}):
                raise ValueError(f'Incomplete or mismatched stress shard: {shard}')
            stress_parts.append(part)
            sources.extend([shard / 'metrics.csv', shard / 'audit_freeze.json'])
    stress = pd.concat(stress_parts, ignore_index=True)
    if stress.duplicated(['condition', 'seed', 'symbol', 'stress_date']).any():
        raise ValueError('Duplicate stress evaluation')
    output.mkdir(parents=True)
    fields = ['net_return', 'bar_max_drawdown', 'turnover', 'friction_cost',
              'pre_cost_return_addback', 'mean_drawdown_excess', 'flat_fraction',
              'mean_exposure', 'trade_count']
    metrics.to_csv(output / 'frozen_test_by_symbol.csv', index=False)
    per_seed = metrics.groupby(['fold', 'condition', 'seed'], as_index=False)[fields].mean()
    per_seed.to_csv(output / 'frozen_test_by_seed.csv', index=False)
    summary = per_seed.groupby(['fold', 'condition'])[fields].agg(['mean', 'min', 'max'])
    summary.columns = ['_'.join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(output / 'frozen_test_summary.csv', index=False)

    # Paired seed differences are descriptive: do not treat correlated stocks,
    # overlapping folds or three optimizer seeds as independent market trials.
    effects = []
    reference = per_seed.loc[per_seed.condition == 'I'].set_index(['fold', 'seed'])
    for condition in ('J', 'K'):
        candidate = per_seed.loc[per_seed.condition == condition].set_index(['fold', 'seed'])
        difference = candidate[fields] - reference[fields]
        difference['comparison'] = f'{condition}-I'
        effects.append(difference.reset_index())
    pd.concat(effects).to_csv(output / 'paired_seed_differences.csv', index=False)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [('net_return', 100, 'Net return (%)'),
              ('bar_max_drawdown', -100, 'Maximum drawdown magnitude (%)'),
              ('turnover', 1, 'Turnover (equity multiples)'),
              ('mean_drawdown_excess', 1, 'Mean drawdown excess (risk cost)')]
    folds = sorted(per_seed.fold.unique())
    for axis, (field, scale, title) in zip(axes.flat, panels):
        for offset, condition in zip((-.22, 0, .22), LABELS):
            data = per_seed.loc[per_seed.condition == condition]
            for fold in folds:
                values = data.loc[data.fold == fold, field].to_numpy() * scale
                x = fold + offset
                axis.scatter(x + np.linspace(-.035, .035, len(values)), values,
                             color=COLORS[condition], alpha=.7, s=25)
                axis.plot([x-.07, x+.07], [values.mean()]*2, color=COLORS[condition], lw=3,
                          label=LABELS[condition] if fold == folds[0] else None)
        if field == 'mean_drawdown_excess':
            axis.axhline(freeze['risk_budget'], ls='--', color='black', lw=1, label='Budget')
        else:
            axis.axhline(0, color='gray', lw=.7)
        axis.set(title=title, xlabel='Historical fold', xticks=folds)
        axis.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=9)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('Frozen historical tests: stock mean within seed; dots = 3 seeds\n'
                 'Reused historical data; no validation-feasible condition', fontsize=13)
    fig.savefig(output / 'frozen_comparison.png', dpi=180)
    fig.savefig(output / 'frozen_comparison.svg')
    plt.close(fig)

    stress.to_csv(output / 'stress_by_symbol.csv', index=False)
    stress_fields = ['stress_day_return', 'window_net_return', 'window_max_drawdown',
                     'mean_drawdown_excess', 'stress_day_trades', 'stress_day_mean_exposure']
    stress_seed = stress.groupby(['condition', 'seed', 'stress_date'], as_index=False)[stress_fields].mean()
    stress_seed.to_csv(output / 'stress_by_seed_date.csv', index=False)
    stress_summary = stress_seed.groupby('condition')[stress_fields].agg(['mean', 'min', 'max'])
    stress_summary.columns = ['_'.join(parts) for parts in stress_summary.columns]
    stress_summary.reset_index().to_csv(output / 'stress_summary.csv', index=False)
    dates = sorted(stress.stress_date.unique())
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for axis, field, scale, title in zip(axes, ['stress_day_return', 'stress_day_mean_exposure'],
                                        [100, 100], ['Return on sparse session (%)', 'Exposure on sparse session (%)']):
        for condition in LABELS:
            by_date = stress_seed.loc[stress_seed.condition == condition].groupby('stress_date')[field]
            stats = by_date.agg(['mean', 'min', 'max']).reindex(dates) * scale
            x = np.arange(len(dates))
            axis.plot(x, stats['mean'], marker='o', color=COLORS[condition], label=LABELS[condition])
            axis.fill_between(x, stats['min'], stats['max'], color=COLORS[condition], alpha=.15)
        axis.set_ylabel(title)
        axis.axhline(0, color='gray', lw=.7)
        axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=9)
    axes[-1].set_xticks(np.arange(len(dates)), dates, rotation=35, ha='right')
    fig.suptitle('Exploratory sparse-session audit: frozen policies, observed bars only\n'
                 'Shading = seed range; overlapping windows; halt classification incomplete', fontsize=12)
    fig.savefig(output / 'stress_comparison.png', dpi=180)
    fig.savefig(output / 'stress_comparison.svg')
    plt.close(fig)

    checksums = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sources}
    result = {'completed': True, 'report_generated': False, 'training_performed': False,
              'selection_changed': False, 'frozen_test_rows': len(metrics), 'stress_rows': len(stress),
              'selected_condition': freeze['selected_condition'],
              'validation_feasible_conditions': freeze['validation_feasible_conditions'],
              'source_sha256': checksums,
              'limitations': ['Retrospective data reuse; earlier tests may overlap later validation',
                              'No condition met the validation risk budget for every symbol/seed/fold',
                              'Three seeds measure optimizer variation, not market uncertainty',
                              'Sparse days excluded from qualified study; stress audit is exploratory',
                              'Stress windows overlap and some precede the frozen model training endpoint',
                              'Cost addback is an accounting diagnostic, not a zero-cost policy counterfactual',
                              'Prospective holdout awaits future observations']}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/completion-v3'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.root, args.output)
    print(json.dumps({key: result[key] for key in ('completed', 'frozen_test_rows', 'stress_rows')}))
