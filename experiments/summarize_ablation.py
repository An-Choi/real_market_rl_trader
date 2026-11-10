"""Export validation/test tables, plots and an honest research summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "study_manifest.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    rows = []
    for run in results:
        for split, symbols in run["splits"].items():
            for symbol, metrics in symbols.items():
                rows.append({"fold": run["fold"], "condition": run["condition"],
                             "seed": run["seed"], "split": split, "symbol": symbol,
                             "timesteps": run["actual_timesteps"],
                             "parameters": run["parameter_count"],
                             "best_step": run["best_validation_step"],
                             **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
    table = pd.DataFrame(rows)
    table.to_csv(directory / "metrics.csv", index=False)
    metric_columns = ["net_return", "bar_max_drawdown", "flat_fraction", "mean_exposure", "turnover", "trade_count"]
    seeds = table.groupby(["split", "fold", "condition", "seed"])[metric_columns].mean().reset_index()
    seeds.to_csv(directory / "per_seed_metrics.csv", index=False)
    aggregates = seeds.groupby(["split", "condition"])[metric_columns].agg(["mean", "std", "min", "max"])
    aggregates.to_csv(directory / "condition_summary.csv")
    effects = []
    for (split, fold, seed), group in seeds.groupby(["split", "fold", "seed"]):
        indexed = group.set_index("condition")
        for before, after in (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")):
            if before in indexed.index and after in indexed.index:
                effects.append({"split": split, "fold": fold, "seed": seed,
                                "comparison": f"{after}-{before}",
                                **{m: indexed.loc[after, m] - indexed.loc[before, m]
                                   for m in metric_columns}})
    pd.DataFrame(effects).to_csv(directory / "paired_effects.csv", index=False)
    for split, part in seeds.groupby("split"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for axis, metric, label in zip(axes, ["net_return", "bar_max_drawdown", "mean_exposure"],
                                      ["Net return (%)", "Bar max drawdown (%)", "Mean exposure (%)"]):
            for i, (condition, group) in enumerate(part.groupby("condition")):
                values = group[metric].to_numpy() * 100
                axis.scatter(np.full(len(values), i) + np.linspace(-.08, .08, len(values)), values, s=24)
                axis.plot([i - .2, i + .2], [values.mean()] * 2, color="black")
            axis.set_xticks(range(part.condition.nunique()), sorted(part.condition.unique()))
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
        fig.suptitle(f"{split.title()} | seed points, black mean | research only")
        fig.tight_layout()
        fig.savefig(directory / f"{split}_comparison.png", dpi=160)
        plt.close(fig)
    # Same fold, seed and symbol across conditions for interpretable trajectories.
    first = results[0]
    symbol = next(iter(first["splits"]["validation"]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for condition in sorted(table.condition.unique()):
        path = directory / f"fold_{first['fold']:02d}" / condition / f"seed_{first['seed']}" / f"validation_{symbol}_trace.csv"
        if not path.exists():
            continue
        trace = pd.read_csv(path)
        initial = manifest["config"]["environment"]["initial_cash"]
        axes[0].plot(trace.portfolio_value.to_numpy() / initial - 1, label=condition)
        axes[1].plot(trace.exposure.to_numpy(), label=condition, alpha=.7)
    axes[0].set_ylabel("Marked return (unsettled)")
    axes[1].set_ylabel("Exposure")
    axes[1].set_xlabel("Validation bar")
    axes[0].legend(ncol=4)
    fig.suptitle(f"Example only: {symbol}, seed {first['seed']}, fold {first['fold']}")
    fig.tight_layout()
    fig.savefig(directory / "validation_example.png", dpi=160)
    plt.close(fig)
    # Learning curves are validation checkpoint evaluations, not test tuning.
    fig, axis = plt.subplots(figsize=(8, 4))
    for run in results:
        path = directory / f"fold_{run['fold']:02d}" / run["condition"] / f"seed_{run['seed']}" / "validation.json"
        history = json.loads(path.read_text())
        axis.plot([h["timesteps"] for h in history], [h["score"] for h in history],
                  label=f"{run['condition']}/s{run['seed']}/f{run['fold']}", alpha=.8)
    axis.set_xlabel("Training environment transitions (all symbols)")
    axis.set_ylabel("Validation selection score")
    axis.legend(fontsize=7, ncol=3)
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / "validation_learning.png", dpi=160)
    plt.close(fig)
    text = ["# Offline RL ablation results", "", f"Study: `{manifest['study_type']}`.",
            f"Holdout: `{manifest['holdout_status']}`.",
            "", "Validation selected the checkpoint; its reported score is not an unbiased out-of-sample estimate.",
            "Financial returns below do not establish profitability. Synthetic runs are pipeline diagnostics only.",
            "", f"Completed training runs: {len(results)}. Requested transitions per run: {manifest['arguments']['timesteps']}.",
            "Actual transitions may exceed the request by one PPO rollout; see metrics.csv.",
            "", "Each seed point averages symbols within a fold; symbols are not independent training seeds.",
            "", "| Split | Condition | Seed/fold points | Mean net return | Std net return | Mean exposure |", "|---|---|---:|---:|---:|---:|"]
    for (split, condition), group in seeds.groupby(["split", "condition"]):
        std = group.net_return.std()
        text.append(f"| {split} | {condition} | {len(group)} | {group.net_return.mean():.3%} | {std:.3%} | {group.mean_exposure.mean():.3%} |")
    baseline_rows = []
    for path in sorted(directory.glob("fold_*_baselines.json")):
        payload = json.loads(path.read_text())
        for split, symbols in payload.items():
            for symbol, baselines in symbols.items():
                for name, metrics in baselines.items():
                    baseline_rows.append({"file": path.name, "split": split,
                                          "symbol": symbol, "baseline": name,
                                          **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
    if baseline_rows:
        baseline_table = pd.DataFrame(baseline_rows)
        baseline_table.to_csv(directory / "baseline_metrics.csv", index=False)
        text += ["", "## Fixed baseline controls", "", "Deterministic controls, averaged across symbols/folds; no duplicated seed points.",
                 "", "| Split | Baseline | Mean net return | Mean exposure |", "|---|---|---:|---:|"]
        for (split, baseline), group in baseline_table.groupby(["split", "baseline"]):
            text.append(f"| {split} | {baseline} | {group.net_return.mean():.3%} | {group.mean_exposure.mean():.3%} |")
    if "test" not in table.split.unique():
        text += ["", "Test returns were not evaluated. Freeze settings before an explicit final evaluation."]
    text += ["", "Artifacts are research-only and cannot be loaded by the production v5 serving path.",
             "The A/B/C/D environments share revised observed-price order sizing, exposure-based reward penalties,",
             "the same terminal mark settlement, and common account inputs. A is not the unchanged historical A0 policy.",
             "C/D add intraday history with validity bits; actor/critic parameter counts are approximately matched.",
             "Missing next-execution prints cause no fill, not a fabricated transaction at an earlier price.",
             "", "See study_manifest.json, per_seed_metrics.csv, per-run validation.json/progress.csv and trace files."]
    (directory / "REPORT.md").write_text("\n".join(text) + "\n")
    print(directory / "REPORT.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    summarize(p.parse_args().directory)
