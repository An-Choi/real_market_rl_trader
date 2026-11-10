"""Export machine-readable experiment tables and figures; no written report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["net_return", "bar_max_drawdown", "mean_exposure", "flat_fraction",
           "turnover", "trade_count", "friction_cost", "pre_cost_return_addback",
           "mean_drawdown_excess", "drawdown_exceedance_fraction"]
METRICS += ["discounted_drawdown_excess", "liquidation_max_drawdown"]


def export(directory):
    directory = Path(directory)
    results = json.loads((directory / "results.json").read_text())
    rows = []
    for run in results:
        for split, symbols in run["splits"].items():
            for symbol, metrics in symbols.items():
                rows.append({"fold": run["fold"], "condition": run["condition"],
                    "seed": run["seed"], "split": split, "symbol": symbol,
                    "training_decisions": run["actual_timesteps"],
                    "training_bars": run.get("actual_training_bars", run["actual_timesteps"]),
                    "selected_step": run["best_validation_step"],
                    "risk_multiplier": run.get("risk_multiplier"),
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
    table = pd.DataFrame(rows)
    table.to_csv(directory / "metrics.csv", index=False)
    metrics = [m for m in METRICS if m in table]
    seeds = table.groupby(["split", "fold", "condition", "seed"])[metrics].mean().reset_index()
    seeds.to_csv(directory / "per_seed_metrics.csv", index=False)
    # Keep fold effects separate; overlapping expanding folds are not IID.
    seeds.groupby(["split", "fold", "condition"])[metrics].agg(["mean", "std", "min", "max"]).to_csv(
        directory / "condition_by_fold.csv")
    effects = []
    for (split, fold, seed), group in seeds.groupby(["split", "fold", "seed"]):
        indexed = group.set_index("condition")
        for before, after in (("Q", "R"), ("R", "E"), ("E", "F"), ("F", "G"), ("F", "H"), ("G", "H"),
                              ("I", "J"), ("I", "K"), ("J", "K")):
            if before in indexed.index and after in indexed.index:
                effects.append({"split": split, "fold": fold, "seed": seed,
                    "comparison": f"{after}-{before}",
                    **{m: indexed.loc[after, m] - indexed.loc[before, m] for m in metrics}})
    pd.DataFrame(effects).to_csv(directory / "paired_effects.csv", index=False)
    baselines = []
    for path in sorted(directory.glob("fold_*_baselines.json")):
        for split, symbols in json.loads(path.read_text()).items():
            for symbol, controls in symbols.items():
                for name, values in controls.items():
                    baselines.append({"fold": int(path.name.split("_")[1]), "split": split,
                        "symbol": symbol, "baseline": name,
                        **{k: v for k, v in values.items() if isinstance(v, (int, float))}})
    if baselines:
        pd.DataFrame(baselines).to_csv(directory / "baseline_metrics.csv", index=False)
    for (split, fold), part in seeds.groupby(["split", "fold"]):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        for axis, metric, label, factor in zip(axes,
            ["net_return", "bar_max_drawdown", "turnover", "mean_drawdown_excess"],
            ["Net return (%)", "Max drawdown (%)", "Turnover", "Mean DD excess (%)"],
            [100, 100, 1, 100]):
            if metric not in part:
                continue
            for index, (condition, group) in enumerate(part.groupby("condition")):
                values = group[metric].to_numpy() * factor
                axis.scatter(index + np.linspace(-.08, .08, len(values)), values)
                axis.plot([index - .2, index + .2], [values.mean()]*2, color="black")
            axis.set_xticks(range(part.condition.nunique()), sorted(part.condition.unique()))
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
        fig.suptitle(f"{split} | fold {fold} | seed means across symbols")
        fig.tight_layout()
        fig.savefig(directory / f"{split}_fold_{fold:02d}.png", dpi=160)
        plt.close(fig)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    export(parser.parse_args().directory)
