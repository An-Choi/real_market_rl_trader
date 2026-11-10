"""Execute capability checks, cost diagnostics, ablations and frozen evaluation.

Produces models, JSON/CSV logs and figures. It does not generate a report,
contact a broker, or place orders. Each stage preserves its earlier artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "agent/src", ROOT / "env/src"):
    sys.path.insert(0, str(source))

import pandas as pd
import torch

from experiments.run_ablation import (write_json, code_snapshot, load_market,
    load_research_bundle, make_env, evaluate, data_fingerprints)
from experiments.export_research_metrics import export
from experiments.finalize_research import fold_from_dict
from models.walk_forward import slice_walk_forward_fold

COSTS = (0., .5, 1., 2.)


def execute(arguments):
    command = [sys.executable, *map(str, arguments)]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def snapshot(directory, manifest_name="study_manifest.json"):
    directory = Path(directory)
    if (directory / "source_snapshot.tar.gz").exists():
        return
    manifest = json.loads((directory / manifest_name).read_text())
    if manifest["code"]["python_source_sha256"] != code_snapshot()["python_source_sha256"]:
        raise ValueError("code changed while experiment was running")
    with tarfile.open(directory / "source_snapshot.tar.gz", "w:gz") as archive:
        for area in ("agent/src", "env/src", "evaluation", "experiments"):
            for path in sorted((ROOT / area).rglob("*.py")):
                archive.add(path, arcname=str(path.relative_to(ROOT)))
        archive.add(ROOT / "env/configs/config.yaml", arcname="env/configs/config.yaml")


def completed(directory):
    path = Path(directory) / "completion.json"
    return path.exists() and json.loads(path.read_text()).get("completed", False)


def cost_cross_evaluation(root):
    output = root / "cost_cross_evaluation"
    if completed(output):
        return
    if output.exists():
        raise ValueError("incomplete cost evaluation exists; preserve it and use a new output")
    data, _, _ = load_market("005930,000660,034220,066570,009150".split(","))
    output.mkdir()
    rows = []
    for training_cost in COSTS:
        study = root / f"cost_{training_cost:g}"
        manifest = json.loads((study / "study_manifest.json").read_text())
        if data_fingerprints(data) != manifest["data_fingerprints"]:
            raise ValueError("data changed during cost diagnostics")
        fold = fold_from_dict(manifest["folds"][0])
        for run in json.loads((study / "results.json").read_text()):
            directory = study / f"fold_{run['fold']:02d}" / run["condition"] / f"seed_{run['seed']}"
            for symbol, source in data.items():
                frame = slice_walk_forward_fold(source, fold, segment="validation")
                model, original, bundle = load_research_bundle(directory, frame)
                for evaluation_cost in COSTS:
                    config = copy.deepcopy(bundle["config"])
                    config.setdefault("research", {})["friction_multiplier"] = evaluation_cost
                    metrics, _ = evaluate(model, make_env(frame, config, run["condition"], original.normalizer))
                    rows.append({"train_cost_multiplier": training_cost,
                        "evaluation_cost_multiplier": evaluation_cost, "seed": run["seed"],
                        "symbol": symbol,
                        **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                        "mean_action_probability": json.dumps(metrics["mean_action_probability"])})
            print(json.dumps({"cost_cross_evaluation": training_cost, "seed": run["seed"]}), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    numeric = [c for c in ("net_return", "pre_cost_return_addback", "friction_cost", "turnover",
                           "mean_exposure", "mean_policy_entropy", "bar_max_drawdown") if c in table]
    table.groupby(["train_cost_multiplier", "evaluation_cost_multiplier", "seed"])[numeric].mean().to_csv(
        output / "per_seed_metrics.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cost, group in table.groupby("train_cost_multiplier"):
        means = group.groupby("evaluation_cost_multiplier")[numeric].mean()
        axes[0].plot(means.index, means.net_return * 100, marker="o", label=f"train cost {cost:g}")
        axes[1].plot(means.index, means.turnover, marker="o", label=f"train cost {cost:g}")
    for axis in axes:
        axis.set_xlabel("Evaluation cost multiplier")
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_ylabel("Validation net return (%)")
    axes[1].set_ylabel("Turnover")
    fig.tight_layout()
    fig.savefig(output / "cost_sensitivity.png", dpi=160)
    plt.close(fig)
    write_json(output / "completion.json", {"completed": True, "rows": len(rows),
        "split": "validation only", "training_performed": False,
        "cost_addback": "same realized path arithmetic addback, not a no-cost strategy counterfactual",
        "counterfactual": "evaluation at each cost reruns the fixed policy; its actions and quantities may change"})


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    stages = {args.stage} if args.stage != "all" else {"diagnostics", "costs", "study", "finalize"}
    if not (root / "data_audit.json").exists():
        execute(["scripts/research_data_audit.py", "--output", root / "data_audit.json"])
        audit = json.loads((root / "data_audit.json").read_text())
        repair = [{"symbol": symbol["symbol"], "date": row["date"],
                   "reasons": row["reasons"], "status": "excluded or deterministically sanitized; raw preserved"}
                  for symbol in audit["symbols"] for row in symbol["rows"] if not row["raw_valid"]]
        write_json(root / "data_repair_plan.json", {"raw_modified": False,
            "recollection": "requires locally configured KIS credentials; not needed for offline research",
            "no_fabricated_bars": True, "items": repair})
        paths = sorted((ROOT / "data/raw").rglob("*.parquet"))
        write_json(root / "raw_data_inventory.json", [{"path": str(path.relative_to(ROOT / "data")),
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths])
    if "diagnostics" in stages:
        directory = root / "learning_diagnostics"
        if not completed(directory):
            execute(["experiments/run_learning_diagnostics.py", "--output", directory,
                     "--seeds", args.seeds, "--bar-budget", args.diagnostic_bars,
                     "--gamma", ".99", "--gae-lambda", ".95", "--ent-coef", ".5", "--episode-days", "1"])
        snapshot(directory, "manifest.json")
        if not json.loads((directory / "completion.json").read_text())["all_capability_checks_passed"]:
            raise RuntimeError("easy market capability check failed; inspect and fix before market tuning")
    if stages.intersection({"costs", "study"}):
        diagnostics = root / "learning_diagnostics"
        if not completed(diagnostics) or not json.loads((diagnostics / "completion.json").read_text())["all_capability_checks_passed"]:
            raise RuntimeError("complete and pass easy-market diagnostics before market studies")
        diagnostic_manifest = json.loads((diagnostics / "manifest.json").read_text())
        if set(diagnostic_manifest["regimes"]) != {"rising", "falling", "flat"}:
            raise RuntimeError("all three capability regimes must be checked")
    if "costs" in stages:
        for cost in COSTS:
            directory = root / f"cost_{cost:g}"
            if not completed(directory):
                execute(["experiments/run_ablation.py", "--output", directory, "--conditions", "R",
                    "--seeds", args.seeds, "--bar-budget", args.cost_bars, "--timesteps", args.cost_bars,
                    "--eval-freq", args.cost_bars, "--friction-multiplier", cost,
                    "--gamma", ".99", "--gae-lambda", ".95", "--ent-coef", ".5", "--episode-days", "1"])
            snapshot(directory)
            export(directory)
        cost_cross_evaluation(root)
    if "study" in stages:
        directory = root / "ablation"
        if not completed(directory):
            execute(["experiments/run_ablation.py", "--output", directory, "--conditions", "Q,R,E,F,G,H",
                "--seeds", args.seeds, "--folds", "2", "--validation-days", "20", "--test-days", "20",
                "--bar-budget", args.bar_budget, "--timesteps", args.bar_budget, "--eval-freq", "25000",
                "--gamma", ".99", "--gae-lambda", ".95", "--ent-coef", ".5", "--episode-days", "1"])
        snapshot(directory)
        export(directory)
    if "finalize" in stages:
        directory = root / "ablation"
        if not completed(directory / "frozen_evaluation"):
            execute(["experiments/finalize_research.py", directory])
    write_json(root / f"stage_{args.stage}_completion.json", {"completed": True,
        "stage": args.stage, "code": code_snapshot(), "arguments": vars(args),
        "report_generated": False, "research_only": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bar-budget", type=int, default=75000)
    parser.add_argument("--cost-bars", type=int, default=25000)
    parser.add_argument("--diagnostic-bars", type=int, default=25000)
    parser.add_argument("--stage", choices=["all", "diagnostics", "costs", "study", "finalize"], default="all")
    args = parser.parse_args()
    if min(args.bar_budget, args.cost_bars, args.diagnostic_bars) < 1:
        raise SystemExit("budgets must be positive")
    run(args)
