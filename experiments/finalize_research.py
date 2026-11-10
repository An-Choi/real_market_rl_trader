"""Freeze validation selection, then evaluate saved policies with no training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "agent/src", ROOT / "env/src"):
    sys.path.insert(0, str(source))

import numpy as np
import pandas as pd
import torch

from experiments.run_ablation import (write_json, code_snapshot, data_fingerprints,
    load_market, load_research_bundle, evaluate, make_env, synthetic_data)
from experiments.export_research_metrics import export
from models.walk_forward import ExpandingWalkForwardFold, slice_walk_forward_fold
from models.normalization import FeatureNormalizer


def fold_from_dict(payload):
    from datetime import date
    from dataclasses import fields
    names = {field.name for field in fields(ExpandingWalkForwardFold)}
    return ExpandingWalkForwardFold(**{k: date.fromisoformat(v) if k.endswith(("start", "end")) else v
                                       for k, v in payload.items() if k in names})


def run(study):
    study = Path(study).resolve()
    output = study / "frozen_evaluation"
    if output.exists():
        raise ValueError("final evaluation already exists; do not overwrite or retune against it")
    manifest = json.loads((study / "study_manifest.json").read_text())
    completion = json.loads((study / "completion.json").read_text())
    if not completion["completed"] or completion["test_evaluated"]:
        raise ValueError("require complete validation-only study")
    results = json.loads((study / "results.json").read_text())
    data, _, _ = (synthetic_data() if manifest["arguments"].get("synthetic", False) else
                  load_market(manifest["arguments"]["symbols"].split(",")))
    if data_fingerprints(data) != manifest["data_fingerprints"]:
        raise ValueError("research data changed since training")
    if code_snapshot()["python_source_sha256"] != manifest["code"]["python_source_sha256"]:
        raise ValueError("research code changed since training; review and rerun study before final evaluation")
    torch.set_num_threads(2)
    scores = {}
    risk_values = {}
    artifacts = []
    for run in results:
        values = list(run["splits"]["validation"].values())
        returns = [v["net_return"] for v in values]
        score = float(np.median(returns) + .5 * min(min(returns), 0) +
                      .5 * np.mean([v["bar_max_drawdown"] for v in values]))
        scores.setdefault(run["condition"], []).append(score)
        risk_values.setdefault(run["condition"], []).extend(
            value["discounted_drawdown_excess"] for value in values)
        directory = study / f"fold_{run['fold']:02d}" / run["condition"] / f"seed_{run['seed']}"
        artifacts.append({"directory": str(directory.relative_to(study)),
            "model_sha256": hashlib.sha256((directory / "research_model.zip").read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256((directory / "research_manifest.json").read_bytes()).hexdigest()})
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    budget = manifest["config"]["research"]["risk_budget"]
    feasible = [k for k, values in risk_values.items() if max(values) <= budget]
    selected = sorted(feasible or list(means), key=lambda k: (-means[k], k))[0]
    output.mkdir()
    freeze = {"selected_condition": selected, "validation_mean_scores": means,
        "validation_max_risk_cost": {k: max(v) for k, v in risk_values.items()},
        "risk_budget": budget, "validation_feasible_conditions": feasible,
        "selection_status": "validation-risk-feasible" if feasible else "research-comparator-only; no feasible condition",
        "selection": "prefer conditions meeting risk budget for every validation symbol/seed/fold; then mean robust score; tie by name",
        "evaluation": "all frozen ablations and seeds, no training or checkpoint reselection",
        "primary_test_fold": max(r["fold"] for r in results),
        "holdout_status": ("synthetic diagnostic only" if manifest["arguments"].get("synthetic", False)
            else "retrospective reused data, not a pristine prospective holdout"),
        "earlier_fold_tests": "may overlap later validation; diagnostic only",
        "code": manifest["code"], "data_fingerprints": manifest["data_fingerprints"], "artifacts": artifacts}
    # This file is written before reading any test return or selecting examples.
    write_json(output / "selection_freeze.json", freeze)
    evaluated = []
    for run in results:
        result = copy.deepcopy(run)
        result["splits"] = {"test": {}}
        fold = fold_from_dict(manifest["folds"][run["fold"] - 1])
        directory = study / f"fold_{run['fold']:02d}" / run["condition"] / f"seed_{run['seed']}"
        trace_directory = output / f"fold_{run['fold']:02d}" / run["condition"] / f"seed_{run['seed']}"
        trace_directory.mkdir(parents=True)
        for symbol, frame in data.items():
            frame = slice_walk_forward_fold(frame, fold, segment="test")
            model, env, _ = load_research_bundle(directory, frame)
            metrics, trace = evaluate(model, env)
            result["splits"]["test"][symbol] = metrics
            trace.to_csv(trace_directory / f"test_{symbol}_trace.csv", index=False)
        evaluated.append(result)
        write_json(output / "results.json", evaluated)
        print(json.dumps({"final_test": run["condition"], "fold": run["fold"], "seed": run["seed"],
            "mean_return": float(np.mean([m["net_return"] for m in result["splits"]["test"].values()]))}), flush=True)
    # Same deterministic controls and friction configuration as validation.
    for fold_payload in manifest["folds"]:
        fold = fold_from_dict(fold_payload)
        example = next(r for r in results if r["fold"] == fold.index)
        directory = study / f"fold_{fold.index:02d}" / example["condition"] / f"seed_{example['seed']}"
        normalizer = FeatureNormalizer.load(directory / "normalization.json")
        config = copy.deepcopy(manifest["config"])
        config["environment"]["unit_fraction"] = .995
        controls = {symbol: {name: evaluate(None,
            make_env(slice_walk_forward_fold(frame, fold, segment="test"), config, "A", normalizer), baseline=name)[0]
            for name in ("cash", "buy_and_hold", "mean_reversion")} for symbol, frame in data.items()}
        intraday_config = copy.deepcopy(config)
        intraday_config["environment"]["episode_days"] = 1
        for symbol, frame in data.items():
            for name in ("intraday_buy_and_hold", "intraday_mean_reversion"):
                controls[symbol][name] = evaluate(None, make_env(
                    slice_walk_forward_fold(frame, fold, segment="test"), intraday_config, "R", normalizer), baseline=name)[0]
        write_json(output / f"fold_{fold.index:02d}_baselines.json", {"test": controls})
    write_json(output / "study_manifest.json", manifest)
    export(output)
    write_json(output / "completion.json", {"completed": True, "runs": len(evaluated),
        "selected_condition_before_test": selected, "training_performed": False,
        "primary_test_fold": freeze["primary_test_fold"], "research_only": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    run(parser.parse_args().study)
