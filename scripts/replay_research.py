"""Replay a frozen local research policy on its validation or final test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "agent/src", ROOT / "env/src"):
    sys.path.insert(0, str(source))

from experiments.run_ablation import load_market, load_research_bundle, evaluate, write_json, data_fingerprints
from experiments.finalize_research import fold_from_dict
from models.walk_forward import slice_walk_forward_fold


def replay(args):
    study = args.study.resolve()
    manifest = json.loads((study / "study_manifest.json").read_text())
    freeze = json.loads((study / "frozen_evaluation/selection_freeze.json").read_text())
    condition = args.condition or freeze["selected_condition"]
    fold = fold_from_dict(manifest["folds"][-1])
    bundle = study / f"fold_{fold.index:02d}" / condition / f"seed_{args.seed}"
    proof = next((item for item in freeze["artifacts"] if item["directory"] == str(bundle.relative_to(study))), None)
    if proof is None or hashlib.sha256((bundle / "research_model.zip").read_bytes()).hexdigest() != proof["model_sha256"]:
        raise ValueError("policy not in frozen evaluation")
    if hashlib.sha256((bundle / "research_manifest.json").read_bytes()).hexdigest() != proof["manifest_sha256"]:
        raise ValueError("research manifest changed after freeze")
    if args.output.exists():
        raise ValueError("output exists; choose a new replay directory")
    frames, _, _ = load_market(manifest["arguments"]["symbols"].split(","))
    if data_fingerprints(frames) != manifest["data_fingerprints"]:
        raise ValueError("research data changed since training")
    frame = slice_walk_forward_fold(frames[args.symbol], fold, segment=args.split)
    model, env, _ = load_research_bundle(bundle, frame)
    metrics, trace = evaluate(model, env)
    args.output.mkdir(parents=True)
    trace.to_csv(args.output / "trace.csv", index=False)
    write_json(args.output / "metrics.json", {"condition": condition, "seed": args.seed,
        "symbol": args.symbol, "split": args.split, "fold": fold.to_dict(),
        "metrics": metrics, "research_only": True})
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbol", default="005930")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    replay(parser.parse_args())
