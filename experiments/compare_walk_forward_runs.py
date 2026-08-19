"""Aggregate multiple walk-forward runs into one reproducible comparison JSON."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_run(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    folds = [
        json.loads(file.read_text(encoding="utf-8"))
        for file in sorted(path.glob("fold_*.json"))
    ]
    fold_rows = []
    baseline_returns: dict[str, list[float]] = {}
    for payload in folds:
        symbol_payloads = list(payload["per_symbol"].values())
        hold_rates = [
            float(item["model"]["metrics"]["hold_action_rate"])
            for item in symbol_payloads
        ]
        turnovers = [
            float(item["model"]["metrics"]["turnover"])
            for item in symbol_payloads
        ]
        trade_counts = [
            float(item["model"]["metrics"]["trade_count"])
            for item in symbol_payloads
        ]
        fold_baselines: dict[str, list[float]] = {}
        for item in symbol_payloads:
            for baseline in item["baselines"]:
                name = baseline["agent"]
                value = float(baseline["metrics"]["total_return"])
                fold_baselines.setdefault(name, []).append(value)
                baseline_returns.setdefault(name, []).append(value)
        fold_rows.append({
            "fold": int(payload["fold"]["index"]),
            "training_seed": payload.get("training_seed"),
            "deployment_status": payload.get("deployment_status", "legacy"),
            "test_regime": payload["fold"]["test_regime"],
            "mean_total_return": float(payload["aggregate"]["mean_total_return"]),
            "mean_max_drawdown": float(payload["aggregate"]["mean_max_drawdown"]),
            "mean_hold_action_rate": _mean(hold_rates),
            "mean_turnover": _mean(turnovers),
            "mean_trade_count": _mean(trade_counts),
            "validation_score": (
                float(payload["validation_best"]["selection_score"])
                if payload.get("validation_best", {}).get("selection_score") is not None
                else None
            ),
            "validation_qualified": bool(payload["validation_qualified"]),
            "baseline_mean_returns": {
                name: _mean(values) for name, values in fold_baselines.items()
            },
        })
    returns = [row["mean_total_return"] for row in fold_rows]
    return {
        "run": str(path.relative_to(PROJECT_ROOT)),
        "feature_columns": summary["feature_columns"],
        "excluded_features": summary["excluded_features"],
        "folds": fold_rows,
        "overall": {
            "mean_total_return": _mean(returns),
            "std_fold_return": float(statistics.pstdev(returns)),
            "mean_max_drawdown": _mean(
                [row["mean_max_drawdown"] for row in fold_rows]
            ),
            "mean_hold_action_rate": _mean(
                [row["mean_hold_action_rate"] for row in fold_rows]
            ),
            "mean_turnover": _mean([row["mean_turnover"] for row in fold_rows]),
            "mean_trade_count": _mean(
                [row["mean_trade_count"] for row in fold_rows]
            ),
            "qualified_folds": sum(row["validation_qualified"] for row in fold_rows),
            "training_seeds": sorted({
                row["training_seed"] for row in fold_rows
                if row["training_seed"] is not None
            }),
            "baseline_mean_returns": {
                name: _mean(values) for name, values in baseline_returns.items()
            },
        },
    }


def main() -> None:
    args = parse_args()
    runs = [path if path.is_absolute() else PROJECT_ROOT / path for path in args.runs]
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screening_only": True,
        "action_schema": ["hold", "add_unit", "reduce_unit", "clear"],
        "runs": [summarize_run(path) for path in runs],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
