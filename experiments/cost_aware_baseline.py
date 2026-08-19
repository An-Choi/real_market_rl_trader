"""Run a leakage-safe, cost-aware mean-reversion walk-forward baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "env" / "src", PROJECT_ROOT / "agent" / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.common import load_feature_data, make_data_loader, resolve_symbols
from friction.friction_model import FrictionModel
from models.walk_forward import (
    align_data_on_common_days,
    build_expanding_walk_forward_folds,
    slice_walk_forward_fold,
)
from policies.cost_aware import (
    ConfidenceGate,
    DEFAULT_ENTRY_QUANTILES,
    DEFAULT_HOLD_BARS,
    candidate_grid,
    run_leakage_safe_fold,
)
from utils.config_loader import load_config


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--folds", type=_positive_int, default=None)
    parser.add_argument("--validation-days", type=_positive_int, default=None)
    parser.add_argument("--test-days", type=_positive_int, default=None)
    parser.add_argument("--purge-days", type=int, default=None)
    parser.add_argument("--order-notional", type=float, default=None)
    parser.add_argument("--minimum-events", type=_positive_int, default=8)
    parser.add_argument("--minimum-days", type=_positive_int, default=5)
    parser.add_argument("--minimum-blocks", type=_positive_int, default=3)
    parser.add_argument("--block-days", type=_positive_int, default=5)
    parser.add_argument("--bootstrap-samples", type=_positive_int, default=2_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _json_safe(payload: Any) -> Any:
    """Fail closed instead of emitting non-standard NaN/Infinity JSON."""
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, tuple):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.floating,)):
        payload = float(payload)
    if isinstance(payload, float) and not np.isfinite(payload):
        return None
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    symbols = resolve_symbols(config=config, cli_symbols=args.symbols)
    loader = make_data_loader(project_root=PROJECT_ROOT, config=config)
    raw = {
        symbol: load_feature_data(
            symbol=symbol,
            data_loader=loader,
            force_rebuild=args.force_rebuild,
        )
        for symbol in symbols
    }
    aligned = align_data_on_common_days(raw)
    walk_forward = config.get("walk_forward", {})
    purge_days = (
        args.purge_days
        if args.purge_days is not None
        else int(config["data"].get("split", {}).get("purge_days", 0))
    )
    if purge_days < 0:
        raise SystemExit("--purge-days must be >= 0")
    folds = build_expanding_walk_forward_folds(
        aligned,
        n_folds=args.folds or int(walk_forward.get("folds", 3)),
        validation_days=(
            args.validation_days or int(walk_forward.get("validation_days", 60))
        ),
        test_days=args.test_days or int(walk_forward.get("test_days", 20)),
        purge_days=purge_days,
    )
    order_notional = args.order_notional
    if order_notional is None:
        order_notional = float(config["environment"]["initial_cash"]) * float(
            config["environment"]["unit_fraction"]
        )
    gate = ConfidenceGate(
        minimum_events=args.minimum_events,
        minimum_days=args.minimum_days,
        minimum_blocks=args.minimum_blocks,
        block_days=args.block_days,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    friction_model = FrictionModel(**config["friction"])
    specs = candidate_grid(DEFAULT_ENTRY_QUANTILES, DEFAULT_HOLD_BARS)

    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        partitions = {
            segment: {
                symbol: slice_walk_forward_fold(frame, fold, segment=segment)
                for symbol, frame in aligned.items()
            }
            for segment in ("train", "validation", "test")
        }
        result = run_leakage_safe_fold(
            train_data=partitions["train"],
            validation_data=partitions["validation"],
            test_data=partitions["test"],
            friction_model=friction_model,
            order_notional=order_notional,
            gate=gate,
            specs=specs,
        )
        fold_results.append({"fold": fold.to_dict(), **result})

    validation_passes = [
        bool(result["selection"]["gate_passed"]) for result in fold_results
    ]
    test_returns = [
        float(result["test"]["metrics"]["net_total_return"])
        for result in fold_results
    ]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "common_trading_days": int(
            pd.to_datetime(next(iter(aligned.values()))["Timestamp"]).dt.date.nunique()
        ),
        "candidate_grid": [spec.to_dict() for spec in specs],
        "order_notional": float(order_notional),
        "friction": config["friction"],
        "gate": gate.to_dict(),
        "methodology": {
            "signal": "negative robust z-score blend: 0.50 log_ret_12 + 0.35 vwap_dev + 0.15 log_ret_1",
            "fit": "per-symbol centers, scales, tail thresholds, and expected rebound use train only",
            "selection": "candidate is chosen by validation block-bootstrap LCB; test is never selection input",
            "execution": "same-day, non-overlapping events; integer shares; entry and exit friction use row price/date/Adv20",
        },
        "folds": fold_results,
        "summary": {
            "validation_gate_passed_folds": int(sum(validation_passes)),
            "validation_gate_total_folds": len(validation_passes),
            "all_validation_gates_passed": bool(all(validation_passes)),
            "mean_test_net_return": float(np.mean(test_returns)) if test_returns else 0.0,
            "worst_test_net_return": float(np.min(test_returns)) if test_returns else 0.0,
            "test_is_audit_only": True,
        },
    }
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output = PROJECT_ROOT / "runs" / "cost_aware_baseline" / f"baseline-{stamp}.json"
    elif not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    output.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **safe_payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
