"""Train and evaluate independent expanding walk-forward folds."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
AGENT_SRC = PROJECT_ROOT / "agent" / "src"
for path in (ENV_SRC, AGENT_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.common import load_feature_data, make_data_loader, resolve_symbols
from data.feature_engineer import FeatureEngineer
from models.artifact import check_env_compatibility, load_artifact
from models.training import resolve_training_feature_columns, train_ppo_artifact
from models.walk_forward import (
    SplitBoundaries,
    build_expanding_walk_forward_folds,
    common_trading_days,
    slice_walk_forward_fold,
)
from policies.evaluation import (
    build_backtest_environment,
    compare_baselines,
    run_agent_backtest,
)
from utils.config_loader import load_config
from utils.logger import setup_logger


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe expanding walk-forward training and evaluation"
    )
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--folds", type=_positive_int, default=None)
    parser.add_argument("--validation-days", type=_positive_int, default=None)
    parser.add_argument("--validation-window-days", type=_positive_int, default=None)
    parser.add_argument("--min-validation-regimes", type=_positive_int, default=None)
    parser.add_argument("--regime-threshold", type=float, default=None)
    parser.add_argument("--test-days", type=_positive_int, default=None)
    parser.add_argument("--purge-days", type=int, default=None)
    parser.add_argument("--total-timesteps", type=_positive_int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated training seeds applied independently to every fold.",
    )
    parser.add_argument("--minimum-coverage", type=float, default=None)
    parser.add_argument(
        "--exclude-features",
        default="",
        help="Comma-separated schema-v4 market features to exclude for ablation.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--tensorboard", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def _parse_training_seeds(raw: str | None, fallback: int) -> list[int]:
    if raw is None:
        return [int(fallback)]
    try:
        seeds = [int(token.strip()) for token in raw.split(",") if token.strip()]
    except ValueError as exc:
        raise SystemExit("--seeds must contain comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("--seeds must contain unique comma-separated integers")
    return seeds


def _reference_calendar(
    loader,
    symbols: list[str],
    feature_data: dict[str, pd.DataFrame],
) -> list:
    """Daily-data exchange proxy, clipped to the jointly usable feature span."""
    calendar: set | None = None
    for symbol in symbols:
        daily = loader.load_raw_parquet_all(symbol, "1d")
        date_col = "Date" if "Date" in daily else "Timestamp"
        days = set(pd.to_datetime(daily[date_col]).dt.date)
        calendar = days if calendar is None else calendar.intersection(days)
    start = max(
        pd.to_datetime(frame["Timestamp"]).dt.date.min()
        for frame in feature_data.values()
    )
    end = min(
        pd.to_datetime(frame["Timestamp"]).dt.date.max()
        for frame in feature_data.values()
    )
    return sorted(day for day in (calendar or set()) if start <= day <= end)


def _max_missing_run(expected: list, observed: set) -> int:
    longest = current = 0
    for day in expected:
        if day in observed:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _fold_coverage(
    fold,
    data_by_symbol: dict[str, pd.DataFrame],
    reference_days: list,
) -> dict[str, Any]:
    bounds = {
        "train": (fold.train_start, fold.train_end),
        "validation": (fold.validation_start, fold.validation_end),
        "test": (fold.test_start, fold.test_end),
    }
    result: dict[str, Any] = {}
    for segment, (start, end) in bounds.items():
        expected = [day for day in reference_days if start <= day <= end]
        per_symbol = {}
        for symbol, frame in data_by_symbol.items():
            observed = set(pd.to_datetime(frame["Timestamp"]).dt.date)
            observed_in_window = observed.intersection(expected)
            per_symbol[symbol] = {
                "expected_days": len(expected),
                "observed_days": len(observed_in_window),
                "coverage": (
                    float(len(observed_in_window) / len(expected)) if expected else 0.0
                ),
                "missing_days": [day.isoformat() for day in expected if day not in observed],
                "max_consecutive_missing_days": _max_missing_run(expected, observed),
                "calendar_span_days": int((end - start).days + 1),
            }
        result[segment] = per_symbol
    return result


def classify_market_regime(mean_market_return: float, threshold: float = 0.05) -> str:
    """Human-readable ex-post audit label; never used to move fold boundaries."""
    if mean_market_return >= threshold:
        return "bull"
    if mean_market_return <= -threshold:
        return "bear"
    return "sideways"


def _market_return(frame: pd.DataFrame) -> float:
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 2 or close.iloc[0] == 0:
        return 0.0
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def describe_validation_windows(
    fold,
    data_by_symbol: dict[str, pd.DataFrame],
    *,
    window_days: int,
    regime_threshold: float,
) -> list[dict[str, Any]]:
    """Audit regime diversity inside validation using only validation data."""
    validation = {
        symbol: slice_walk_forward_fold(data, fold, segment="validation")
        for symbol, data in data_by_symbol.items()
    }
    shared_days = sorted({
        day
        for frame in validation.values()
        for day in pd.to_datetime(frame["Timestamp"]).dt.date.unique()
    })
    if not shared_days:
        raise ValueError(f"fold {fold.index} validation has no trading days")
    chunks = [
        shared_days[start:start + window_days]
        for start in range(0, len(shared_days), window_days)
    ]
    min_tail_days = max(1, window_days // 2)
    if len(chunks) > 1 and len(chunks[-1]) < min_tail_days:
        chunks[-2].extend(chunks[-1])
        chunks.pop()

    windows: list[dict[str, Any]] = []
    for days in chunks:
        start, end = days[0], days[-1]
        market_returns: dict[str, float] = {}
        for symbol, frame in validation.items():
            row_days = pd.to_datetime(frame["Timestamp"]).dt.date
            window_frame = frame.loc[(row_days >= start) & (row_days <= end)]
            if not window_frame.empty:
                market_returns[symbol] = _market_return(window_frame)
        mean_market_return = float(np.mean(list(market_returns.values())))
        windows.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days": len(days),
            "market_returns": market_returns,
            "mean_market_return": mean_market_return,
            "regime": classify_market_regime(mean_market_return, regime_threshold),
        })
    return windows


def describe_fold(
    fold,
    data_by_symbol: dict[str, pd.DataFrame],
    *,
    reference_days: list,
    minimum_coverage: float,
    validation_window_days: int,
    regime_threshold: float,
    min_validation_regimes: int,
) -> dict[str, Any]:
    validation_windows = describe_validation_windows(
        fold,
        data_by_symbol,
        window_days=validation_window_days,
        regime_threshold=regime_threshold,
    )
    validation_regimes = sorted({window["regime"] for window in validation_windows})
    market_returns = {
        symbol: _market_return(slice_walk_forward_fold(data, fold, segment="test"))
        for symbol, data in data_by_symbol.items()
    }
    mean_market_return = float(np.mean(list(market_returns.values())))
    coverage = _fold_coverage(fold, data_by_symbol, reference_days)
    coverage_values = [
        details["coverage"]
        for segment in coverage.values()
        for details in segment.values()
    ]
    return {
        **fold.to_dict(),
        "validation_windows": validation_windows,
        "validation_regimes": validation_regimes,
        "validation_regime_count": len(validation_regimes),
        "min_validation_regimes": min_validation_regimes,
        "validation_regime_check_passed": (
            len(validation_regimes) >= min_validation_regimes
        ),
        "test_market_returns": market_returns,
        "mean_test_market_return": mean_market_return,
        "test_regime": classify_market_regime(mean_market_return, regime_threshold),
        "coverage": coverage,
        "minimum_coverage": minimum_coverage,
        "minimum_observed_coverage": float(min(coverage_values)),
        "coverage_check_passed": bool(
            coverage_values and min(coverage_values) >= minimum_coverage
        ),
    }


def aggregate_model_metrics(per_symbol: dict[str, dict[str, Any]]) -> dict[str, float]:
    metrics = [payload["model"]["metrics"] for payload in per_symbol.values()]
    returns = np.asarray([item["total_return"] for item in metrics], dtype=np.float64)
    drawdowns = np.asarray([item["max_drawdown"] for item in metrics], dtype=np.float64)
    market = np.asarray([item["market_return"] for item in metrics], dtype=np.float64)
    return {
        "mean_total_return": float(np.mean(returns)),
        "median_total_return": float(np.median(returns)),
        "worst_total_return": float(np.min(returns)),
        "positive_symbol_rate": float(np.mean(returns > 0.0)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "mean_market_return": float(np.mean(market)),
        "mean_excess_market_return": float(np.mean(returns - market)),
    }


def _make_output_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else PROJECT_ROOT / explicit
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = PROJECT_ROOT / "runs" / "walk_forward" / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    symbols = resolve_symbols(config=config, cli_symbol=args.symbol, cli_symbols=args.symbols)
    excluded_features = [
        token.strip() for token in args.exclude_features.split(",") if token.strip()
    ]
    if len(excluded_features) != len(set(excluded_features)):
        raise SystemExit("--exclude-features contains duplicates")
    unknown_exclusions = sorted(
        set(excluded_features).difference(FeatureEngineer.FEATURE_COLUMNS)
    )
    if unknown_exclusions:
        raise SystemExit(f"unknown --exclude-features: {unknown_exclusions}")
    feature_columns = resolve_training_feature_columns([
        column
        for column in FeatureEngineer.FEATURE_COLUMNS
        if column not in excluded_features
    ])
    loader = make_data_loader(project_root=PROJECT_ROOT, config=config)
    raw_feature_data = {
        symbol: load_feature_data(
            symbol=symbol, data_loader=loader, force_rebuild=args.force_rebuild
        )
        for symbol in symbols
    }
    source_day_counts = {
        symbol: int(pd.to_datetime(frame["Timestamp"]).dt.date.nunique())
        for symbol, frame in raw_feature_data.items()
    }
    # Keep every high-quality day available for each symbol.  Fold boundaries
    # come from a separate daily reference calendar; missing feature days are
    # now visible in the coverage audit instead of silently shrinking time.
    all_data = raw_feature_data
    common_day_count = len(common_trading_days(raw_feature_data))
    reference_days = _reference_calendar(loader, symbols, raw_feature_data)
    if not reference_days:
        raise SystemExit("daily reference calendar does not overlap usable feature data")
    purge_days = (
        args.purge_days
        if args.purge_days is not None
        else int(config["data"].get("split", {}).get("purge_days", 0))
    )
    if purge_days < 0:
        raise SystemExit("--purge-days must be >= 0")
    wf_config = config.get("walk_forward", {})
    n_folds = args.folds or int(wf_config.get("folds", 3))
    validation_days = args.validation_days or int(wf_config.get("validation_days", 60))
    validation_window_days = args.validation_window_days or int(
        wf_config.get("validation_window_days", 20)
    )
    min_validation_regimes = args.min_validation_regimes or int(
        wf_config.get("min_validation_regimes", 1)
    )
    regime_threshold = (
        args.regime_threshold
        if args.regime_threshold is not None
        else float(wf_config.get("regime_threshold", 0.05))
    )
    if not np.isfinite(regime_threshold) or regime_threshold <= 0:
        raise SystemExit("--regime-threshold must be finite and positive")
    test_days = args.test_days or int(wf_config.get("test_days", 20))
    minimum_coverage = (
        args.minimum_coverage
        if args.minimum_coverage is not None
        else float(wf_config.get("minimum_coverage", 0.95))
    )
    if not 0.0 < minimum_coverage <= 1.0:
        raise SystemExit("--minimum-coverage must be in (0, 1]")
    folds = build_expanding_walk_forward_folds(
        all_data,
        n_folds=n_folds,
        validation_days=validation_days,
        test_days=test_days,
        purge_days=purge_days,
        reference_days=reference_days,
    )
    plan = [
        describe_fold(
            fold,
            all_data,
            reference_days=reference_days,
            minimum_coverage=minimum_coverage,
            validation_window_days=validation_window_days,
            regime_threshold=regime_threshold,
            min_validation_regimes=min_validation_regimes,
        )
        for fold in folds
    ]
    failed_regimes = [
        item["index"] for item in plan if not item["validation_regime_check_passed"]
    ]
    failed_coverage = [item["index"] for item in plan if not item["coverage_check_passed"]]
    if args.seeds is not None and args.seed is not None:
        raise SystemExit("--seed and --seeds are mutually exclusive")
    fallback_seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    training_seeds = _parse_training_seeds(args.seeds, fallback_seed)
    if args.plan_only:
        print(json.dumps({
            "symbols": symbols,
            "calendar_alignment": {
                "source_day_counts": source_day_counts,
                "common_day_count": common_day_count,
                "reference_day_count": len(reference_days),
                "reference_start": reference_days[0].isoformat(),
                "reference_end": reference_days[-1].isoformat(),
                "minimum_coverage": minimum_coverage,
            },
            "feature_columns": feature_columns,
            "excluded_features": excluded_features,
            "training_seeds": training_seeds,
            "folds": plan,
        }, indent=2))
        return
    if failed_coverage:
        raise SystemExit(
            "data coverage check failed for folds "
            f"{failed_coverage}; repair/quarantine missing minute data before training"
        )
    if failed_regimes:
        raise SystemExit(
            "validation regime diversity check failed for folds "
            f"{failed_regimes}; collect more history or change only the audit parameters"
        )

    output_dir = _make_output_dir(args.output_dir)
    total_timesteps = args.total_timesteps or int(config["agent"]["total_timesteps"])
    fold_results: list[dict[str, Any]] = []

    for fold, fold_plan in zip(folds, plan):
        logger.info(
            "Fold %d: train %s..%s, validation %s..%s, test %s..%s (%s)",
            fold.index,
            fold.train_start,
            fold.train_end,
            fold.validation_start,
            fold.validation_end,
            fold.test_start,
            fold.test_end,
            fold_plan["test_regime"],
        )
        train_data = {
            symbol: slice_walk_forward_fold(data, fold, segment="train")
            for symbol, data in all_data.items()
        }
        validation_data = {
            symbol: slice_walk_forward_fold(data, fold, segment="validation")
            for symbol, data in all_data.items()
        }
        test_data = {
            symbol: slice_walk_forward_fold(data, fold, segment="test")
            for symbol, data in all_data.items()
        }
        boundaries = SplitBoundaries(
            fold.train_end, fold.validation_end, purge_days=fold.purge_days
        )
        for training_seed in training_seeds:
            logger.info("Fold %d training seed %d", fold.index, training_seed)
            fold_config = copy.deepcopy(config)
            if args.tensorboard is not None:
                fold_config["agent"]["tensorboard"]["enabled"] = args.tensorboard
            fold_config["agent"]["tensorboard"]["log_dir"] = str(
                output_dir / "tensorboard"
            )
            fold_config["agent"]["tensorboard"]["log_name"] = (
                f"fold_{fold.index:02d}_seed_{training_seed}"
            )
            artifact = train_ppo_artifact(
                featured_data=train_data,
                validation_data=validation_data,
                config=fold_config,
                total_timesteps=total_timesteps,
                seed=training_seed,
                artifacts_dir=(
                    output_dir / "artifacts" / f"fold_{fold.index:02d}"
                    / f"seed_{training_seed}"
                ),
                trained_split="train",
                split_boundaries=boundaries.to_metadata(),
                tensorboard_log_dir=output_dir / "tensorboard",
                feature_columns=feature_columns,
            )

            agent = None
            metadata = None
            per_symbol: dict[str, dict[str, Any]] = {}
            for offset, (symbol, frame) in enumerate(test_data.items()):
                environment = build_backtest_environment(
                    frame, fold_config, feature_columns=feature_columns
                )
                if agent is None:
                    agent, metadata = load_artifact(artifact, env=environment)
                else:
                    check_env_compatibility(metadata, environment)
                model_result = run_agent_backtest(
                    agent=agent,
                    agent_name=metadata.artifact_id,
                    environment=environment,
                    max_steps=None,
                    seed=training_seed,
                )
                baselines = compare_baselines(
                    featured_data=frame,
                    config=fold_config,
                    max_steps=None,
                    seed=training_seed + offset,
                    artifact_path=None,
                )
                per_symbol[symbol] = {"model": model_result, "baselines": baselines}

            validation_summary = metadata.training_params.get("validation", {})
            qualified_validation = validation_summary.get("best") or {}
            diagnostic_validation = (
                qualified_validation
                or validation_summary.get("best_candidate")
                or validation_summary.get("latest")
                or {}
            )
            result = {
                "fold": fold_plan,
                "training_seed": training_seed,
                "artifact": str(artifact),
                "deployment_status": metadata.deployment_status,
                "validation_qualified": bool(
                    qualified_validation.get("qualified", False)
                    and metadata.deployment_status == "approved"
                ),
                "validation_best": diagnostic_validation,
                "aggregate": aggregate_model_metrics(per_symbol),
                "per_symbol": per_symbol,
            }
            fold_results.append(result)
            result_path = output_dir / (
                f"fold_{fold.index:02d}_seed_{training_seed}.json"
            )
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )

    fold_returns = np.asarray(
        [result["aggregate"]["mean_total_return"] for result in fold_results]
    )
    def _group_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
        values = np.asarray(
            [row["aggregate"]["mean_total_return"] for row in rows], dtype=float
        )
        return {
            "models": int(len(rows)),
            "mean_return": float(np.mean(values)),
            "std_return": float(np.std(values)),
            "worst_return": float(np.min(values)),
            "positive_rate": float(np.mean(values > 0.0)),
            "qualified_rate": float(
                np.mean([row["validation_qualified"] for row in rows])
            ),
        }

    summary = {
        "symbols": symbols,
        "feature_schema_version": FeatureEngineer.FEATURE_SCHEMA_VERSION,
        "feature_columns": feature_columns,
        "excluded_features": excluded_features,
        "training_seeds": training_seeds,
        "total_timesteps_per_fold": total_timesteps,
        "total_models": len(fold_results),
        "calendar": {
            "source_day_counts": source_day_counts,
            "common_feature_days": common_day_count,
            "reference_days": len(reference_days),
            "minimum_coverage": minimum_coverage,
        },
        "holdout_status": "research_reused_not_pristine",
        "folds": fold_results,
        "by_seed": {
            str(seed): _group_stats(
                [row for row in fold_results if row["training_seed"] == seed]
            )
            for seed in training_seeds
        },
        "by_fold": {
            str(fold.index): _group_stats(
                [
                    row for row in fold_results
                    if row["fold"]["index"] == fold.index
                ]
            )
            for fold in folds
        },
        "across_folds": {
            "mean_return": float(np.mean(fold_returns)),
            "median_return": float(np.median(fold_returns)),
            "worst_return": float(np.min(fold_returns)),
            "std_return": float(np.std(fold_returns)),
            "positive_fold_seed_rate": float(np.mean(fold_returns > 0.0)),
            "validation_qualified_fold_rate": float(
                np.mean([result["validation_qualified"] for result in fold_results])
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **summary["across_folds"]}, indent=2))


if __name__ == "__main__":
    main()
