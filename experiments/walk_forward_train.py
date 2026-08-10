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
from models.artifact import check_env_compatibility, load_artifact
from models.training import train_ppo_artifact
from models.walk_forward import (
    SplitBoundaries,
    build_expanding_walk_forward_folds,
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
    parser.add_argument("--test-days", type=_positive_int, default=None)
    parser.add_argument("--purge-days", type=int, default=None)
    parser.add_argument("--total-timesteps", type=_positive_int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def classify_market_regime(mean_market_return: float, threshold: float = 0.05) -> str:
    """Human-readable ex-post label; never used to construct a fold."""
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


def describe_fold(fold, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, Any]:
    market_returns = {
        symbol: _market_return(slice_walk_forward_fold(data, fold, segment="test"))
        for symbol, data in data_by_symbol.items()
    }
    mean_market_return = float(np.mean(list(market_returns.values())))
    return {
        **fold.to_dict(),
        "test_market_returns": market_returns,
        "mean_test_market_return": mean_market_return,
        "test_regime": classify_market_regime(mean_market_return),
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
    symbols = resolve_symbols(
        config=config, cli_symbol=args.symbol, cli_symbols=args.symbols
    )
    loader = make_data_loader(project_root=PROJECT_ROOT, config=config)
    all_data = {
        symbol: load_feature_data(
            symbol=symbol, data_loader=loader, force_rebuild=args.force_rebuild
        )
        for symbol in symbols
    }
    purge_days = (
        args.purge_days
        if args.purge_days is not None
        else int(config["data"].get("split", {}).get("purge_days", 0))
    )
    walk_forward_config = config.get("walk_forward", {})
    n_folds = args.folds or int(walk_forward_config.get("folds", 3))
    validation_days = args.validation_days or int(
        walk_forward_config.get("validation_days", 20)
    )
    test_days = args.test_days or int(walk_forward_config.get("test_days", 20))
    folds = build_expanding_walk_forward_folds(
        all_data,
        n_folds=n_folds,
        validation_days=validation_days,
        test_days=test_days,
        purge_days=purge_days,
    )
    plan = [describe_fold(fold, all_data) for fold in folds]
    if args.plan_only:
        print(json.dumps({"symbols": symbols, "folds": plan}, indent=2))
        return

    output_dir = _make_output_dir(args.output_dir)
    total_timesteps = args.total_timesteps or int(config["agent"]["total_timesteps"])
    base_seed = args.seed if args.seed is not None else int(config.get("seed", 42))
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
        fold_config = copy.deepcopy(config)
        if args.tensorboard is not None:
            fold_config["agent"]["tensorboard"]["enabled"] = args.tensorboard
        fold_config["agent"]["tensorboard"]["log_dir"] = str(
            output_dir / "tensorboard"
        )
        fold_config["agent"]["tensorboard"]["log_name"] = f"fold_{fold.index:02d}"
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
        artifact = train_ppo_artifact(
            featured_data=train_data,
            validation_data=validation_data,
            config=fold_config,
            total_timesteps=total_timesteps,
            seed=base_seed + fold.index - 1,
            artifacts_dir=output_dir / "artifacts" / f"fold_{fold.index:02d}",
            trained_split="train",
            split_boundaries=boundaries.to_metadata(),
            tensorboard_log_dir=output_dir / "tensorboard",
        )

        agent = None
        metadata = None
        per_symbol: dict[str, dict[str, Any]] = {}
        for offset, (symbol, frame) in enumerate(test_data.items()):
            environment = build_backtest_environment(frame, fold_config)
            if agent is None:
                agent, metadata = load_artifact(artifact, env=environment)
            else:
                check_env_compatibility(metadata, environment)
            model_result = run_agent_backtest(
                agent=agent,
                agent_name=metadata.artifact_id,
                environment=environment,
                max_steps=None,
                seed=base_seed,
            )
            baselines = compare_baselines(
                featured_data=frame,
                config=fold_config,
                max_steps=None,
                seed=base_seed + offset,
                artifact_path=None,
            )
            per_symbol[symbol] = {"model": model_result, "baselines": baselines}

        result = {
            "fold": fold_plan,
            "artifact": str(artifact),
            "aggregate": aggregate_model_metrics(per_symbol),
            "per_symbol": per_symbol,
        }
        fold_results.append(result)
        (output_dir / f"fold_{fold.index:02d}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )

    fold_returns = np.asarray(
        [result["aggregate"]["mean_total_return"] for result in fold_results]
    )
    summary = {
        "symbols": symbols,
        "total_timesteps_per_fold": total_timesteps,
        "folds": fold_results,
        "across_folds": {
            "mean_return": float(np.mean(fold_returns)),
            "median_return": float(np.median(fold_returns)),
            "worst_return": float(np.min(fold_returns)),
            "positive_fold_rate": float(np.mean(fold_returns > 0.0)),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **summary["across_folds"]}, indent=2))


if __name__ == "__main__":
    main()
