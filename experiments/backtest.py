"""Backtest a saved PPO artifact or a simple baseline agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
AGENT_SRC = PROJECT_ROOT / "agent" / "src"

for path in (ENV_SRC, AGENT_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.common import load_feature_data, make_data_loader, resolve_project_path
from models.walk_forward import split_by_trading_day
from policies import SUPPORTED_BASELINES
from policies.evaluation import (
    compare_baselines,
    compare_baselines_multi_seed,
    evaluate_artifact,
    evaluate_artifact_multi_seed,
    evaluate_baseline,
    evaluate_baseline_multi_seed,
)
from utils.config_loader import load_config
from utils.logger import setup_logger


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest PPO artifact or baseline")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--baseline", choices=SUPPORTED_BASELINES, default=None)
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for robustness evaluation, e.g. 1,2,3")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="test")
    parser.add_argument(
        "--episode-days",
        type=_positive_int,
        default=None,
        help="Trading days per episode; overrides environment.episode_days.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    return parser.parse_args()


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    seeds = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not seeds:
        raise SystemExit("--seeds requires at least one integer")
    return seeds


def main() -> None:
    logger = setup_logger()
    args = parse_args()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    if args.episode_days is not None:
        config["environment"]["episode_days"] = args.episode_days

    symbol = args.symbol or config["data"]["symbol"]
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    seeds = _parse_seeds(args.seeds)
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = config["backtest"].get("max_steps")

    data_loader = make_data_loader(
        project_root=PROJECT_ROOT,
        config=config,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    featured_data = load_feature_data(
        symbol=symbol,
        data_loader=data_loader,
        force_rebuild=args.force_rebuild,
    )
    featured_data = split_by_trading_day(featured_data, split=args.split)
    logger.info(
        "Evaluation episode length: %d trading day(s)",
        int(config["environment"].get("episode_days", 1)),
    )

    if args.compare_baselines:
        logger.info("Backtesting baseline suite on %s (%s split)", symbol, args.split)
        if seeds is None:
            summaries = compare_baselines(
                featured_data=featured_data,
                config=config,
                max_steps=max_steps,
                seed=seed,
                artifact_path=resolve_project_path(PROJECT_ROOT, args.artifact) if args.artifact else None,
            )
        else:
            summaries = compare_baselines_multi_seed(
                featured_data=featured_data,
                config=config,
                max_steps=max_steps,
                seeds=seeds,
                artifact_path=resolve_project_path(PROJECT_ROOT, args.artifact) if args.artifact else None,
            )
        print(json.dumps(
            {"symbol": symbol, "split": args.split, "seeds": seeds, "results": summaries},
            indent=2,
            sort_keys=True,
        ))
        return

    if args.artifact is not None:
        logger.info("Backtesting artifact on %s (%s split)", symbol, args.split)
        if seeds is None:
            summary = evaluate_artifact(
                artifact_path=resolve_project_path(PROJECT_ROOT, args.artifact),
                featured_data=featured_data,
                config=config,
                max_steps=max_steps,
                seed=seed,
            )
        else:
            summary = evaluate_artifact_multi_seed(
                artifact_path=resolve_project_path(PROJECT_ROOT, args.artifact),
                featured_data=featured_data,
                config=config,
                max_steps=max_steps,
                seeds=seeds,
            )
    else:
        baseline_name = args.baseline or config["agent"].get("baseline_name", "buy_and_hold")
        logger.info("Backtesting %s on %s (%s split)", baseline_name, symbol, args.split)
        if seeds is None:
            summary = evaluate_baseline(
                baseline_name=baseline_name,
                featured_data=featured_data,
                config=config,
                max_steps=max_steps,
                seed=seed,
            )
        else:
            summary = evaluate_baseline_multi_seed(
                baseline_name=baseline_name,
                featured_data=featured_data,
                config=config,
                max_steps=max_steps,
                seeds=seeds,
            )

    payload = {"agent": summary["agent"], "symbol": symbol, "split": args.split, "seeds": seeds}
    if seeds is None:
        payload["metrics"] = summary["metrics"]
    else:
        payload.update({
            "runs": summary["runs"],
            "mean_metrics": summary["mean_metrics"],
            "std_metrics": summary["std_metrics"],
        })
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
