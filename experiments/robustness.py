"""Train independent seeds across expanding walk-forward folds."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
AGENT_SRC = PROJECT_ROOT / "agent" / "src"

for path in (ENV_SRC, AGENT_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.common import load_feature_data, make_data_loader
from models.artifact import load_metadata
from models.training import train_ppo_artifact
from models.walk_forward import generate_expanding_walk_forward_folds
from policies import SUPPORTED_BASELINES
from policies.evaluation import evaluate_artifact, evaluate_baseline
from utils.config_loader import load_config
from utils.logger import setup_logger


def _parse_ints(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("requires at least one integer")
    return values


def _parse_floats(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("requires at least one number")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent-seed PPO robustness and walk-forward evaluation"
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--seeds", type=_parse_ints, default=[11, 22, 33, 44, 55])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-days", type=int, default=None)
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--ent-coefs", type=_parse_floats, default=[0.01])
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts") / "robustness",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path; defaults to results/robustness-<timestamp>.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe folds and experiment count without training.",
    )
    return parser.parse_args()


def summarize_robustness_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independent training runs by entropy coefficient."""
    summaries: dict[str, Any] = {}
    coefficients = sorted({float(run["ent_coef"]) for run in runs})
    for coefficient in coefficients:
        group = [run for run in runs if float(run["ent_coef"]) == coefficient]
        returns = [float(run["test_metrics"]["total_return"]) for run in group]
        excess_buy_hold = [float(run["excess_vs_buy_and_hold"]) for run in group]
        excess_static_80 = [float(run["excess_vs_static_80pct"]) for run in group]
        max_drawdowns = [float(run["test_metrics"]["max_drawdown"]) for run in group]
        turnovers = [float(run["test_metrics"]["turnover"]) for run in group]
        summaries[str(coefficient)] = {
            "runs": len(group),
            "median_test_return": statistics.median(returns),
            "mean_test_return": statistics.fmean(returns),
            "std_test_return": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
            "worst_test_return": min(returns),
            "median_excess_vs_buy_and_hold": statistics.median(excess_buy_hold),
            "median_excess_vs_static_80pct": statistics.median(excess_static_80),
            "win_rate_vs_buy_and_hold": sum(value > 0 for value in excess_buy_hold) / len(group),
            "win_rate_vs_static_80pct": sum(value > 0 for value in excess_static_80) / len(group),
            "median_max_drawdown": statistics.median(max_drawdowns),
            "median_turnover": statistics.median(turnovers),
        }
    return summaries


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "results" / f"robustness-{timestamp}.json"


def main() -> None:
    args = parse_args()
    if args.folds <= 0 or args.total_timesteps <= 0:
        raise SystemExit("--folds and --total-timesteps must be positive")

    logger = setup_logger()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    symbol = args.symbol or config["data"]["symbol"]
    loader = make_data_loader(
        project_root=PROJECT_ROOT,
        config=config,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    all_data = load_feature_data(
        symbol=symbol,
        data_loader=loader,
        force_rebuild=args.force_rebuild,
    )
    folds = generate_expanding_walk_forward_folds(
        all_data,
        n_folds=args.folds,
        validation_days=args.validation_days,
        test_days=args.test_days,
    )
    fold_descriptions = [fold.describe() for fold in folds]
    experiment_count = len(folds) * len(args.seeds) * len(args.ent_coefs)
    logger.info(
        "Robustness plan: %d folds x %d seeds x %d entropy settings = %d models",
        len(folds),
        len(args.seeds),
        len(args.ent_coefs),
        experiment_count,
    )

    output_path = args.output or _default_output_path()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.is_absolute():
        artifacts_dir = PROJECT_ROOT / artifacts_dir

    payload: dict[str, Any] = {
        "symbol": symbol,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "ent_coefs": args.ent_coefs,
        "total_timesteps": args.total_timesteps,
        "folds": fold_descriptions,
        "experiment_count": experiment_count,
        "runs": [],
        "baselines": [],
    }
    if args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(output_path)
        return

    run_number = 0
    for fold in folds:
        baseline_returns: dict[str, float] = {}
        for baseline_name in SUPPORTED_BASELINES:
            baseline = evaluate_baseline(
                baseline_name=baseline_name,
                featured_data=fold.test,
                config=config,
                max_steps=None,
                seed=fold.index,
            )
            payload["baselines"].append(
                {"fold": fold.index, **baseline}
            )
            baseline_returns[baseline_name] = float(
                baseline["metrics"]["total_return"]
            )

        for ent_coef in args.ent_coefs:
            for seed in args.seeds:
                run_number += 1
                logger.info(
                    "Training run %d/%d: fold=%d seed=%d ent_coef=%g",
                    run_number,
                    experiment_count,
                    fold.index,
                    seed,
                    ent_coef,
                )
                run_config = copy.deepcopy(config)
                run_config["agent"]["tensorboard"]["enabled"] = False
                run_config["agent"]["validation"]["verbose"] = 0
                run_config["agent"]["ppo"]["verbose"] = 0
                run_config["agent"]["ppo"]["ent_coef"] = ent_coef
                artifact_path = train_ppo_artifact(
                    featured_data=fold.train,
                    validation_data=fold.validation,
                    symbol=symbol,
                    config=run_config,
                    total_timesteps=args.total_timesteps,
                    seed=seed,
                    device=args.device,
                    artifacts_dir=artifacts_dir,
                    model_kwargs={"ent_coef": ent_coef, "verbose": 0},
                )
                evaluation = evaluate_artifact(
                    artifact_path=artifact_path,
                    featured_data=fold.test,
                    config=run_config,
                    max_steps=None,
                    seed=seed,
                )
                metadata = load_metadata(artifact_path)
                metrics = evaluation["metrics"]
                run = {
                    "fold": fold.index,
                    "seed": seed,
                    "ent_coef": ent_coef,
                    "artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
                    "validation": metadata.training_params.get("validation"),
                    "test_metrics": metrics,
                    "excess_vs_buy_and_hold": (
                        float(metrics["total_return"])
                        - baseline_returns["buy_and_hold"]
                    ),
                    "excess_vs_static_80pct": (
                        float(metrics["total_return"])
                        - baseline_returns["static_80pct"]
                    ),
                }
                payload["runs"].append(run)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(
                        {
                            **payload,
                            "summary_by_ent_coef": summarize_robustness_runs(
                                payload["runs"]
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    payload["summary_by_ent_coef"] = summarize_robustness_runs(payload["runs"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
