"""Generate feature-quality and trained-policy usage reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
AGENT_SRC = PROJECT_ROOT / "agent" / "src"
for path in (ENV_SRC, AGENT_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data.feature_engineer import FeatureEngineer
from experiments.backtest import resolve_backtest_symbols, resolve_boundaries
from experiments.common import load_feature_data, make_data_loader, resolve_project_path
from models.artifact import check_env_compatibility, load_artifact, load_metadata
from models.feature_diagnostics import (
    build_feature_quality_report,
    feature_redundancy_report,
    feature_temporal_stability_report,
    permutation_policy_sensitivity,
)
from models.walk_forward import split_by_trading_day
from policies.evaluation import build_backtest_environment
from utils.config_loader import load_config


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature quality and policy-use report")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--symbols", default=None)
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="validation"
    )
    parser.add_argument("--max-samples", type=_positive_int, default=5000)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--stability-window-days", type=_positive_int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def collect_policy_observations(agent, environment) -> tuple[np.ndarray, np.ndarray]:
    dates = environment.available_dates
    observation, _ = environment.reset(
        seed=0, options={"start_date": dates[0], "episode_days": len(dates)}
    )
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    done = False
    while not done:
        mask = np.asarray(environment.action_masks(), dtype=bool)
        observations.append(np.asarray(observation, dtype=np.float32))
        masks.append(mask)
        action, _ = agent.predict(observation, deterministic=True, action_masks=mask)
        observation, _, terminated, truncated, _ = environment.step(action)
        done = bool(terminated or truncated)
    return np.asarray(observations), np.asarray(masks)


def _default_output(artifact_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "runs" / "feature_reports" / f"{artifact_id}-{stamp}.json"


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    artifact = resolve_project_path(PROJECT_ROOT, args.artifact)
    metadata = load_metadata(artifact)
    symbols = resolve_backtest_symbols(
        config=config,
        meta=metadata,
        cli_symbol=args.symbol,
        cli_symbols=args.symbols,
    )
    loader = make_data_loader(project_root=PROJECT_ROOT, config=config)
    all_data = {
        symbol: load_feature_data(symbol=symbol, data_loader=loader)
        for symbol in symbols
    }
    boundaries = resolve_boundaries(meta=metadata, data_by_symbol=all_data, config=config)
    selected = {
        symbol: split_by_trading_day(data, split=args.split, boundaries=boundaries)
        for symbol, data in all_data.items()
    }

    agent = None
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for frame in selected.values():
        environment = build_backtest_environment(
            frame, config, feature_columns=metadata.feature_columns
        )
        if agent is None:
            agent, _ = load_artifact(artifact, env=environment)
        else:
            check_env_compatibility(metadata, environment)
        symbol_observations, symbol_masks = collect_policy_observations(agent, environment)
        observations.append(symbol_observations)
        masks.append(symbol_masks)
    pooled_observations = np.concatenate(observations)
    pooled_masks = np.concatenate(masks)
    if len(pooled_observations) > args.max_samples:
        indices = np.linspace(0, len(pooled_observations) - 1, args.max_samples, dtype=int)
        pooled_observations = pooled_observations[indices]
        pooled_masks = pooled_masks[indices]

    quality = build_feature_quality_report(
        selected,
        metadata.feature_columns,
        normalization=agent.observation_normalizer,
    )
    observation_fields = [*metadata.feature_columns, *metadata.portfolio_state_fields]
    sensitivity = permutation_policy_sensitivity(
        agent,
        pooled_observations,
        pooled_masks,
        observation_fields,
        repeats=args.repeats,
    )
    sensitivity_by_field = {row["feature"]: row for row in sensitivity}
    rows = [
        {
            **quality_row,
            **sensitivity_by_field[quality_row["feature"]],
            "description": FeatureEngineer.FEATURE_DESCRIPTIONS.get(
                quality_row["feature"], ""
            ),
        }
        for quality_row in quality
    ]
    rows.sort(key=lambda row: row["probability_shift"], reverse=True)

    feature_count = len(metadata.feature_columns)
    state_rows = []
    for index, field in enumerate(metadata.portfolio_state_fields):
        values = pooled_observations[:, feature_count + index]
        state_rows.append({
            **sensitivity_by_field[field],
            "mean": float(values.mean()),
            "std": float(values.std()),
            "nonzero_rate": float(np.mean(np.abs(values) > 1e-8)),
        })
    state_rows.sort(key=lambda row: row["probability_shift"], reverse=True)

    payload = {
        "artifact_id": metadata.artifact_id,
        "feature_schema_version": metadata.feature_schema_version,
        "split": args.split,
        "symbols": symbols,
        "sample_count": int(len(pooled_observations)),
        "features": rows,
        "portfolio_state_sensitivity": state_rows,
        "feature_redundancy": feature_redundancy_report(
            selected, metadata.feature_columns
        ),
        "temporal_stability": feature_temporal_stability_report(
            selected,
            metadata.feature_columns,
            window_days=args.stability_window_days,
        ),
        "interpretation": {
            "policy_sensitivity": (
                "Permutation sensitivity on states visited by the deterministic policy; "
                "it measures reliance, not causal predictive value. Fixed masks and "
                "constant state fields can make sensitivity unidentifiable."
            ),
            "forward_correlations": (
                "Same-day forward log-return Spearman correlations. Small values and "
                "sign changes are weak or unstable directional evidence."
            ),
        },
    }
    output = args.output or _default_output(metadata.artifact_id)
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"{'feature':<24} {'prob_shift':>10} {'flip':>8} {'clip':>8} "
        f"{'corr+1':>9} {'corr+12':>9}"
    )
    for row in rows:
        print(
            f"{row['feature']:<24} {row['probability_shift']:>10.5f} "
            f"{row['action_flip_rate']:>8.3%} "
            f"{(row['normalization_clip_rate'] or 0.0):>8.3%} "
            f"{row['spearman_forward_1']:>9.4f} "
            f"{row['spearman_forward_12']:>9.4f}"
        )
    print("\nPortfolio-state policy sensitivity")
    for row in state_rows:
        print(
            f"{row['feature']:<24} {row['probability_shift']:>10.5f} "
            f"{row['action_flip_rate']:>8.3%} nonzero={row['nonzero_rate']:>7.2%}"
        )
    print("\nMost correlated feature pairs")
    for pair in payload["feature_redundancy"][:5]:
        print(
            f"{pair['feature_a']} <> {pair['feature_b']}: "
            f"{pair['spearman_correlation']:.3f}"
        )
    print(json.dumps({"output": str(output), "sample_count": len(pooled_observations)}))


if __name__ == "__main__":
    main()
