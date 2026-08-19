"""Measure feature signal strength and stability before expensive RL training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "env" / "src", PROJECT_ROOT / "agent" / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.feature_engineer import FeatureEngineer
from env.observation import adv_value_from_row, liquidity_score_from_adv
from experiments.common import load_feature_data, make_data_loader, resolve_symbols
from friction.friction_model import FrictionModel
from models.walk_forward import (
    align_data_on_common_days,
    build_expanding_walk_forward_folds,
    slice_walk_forward_fold,
)
from utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--horizons", default="1,3,12")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def _forward_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    close = pd.to_numeric(frame["Close"], errors="coerce")
    days = pd.to_datetime(frame["Timestamp"]).dt.date
    future = close.groupby(days).shift(-horizon)
    return future / close - 1.0


def _ic(frame: pd.DataFrame, feature: str, horizon: int) -> tuple[float, int]:
    sample = pd.DataFrame({
        "feature": pd.to_numeric(frame[feature], errors="coerce"),
        "target": _forward_return(frame, horizon),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 30 or sample["feature"].nunique() < 2:
        return float("nan"), len(sample)
    return float(sample["feature"].corr(sample["target"], method="spearman")), len(sample)


def _bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int = 0
) -> list[float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size < 2 or samples < 1:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(finite, size=(samples, finite.size), replace=True), axis=1
    )
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _daily_ic(
    frame: pd.DataFrame,
    feature: str,
    horizon: int,
    *,
    bootstrap_samples: int,
) -> dict:
    work = frame.copy()
    work["__target"] = _forward_return(work, horizon)
    work["__day"] = pd.to_datetime(work["Timestamp"]).dt.date
    values: list[float] = []
    for _, group in work.groupby("__day"):
        sample = group[[feature, "__target"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(sample) >= 10 and sample[feature].nunique() >= 2:
            values.append(float(sample[feature].corr(sample["__target"], method="spearman")))
    finite = [value for value in values if np.isfinite(value)]
    return {
        "days": len(finite),
        "mean_ic": float(np.mean(finite)) if finite else 0.0,
        "median_ic": float(np.median(finite)) if finite else 0.0,
        "bootstrap_95_ci": _bootstrap_mean_ci(
            finite, samples=bootstrap_samples, seed=0
        ),
    }


def _non_overlapping_ic(
    frame: pd.DataFrame, feature: str, horizon: int
) -> tuple[float, int]:
    """IC on one fixed phase per day so forward-return labels do not overlap."""
    work = frame.copy()
    work["__target"] = _forward_return(work, horizon)
    work["__day"] = pd.to_datetime(work["Timestamp"]).dt.date
    sampled = work.groupby("__day", group_keys=False).apply(
        lambda group: group.iloc[::horizon], include_groups=False
    )
    sample = sampled[[feature, "__target"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(sample) < 30 or sample[feature].nunique() < 2:
        return float("nan"), len(sample)
    return float(sample[feature].corr(sample["__target"], method="spearman")), len(sample)


def _roundtrip_cost_rates(frame: pd.DataFrame, config: dict) -> pd.Series:
    friction = FrictionModel(**config["friction"])
    unit_notional = float(config["environment"]["initial_cash"]) * float(
        config["environment"]["unit_fraction"]
    )
    rates = []
    for _, row in frame.iterrows():
        price = float(row["Close"])
        trade_date = pd.Timestamp(row["Timestamp"]).date()
        liquidity = liquidity_score_from_adv(
            adv_value_from_row(row), unit_notional
        )
        buy = friction.calculate_total_friction(
            unit_notional, "buy", liquidity, price, trade_date
        )
        sell = friction.calculate_total_friction(
            unit_notional, "sell", liquidity, price, trade_date
        )
        rates.append(float((buy + sell) / unit_notional))
    return pd.Series(rates, index=frame.index, dtype=float)


def _train_frozen_tail_audit(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    feature: str,
    horizon: int,
    quantile: float,
    config: dict,
) -> dict:
    train_ic, _ = _ic(train, feature, horizon)
    if not np.isfinite(train_ic) or train[feature].dropna().empty:
        return {"events": 0, "reason": "insufficient_train_signal"}
    lower_tail = train_ic < 0
    threshold = float(train[feature].quantile(quantile if lower_tail else 1 - quantile))
    work = target.reset_index(drop=True).copy()
    work["__target"] = _forward_return(work, horizon)
    work["__cost"] = _roundtrip_cost_rates(work, config)
    condition = work[feature] <= threshold if lower_tail else work[feature] >= threshold
    chosen: list[int] = []
    days = pd.to_datetime(work["Timestamp"]).dt.date
    for _, indices in work.groupby(days).groups.items():
        next_allowed = -1
        for idx in sorted(indices):
            if idx >= next_allowed and bool(condition.iloc[idx]):
                chosen.append(idx)
                next_allowed = idx + horizon
    sample = work.loc[chosen, ["__target", "__cost", "Timestamp"]].dropna()
    if sample.empty:
        return {
            "events": 0,
            "train_ic": float(train_ic),
            "threshold": threshold,
            "tail": "lower" if lower_tail else "upper",
        }
    sample["__net"] = sample["__target"] - sample["__cost"]
    daily_net = sample.groupby(pd.to_datetime(sample["Timestamp"]).dt.date)["__net"].mean()
    return {
        "events": int(len(sample)),
        "event_days": int(len(daily_net)),
        "train_ic": float(train_ic),
        "threshold": threshold,
        "tail": "lower" if lower_tail else "upper",
        "mean_gross_return": float(sample["__target"].mean()),
        "mean_roundtrip_cost": float(sample["__cost"].mean()),
        "mean_net_return": float(sample["__net"].mean()),
        "net_win_rate": float((sample["__net"] > 0).mean()),
        "daily_net_bootstrap_95_ci": _bootstrap_mean_ci(
            daily_net.tolist(), samples=1000, seed=0
        ),
    }


def _summarize(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"count": 0, "mean_ic": 0.0, "median_abs_ic": 0.0,
                "sign_consistency": 0.0}
    positive = float(np.mean(finite > 0))
    return {
        "count": int(finite.size),
        "mean_ic": float(np.mean(finite)),
        "median_abs_ic": float(np.median(np.abs(finite))),
        "sign_consistency": float(max(positive, 1.0 - positive)),
    }


def main() -> None:
    args = parse_args()
    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
    if not horizons or min(horizons) < 1:
        raise SystemExit("--horizons must contain positive integers")
    if not 0.0 < args.tail_quantile < 0.5:
        raise SystemExit("--tail-quantile must be in (0, 0.5)")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    symbols = resolve_symbols(config=config, cli_symbol=None, cli_symbols=args.symbols)
    loader = make_data_loader(project_root=PROJECT_ROOT, config=config)
    raw = {
        symbol: load_feature_data(
            symbol=symbol, data_loader=loader, force_rebuild=args.force_rebuild
        )
        for symbol in symbols
    }
    data = align_data_on_common_days(raw)
    wf = config.get("walk_forward", {})
    folds = build_expanding_walk_forward_folds(
        data,
        n_folds=int(wf.get("folds", 3)),
        validation_days=int(wf.get("validation_days", 60)),
        test_days=int(wf.get("test_days", 20)),
        purge_days=int(config["data"].get("split", {}).get("purge_days", 0)),
    )

    observations = []
    frozen_tail_audits = []
    for fold in folds:
        train_by_symbol = {
            symbol: slice_walk_forward_fold(frame, fold, segment="train")
            for symbol, frame in data.items()
        }
        for segment in ("train", "validation", "test"):
            for symbol, frame in data.items():
                split = slice_walk_forward_fold(frame, fold, segment=segment)
                for feature in FeatureEngineer.FEATURE_COLUMNS:
                    for horizon in horizons:
                        ic, rows = _ic(split, feature, horizon)
                        nonoverlap_ic, nonoverlap_rows = _non_overlapping_ic(
                            split, feature, horizon
                        )
                        observations.append({
                            "fold": fold.index, "segment": segment, "symbol": symbol,
                            "feature": feature, "horizon": horizon,
                            "ic": ic if np.isfinite(ic) else None,
                            "observations": rows,
                            "non_overlapping_ic": (
                                nonoverlap_ic if np.isfinite(nonoverlap_ic) else None
                            ),
                            "non_overlapping_observations": nonoverlap_rows,
                            "daily_ic": _daily_ic(
                                split,
                                feature,
                                horizon,
                                bootstrap_samples=args.bootstrap_samples,
                            ),
                        })
                if segment in ("validation", "test"):
                    for feature in ("log_ret_1", "log_ret_12", "vwap_dev"):
                        for horizon in horizons:
                            frozen_tail_audits.append({
                                "fold": fold.index,
                                "segment": segment,
                                "symbol": symbol,
                                "feature": feature,
                                "horizon": horizon,
                                "quantile": args.tail_quantile,
                                **_train_frozen_tail_audit(
                                    train_by_symbol[symbol],
                                    split,
                                    feature=feature,
                                    horizon=horizon,
                                    quantile=args.tail_quantile,
                                    config=config,
                                ),
                            })

    table = pd.DataFrame(observations)
    summaries = []
    for (feature, horizon), group in table.groupby(["feature", "horizon"]):
        validation = _summarize(group.loc[group.segment == "validation", "ic"].tolist())
        test = _summarize(group.loc[group.segment == "test", "ic"].tolist())
        summaries.append({
            "feature": feature,
            "horizon": int(horizon),
            "validation": validation,
            "test_audit": test,
            "validation_score": float(
                validation["median_abs_ic"] * validation["sign_consistency"]
            ),
        })
    summaries.sort(key=lambda item: item["validation_score"], reverse=True)
    common_days = pd.to_datetime(next(iter(data.values()))["Timestamp"]).dt.date
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "common_days": int(common_days.nunique()),
        "date_range": [str(common_days.min()), str(common_days.max())],
        "horizons": horizons,
        "method": "Spearman IC; forward returns never cross a trading-day boundary",
        "robustness_method": (
            "daily IC with trading-day bootstrap; non-overlapping fixed-phase IC; "
            "train-frozen tail thresholds with modeled round-trip costs"
        ),
        "ranking_basis": "validation median_abs_ic * sign_consistency",
        "summary": summaries,
        "observations": observations,
        "train_frozen_tail_audits": frozen_tail_audits,
    }
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output = PROJECT_ROOT / "runs" / "signal_reports" / f"signal-{stamp}.json"
    elif not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "top_signals": summaries[:10]}, indent=2))


if __name__ == "__main__":
    main()
