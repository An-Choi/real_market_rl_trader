"""Feature quality, stability, redundancy, and trained-policy diagnostics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def build_feature_quality_report(
    data_by_symbol: dict[str, pd.DataFrame],
    feature_columns: Iterable[str],
    *,
    normalization: Any | None = None,
    horizons: tuple[int, ...] = (1, 3, 12),
) -> list[dict[str, Any]]:
    """Summarize distributions and causal same-day forward-return associations."""
    feature_columns = tuple(feature_columns)
    frames: list[pd.DataFrame] = []
    for symbol, source in data_by_symbol.items():
        frame = source[["Timestamp", "Close", *feature_columns]].copy()
        frame["symbol"] = symbol
        date = pd.to_datetime(frame["Timestamp"]).dt.date
        close = pd.to_numeric(frame["Close"], errors="coerce")
        for horizon in horizons:
            future = close.groupby(date).shift(-horizon)
            frame[f"forward_return_{horizon}"] = np.log(future / close)
        frames.append(frame)
    if not frames:
        raise ValueError("data_by_symbol must not be empty")
    pooled = pd.concat(frames, ignore_index=True)

    norm_by_feature: dict[str, tuple[float, float, float]] = {}
    if normalization is not None:
        norm_by_feature = {
            column: (mean, scale, float(normalization.clip))
            for column, mean, scale in zip(
                normalization.feature_columns,
                normalization.means,
                normalization.scales,
            )
        }

    report: list[dict[str, Any]] = []
    for feature in feature_columns:
        raw = pd.to_numeric(pooled[feature], errors="coerce")
        finite = raw.replace([np.inf, -np.inf], np.nan).dropna()
        row: dict[str, Any] = {
            "feature": feature,
            "rows": int(len(raw)),
            "missing_rate": float(raw.isna().mean()),
            "nonfinite_rate": float((~np.isfinite(raw.fillna(0.0))).mean()),
            "mean": float(finite.mean()) if not finite.empty else 0.0,
            "std": float(finite.std(ddof=0)) if not finite.empty else 0.0,
            "p01": float(finite.quantile(0.01)) if not finite.empty else 0.0,
            "median": float(finite.median()) if not finite.empty else 0.0,
            "p99": float(finite.quantile(0.99)) if not finite.empty else 0.0,
            "near_constant": bool(finite.empty or float(finite.std(ddof=0)) < 1e-8),
        }
        if feature in norm_by_feature and not finite.empty:
            mean, scale, clip = norm_by_feature[feature]
            normalized = (finite - mean) / max(scale, 1e-12)
            row["normalization_clip_rate"] = float((normalized.abs() >= clip).mean())
        else:
            row["normalization_clip_rate"] = None
        for horizon in horizons:
            target = pd.to_numeric(
                pooled.loc[finite.index, f"forward_return_{horizon}"], errors="coerce"
            )
            valid = finite.notna() & target.notna()
            feature_ranks = finite[valid].rank()
            target_ranks = target[valid].rank()
            correlation = (
                feature_ranks.corr(target_ranks)
                if int(valid.sum()) >= 3
                and float(feature_ranks.std(ddof=0)) > 0.0
                and float(target_ranks.std(ddof=0)) > 0.0
                else 0.0
            )
            row[f"spearman_forward_{horizon}"] = (
                float(correlation) if np.isfinite(correlation) else 0.0
            )
        report.append(row)
    return report


def feature_redundancy_report(
    data_by_symbol: dict[str, pd.DataFrame],
    feature_columns: Iterable[str],
) -> list[dict[str, float | str]]:
    """Rank pairwise Spearman correlations to expose redundant inputs."""
    feature_columns = tuple(feature_columns)
    pooled = pd.concat(
        [frame.loc[:, feature_columns] for frame in data_by_symbol.values()],
        ignore_index=True,
    ).apply(pd.to_numeric, errors="coerce")
    correlation = pooled.rank().corr()
    pairs = [
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "spearman_correlation": float(correlation.loc[feature_a, feature_b]),
        }
        for index, feature_a in enumerate(feature_columns)
        for feature_b in feature_columns[index + 1:]
        if np.isfinite(correlation.loc[feature_a, feature_b])
    ]
    return sorted(
        pairs,
        key=lambda row: abs(float(row["spearman_correlation"])),
        reverse=True,
    )


def feature_temporal_stability_report(
    data_by_symbol: dict[str, pd.DataFrame],
    feature_columns: Iterable[str],
    *,
    window_days: int = 20,
    horizons: tuple[int, ...] = (1, 3, 12),
) -> list[dict[str, Any]]:
    """Expose sign changes hidden by one pooled feature correlation."""
    if window_days < 1:
        raise ValueError("window_days must be positive")
    if not data_by_symbol:
        raise ValueError("data_by_symbol must not be empty")
    feature_columns = tuple(feature_columns)
    trading_days = sorted({
        day
        for frame in data_by_symbol.values()
        for day in pd.to_datetime(frame["Timestamp"]).dt.date.unique()
    })
    if not trading_days:
        raise ValueError("data_by_symbol must contain at least one trading day")

    correlations = {
        feature: {horizon: [] for horizon in horizons}
        for feature in feature_columns
    }
    window_count = 0
    for offset in range(0, len(trading_days), window_days):
        day_set = set(trading_days[offset:offset + window_days])
        selected = {
            symbol: frame.loc[
                pd.to_datetime(frame["Timestamp"]).dt.date.isin(day_set)
            ].copy()
            for symbol, frame in data_by_symbol.items()
        }
        selected = {symbol: frame for symbol, frame in selected.items() if not frame.empty}
        quality = build_feature_quality_report(
            selected, feature_columns, horizons=horizons
        )
        by_feature = {row["feature"]: row for row in quality}
        for feature in feature_columns:
            for horizon in horizons:
                correlations[feature][horizon].append(
                    float(by_feature[feature][f"spearman_forward_{horizon}"])
                )
        window_count += 1

    report: list[dict[str, Any]] = []
    for feature in feature_columns:
        row: dict[str, Any] = {
            "feature": feature,
            "window_days": window_days,
            "window_count": window_count,
        }
        for horizon in horizons:
            values = np.asarray(correlations[feature][horizon], dtype=float)
            prefix = f"spearman_forward_{horizon}"
            row[f"{prefix}_mean"] = float(values.mean())
            row[f"{prefix}_mean_abs"] = float(np.abs(values).mean())
            row[f"{prefix}_std"] = float(values.std(ddof=0))
            row[f"{prefix}_min"] = float(values.min())
            row[f"{prefix}_max"] = float(values.max())
            row[f"{prefix}_positive_window_rate"] = float((values > 0.0).mean())
            row[f"{prefix}_sign_consistency"] = float(np.abs(np.sign(values).mean()))
        report.append(row)
    return report


def _policy_probabilities(agent: Any, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
    import torch

    normalized = (
        agent.observation_normalizer.transform_observation(observations)
        if agent.observation_normalizer is not None
        else np.asarray(observations, dtype=np.float32)
    )
    policy = agent.model.policy
    obs_tensor, _ = policy.obs_to_tensor(normalized)
    with torch.no_grad():
        distribution = policy.get_distribution(
            obs_tensor, action_masks=np.asarray(masks, dtype=bool)
        )
        return distribution.distribution.probs.detach().cpu().numpy()


def permutation_policy_sensitivity(
    agent: Any,
    observations: np.ndarray,
    action_masks: np.ndarray,
    observation_fields: Iterable[str],
    *,
    repeats: int = 3,
    seed: int = 42,
) -> list[dict[str, float | str]]:
    """Measure how much each observation field changes the action distribution."""
    observations = np.asarray(observations, dtype=np.float32)
    action_masks = np.asarray(action_masks, dtype=bool)
    observation_fields = tuple(observation_fields)
    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError("observations must be a non-empty 2D array")
    if observations.shape[1] != len(observation_fields):
        raise ValueError(
            "observation field count must exactly match the observation dimension"
        )
    if repeats < 1:
        raise ValueError("repeats must be positive")

    rng = np.random.default_rng(seed)
    baseline_probs = _policy_probabilities(agent, observations, action_masks)
    baseline_actions = baseline_probs.argmax(axis=1)
    report: list[dict[str, float | str]] = []
    for index, field in enumerate(observation_fields):
        probability_shifts: list[float] = []
        action_flip_rates: list[float] = []
        for _ in range(repeats):
            permuted = observations.copy()
            permuted[:, index] = rng.permutation(permuted[:, index])
            probabilities = _policy_probabilities(agent, permuted, action_masks)
            probability_shifts.append(
                float(np.mean(0.5 * np.abs(probabilities - baseline_probs).sum(axis=1)))
            )
            action_flip_rates.append(
                float(np.mean(probabilities.argmax(axis=1) != baseline_actions))
            )
        report.append({
            "feature": field,
            "probability_shift": float(np.mean(probability_shifts)),
            "action_flip_rate": float(np.mean(action_flip_rates)),
        })
    return sorted(report, key=lambda row: float(row["probability_shift"]), reverse=True)
