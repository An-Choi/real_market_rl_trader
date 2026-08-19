"""Leakage-safe, friction-aware mean-reversion research baseline.

The baseline deliberately stays simple: robust feature scaling and entry
thresholds are fitted separately for every symbol on training data, one
hyper-parameter pair is selected on validation data, and test data is only
evaluated after that selection is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from env.observation import liquidity_score_from_adv
from friction.friction_model import FrictionModel


SIGNAL_FEATURES: tuple[str, ...] = ("log_ret_12", "vwap_dev", "log_ret_1")
SIGNAL_WEIGHTS: tuple[float, ...] = (0.50, 0.35, 0.15)
DEFAULT_ENTRY_QUANTILES: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
DEFAULT_HOLD_BARS: tuple[int, ...] = (3, 6, 12)


@dataclass(frozen=True, order=True)
class CandidateSpec:
    """One validation-selectable policy configuration."""

    entry_quantile: float
    hold_bars: int

    @property
    def key(self) -> str:
        return f"q={self.entry_quantile:g},hold={self.hold_bars}"

    def to_dict(self) -> dict[str, float | int]:
        return {
            "entry_quantile": float(self.entry_quantile),
            "hold_bars": int(self.hold_bars),
        }


@dataclass(frozen=True)
class CandidateCalibration:
    """Train-only values used by one candidate."""

    threshold: float
    expected_gross_return: float
    training_events: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "threshold": float(self.threshold),
            "expected_gross_return": float(self.expected_gross_return),
            "training_events": int(self.training_events),
        }


@dataclass(frozen=True)
class SymbolCalibration:
    """All train-only statistics for one symbol."""

    symbol: str
    centers: Mapping[str, float]
    scales: Mapping[str, float]
    candidates: Mapping[CandidateSpec, CandidateCalibration]
    training_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "training_rows": int(self.training_rows),
            "centers": {key: float(value) for key, value in self.centers.items()},
            "scales": {key: float(value) for key, value in self.scales.items()},
            "candidates": {
                spec.key: {**spec.to_dict(), **calibration.to_dict()}
                for spec, calibration in self.candidates.items()
            },
        }


@dataclass(frozen=True)
class ConfidenceGate:
    """Conservative validation qualification settings."""

    minimum_events: int = 8
    minimum_days: int = 5
    minimum_blocks: int = 3
    block_days: int = 5
    confidence: float = 0.95
    bootstrap_samples: int = 2_000
    minimum_daily_return_lcb: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("minimum_events", "minimum_days", "minimum_blocks", "block_days"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be between 0.5 and 1.0")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_events": self.minimum_events,
            "minimum_days": self.minimum_days,
            "minimum_blocks": self.minimum_blocks,
            "block_days": self.block_days,
            "confidence": self.confidence,
            "bootstrap_samples": self.bootstrap_samples,
            "minimum_daily_return_lcb": self.minimum_daily_return_lcb,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ValidationSelection:
    """A candidate chosen solely from validation metrics."""

    selected: CandidateSpec
    gate_passed: bool
    candidate_metrics: Mapping[CandidateSpec, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "gate_passed": bool(self.gate_passed),
            "selection_source": "validation_only",
            "candidates": {
                spec.key: {"candidate": spec.to_dict(), "metrics": dict(metrics)}
                for spec, metrics in self.candidate_metrics.items()
            },
        }


def candidate_grid(
    entry_quantiles: Sequence[float] = DEFAULT_ENTRY_QUANTILES,
    hold_bars: Sequence[int] = DEFAULT_HOLD_BARS,
) -> tuple[CandidateSpec, ...]:
    """Return the deterministic candidate grid used for validation selection."""
    if not entry_quantiles or not hold_bars:
        raise ValueError("candidate grid must not be empty")
    quantiles = tuple(float(value) for value in entry_quantiles)
    holds = tuple(int(value) for value in hold_bars)
    if any(not 0.0 < value < 0.5 for value in quantiles):
        raise ValueError("entry quantiles must be between 0 and 0.5")
    if any(value < 1 for value in holds):
        raise ValueError("hold bars must be positive")
    if len(set(quantiles)) != len(quantiles) or len(set(holds)) != len(holds):
        raise ValueError("candidate values must be unique")
    return tuple(CandidateSpec(q, hold) for q in quantiles for hold in holds)


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"Timestamp", "Close", "Adv20", *SIGNAL_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing baseline columns: {missing}")
    if frame.empty:
        raise ValueError("baseline frame must not be empty")


def _robust_location_scale(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"feature {series.name!r} has no finite training values")
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def mean_reversion_score(
    frame: pd.DataFrame,
    *,
    centers: Mapping[str, float],
    scales: Mapping[str, float],
) -> pd.Series:
    """Compute a long-only oversold score using frozen train statistics."""
    score = np.zeros(len(frame), dtype=float)
    valid = np.ones(len(frame), dtype=bool)
    for feature, weight in zip(SIGNAL_FEATURES, SIGNAL_WEIGHTS):
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        valid &= np.isfinite(values)
        score -= weight * (values - centers[feature]) / scales[feature]
    score[~valid] = np.nan
    return pd.Series(score, index=frame.index, name="mean_reversion_score")


def _price(row: pd.Series) -> float:
    value = row.get("ExecPrice", row.get("Close"))
    if value is None or not np.isfinite(float(value)) or float(value) <= 0:
        value = row.get("Close")
    if value is None or not np.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("row has no positive execution price")
    return float(value)


def _adv(row: pd.Series) -> float | None:
    value = row.get("Adv20")
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _same_day(frame: pd.DataFrame, start: int, end: int) -> bool:
    return pd.Timestamp(frame.iloc[start]["Timestamp"]).date() == pd.Timestamp(
        frame.iloc[end]["Timestamp"]
    ).date()


def _raw_training_events(
    frame: pd.DataFrame,
    scores: pd.Series,
    *,
    threshold: float,
    hold_bars: int,
) -> list[float]:
    returns: list[float] = []
    index = 0
    while index + hold_bars < len(frame):
        exit_index = index + hold_bars
        score = float(scores.iloc[index])
        if (
            np.isfinite(score)
            and score >= threshold
            and _same_day(frame, index, exit_index)
        ):
            entry_price = _price(frame.iloc[index])
            exit_price = _price(frame.iloc[exit_index])
            returns.append(exit_price / entry_price - 1.0)
            index = exit_index + 1
        else:
            index += 1
    return returns


def fit_symbol_calibration(
    frame: pd.DataFrame,
    *,
    symbol: str,
    specs: Iterable[CandidateSpec] | None = None,
) -> SymbolCalibration:
    """Fit scales, tail thresholds, and expected rebound on training data only."""
    _validate_frame(frame)
    chosen_specs = tuple(specs or candidate_grid())
    if not chosen_specs:
        raise ValueError("specs must not be empty")
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in SIGNAL_FEATURES:
        centers[feature], scales[feature] = _robust_location_scale(frame[feature])
    scores = mean_reversion_score(frame, centers=centers, scales=scales)
    finite_scores = scores[np.isfinite(scores)].to_numpy(dtype=float)
    if finite_scores.size == 0:
        raise ValueError(f"symbol {symbol} has no finite training scores")

    thresholds = {
        quantile: float(np.quantile(finite_scores, 1.0 - quantile))
        for quantile in {spec.entry_quantile for spec in chosen_specs}
    }
    calibrations: dict[CandidateSpec, CandidateCalibration] = {}
    for spec in chosen_specs:
        training_returns = _raw_training_events(
            frame,
            scores,
            threshold=thresholds[spec.entry_quantile],
            hold_bars=spec.hold_bars,
        )
        # The median is deliberately used instead of the best-tail mean: it is
        # less sensitive to one lucky rebound in a small training sample.
        expected = float(np.median(training_returns)) if training_returns else 0.0
        calibrations[spec] = CandidateCalibration(
            threshold=thresholds[spec.entry_quantile],
            expected_gross_return=expected,
            training_events=len(training_returns),
        )
    return SymbolCalibration(
        symbol=str(symbol),
        centers=centers,
        scales=scales,
        candidates=calibrations,
        training_rows=len(frame),
    )


def round_trip_cost(
    entry_row: pd.Series,
    exit_row: pd.Series,
    *,
    friction_model: FrictionModel,
    order_notional: float,
) -> dict[str, float | int] | None:
    """Calculate share-rounded buy and sell friction from price, date, and ADV."""
    if not np.isfinite(order_notional) or order_notional <= 0:
        raise ValueError("order_notional must be positive and finite")
    entry_price = _price(entry_row)
    exit_price = _price(exit_row)
    shares = int(order_notional // entry_price)
    if shares < 1:
        return None
    entry_value = float(shares * entry_price)
    exit_value = float(shares * exit_price)
    entry_date = pd.Timestamp(entry_row["Timestamp"]).date()
    exit_date = pd.Timestamp(exit_row["Timestamp"]).date()
    entry_liquidity = liquidity_score_from_adv(_adv(entry_row), order_notional)
    exit_liquidity = liquidity_score_from_adv(_adv(exit_row), order_notional)
    buy_cost = friction_model.calculate_total_friction(
        trade_value=entry_value,
        side="buy",
        liquidity_score=entry_liquidity,
        price=entry_price,
        trade_date=entry_date,
    )
    sell_cost = friction_model.calculate_total_friction(
        trade_value=exit_value,
        side="sell",
        liquidity_score=exit_liquidity,
        price=exit_price,
        trade_date=exit_date,
    )
    paid = entry_value + buy_cost
    received = exit_value - sell_cost
    return {
        "shares": shares,
        "entry_value": entry_value,
        "exit_value": exit_value,
        "buy_cost": float(buy_cost),
        "sell_cost": float(sell_cost),
        "gross_return": float(exit_value / entry_value - 1.0),
        "net_return": float(received / paid - 1.0),
        "round_trip_cost_rate": float((buy_cost + sell_cost) / paid),
    }


def estimated_round_trip_cost_rate(
    row: pd.Series,
    *,
    friction_model: FrictionModel,
    order_notional: float,
) -> float:
    """Estimate entry-time round-trip cost without seeing a future exit row."""
    estimate = round_trip_cost(
        row, row, friction_model=friction_model, order_notional=order_notional
    )
    return float("inf") if estimate is None else float(estimate["round_trip_cost_rate"])


def simulate_candidate(
    frame: pd.DataFrame,
    *,
    calibration: SymbolCalibration,
    spec: CandidateSpec,
    friction_model: FrictionModel,
    order_notional: float,
) -> list[dict[str, Any]]:
    """Simulate non-overlapping, intraday entries with a train-frozen policy."""
    _validate_frame(frame)
    if spec not in calibration.candidates:
        raise ValueError(f"candidate {spec.key} is absent from calibration")
    fitted = calibration.candidates[spec]
    scores = mean_reversion_score(
        frame, centers=calibration.centers, scales=calibration.scales
    )
    trades: list[dict[str, Any]] = []
    index = 0
    while index + spec.hold_bars < len(frame):
        exit_index = index + spec.hold_bars
        score = float(scores.iloc[index])
        if not (
            np.isfinite(score)
            and score >= fitted.threshold
            and _same_day(frame, index, exit_index)
        ):
            index += 1
            continue
        entry_row = frame.iloc[index]
        estimated_cost = estimated_round_trip_cost_rate(
            entry_row,
            friction_model=friction_model,
            order_notional=order_notional,
        )
        # This expected edge is train-only. A validation/test future return is
        # never used to decide whether to enter.
        if fitted.expected_gross_return <= estimated_cost:
            index += 1
            continue
        exit_row = frame.iloc[exit_index]
        costs = round_trip_cost(
            entry_row,
            exit_row,
            friction_model=friction_model,
            order_notional=order_notional,
        )
        if costs is None:
            index += 1
            continue
        trades.append({
            "symbol": calibration.symbol,
            "entry_index": int(index),
            "exit_index": int(exit_index),
            "entry_timestamp": pd.Timestamp(entry_row["Timestamp"]).isoformat(),
            "exit_timestamp": pd.Timestamp(exit_row["Timestamp"]).isoformat(),
            "entry_date": pd.Timestamp(entry_row["Timestamp"]).date().isoformat(),
            "score": score,
            "threshold": float(fitted.threshold),
            "expected_gross_return_train": float(fitted.expected_gross_return),
            "estimated_round_trip_cost_rate": estimated_cost,
            **costs,
        })
        index = exit_index + 1
    return trades


def _daily_returns(trades: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not trades:
        return np.asarray([], dtype=float)
    table = pd.DataFrame({
        "date": [trade["entry_date"] for trade in trades],
        "net_return": [float(trade["net_return"]) for trade in trades],
    })
    # Equal-weighting events per day prevents a busy symbol/day from dominating
    # validation selection simply because it generated more entries.
    return table.groupby("date", sort=True)["net_return"].mean().to_numpy(dtype=float)


def _block_bootstrap_lcb(
    daily_returns: np.ndarray,
    *,
    gate: ConfidenceGate,
) -> tuple[float | None, int]:
    blocks = [
        daily_returns[start : start + gate.block_days]
        for start in range(0, len(daily_returns), gate.block_days)
    ]
    blocks = [block for block in blocks if len(block) == gate.block_days]
    if len(blocks) < gate.minimum_blocks:
        return None, len(blocks)
    # A block statistic is a daily geometric mean, preserving compounding while
    # keeping unequal partial tails out of the confidence estimate.
    block_daily = np.asarray([
        float(np.prod(1.0 + block) ** (1.0 / len(block)) - 1.0)
        for block in blocks
    ])
    rng = np.random.default_rng(gate.seed)
    draws = rng.integers(0, len(block_daily), size=(gate.bootstrap_samples, len(block_daily)))
    bootstrap_means = block_daily[draws].mean(axis=1)
    alpha = 1.0 - gate.confidence
    return float(np.quantile(bootstrap_means, alpha)), len(blocks)


def summarize_trades(
    trades: Sequence[Mapping[str, Any]],
    *,
    gate: ConfidenceGate,
) -> dict[str, Any]:
    """Summarize events and apply the minimum-sample/block-confidence gate."""
    net = np.asarray([float(trade["net_return"]) for trade in trades], dtype=float)
    gross = np.asarray([float(trade["gross_return"]) for trade in trades], dtype=float)
    friction = np.asarray(
        [float(trade["round_trip_cost_rate"]) for trade in trades], dtype=float
    )
    daily = _daily_returns(trades)
    lcb, block_count = _block_bootstrap_lcb(daily, gate=gate)
    reasons: list[str] = []
    if len(net) < gate.minimum_events:
        reasons.append("insufficient_events")
    if len(daily) < gate.minimum_days:
        reasons.append("insufficient_days")
    if block_count < gate.minimum_blocks:
        reasons.append("insufficient_complete_blocks")
    if lcb is None or lcb <= gate.minimum_daily_return_lcb:
        reasons.append("daily_return_lcb_not_positive")
    passed = not reasons
    return {
        "event_count": int(len(net)),
        "trading_day_count": int(len(daily)),
        "complete_block_count": int(block_count),
        "net_total_return": float(np.prod(1.0 + daily) - 1.0) if len(daily) else 0.0,
        "gross_compounded_event_return": (
            float(np.prod(1.0 + gross) - 1.0) if len(gross) else 0.0
        ),
        "mean_event_net_return": float(np.mean(net)) if len(net) else 0.0,
        "median_event_net_return": float(np.median(net)) if len(net) else 0.0,
        "win_rate": float(np.mean(net > 0.0)) if len(net) else 0.0,
        "mean_round_trip_cost_rate": float(np.mean(friction)) if len(friction) else 0.0,
        "daily_mean_net_return": float(np.mean(daily)) if len(daily) else 0.0,
        "daily_return_lcb": lcb,
        "gate_passed": passed,
        "gate_failure_reasons": reasons,
    }


def evaluate_candidate(
    data_by_symbol: Mapping[str, pd.DataFrame],
    *,
    calibrations: Mapping[str, SymbolCalibration],
    spec: CandidateSpec,
    friction_model: FrictionModel,
    order_notional: float,
    gate: ConfidenceGate,
) -> dict[str, Any]:
    """Evaluate one frozen candidate across symbols."""
    if set(data_by_symbol) != set(calibrations):
        raise ValueError("data and calibration symbol sets must match")
    all_trades: list[dict[str, Any]] = []
    per_symbol: dict[str, Any] = {}
    for symbol, frame in data_by_symbol.items():
        trades = simulate_candidate(
            frame,
            calibration=calibrations[symbol],
            spec=spec,
            friction_model=friction_model,
            order_notional=order_notional,
        )
        all_trades.extend(trades)
        per_symbol[symbol] = {
            "metrics": summarize_trades(trades, gate=gate),
            "trades": trades,
        }
    return {
        "candidate": spec.to_dict(),
        "metrics": summarize_trades(all_trades, gate=gate),
        "per_symbol": per_symbol,
    }


def select_on_validation(
    validation_data: Mapping[str, pd.DataFrame],
    *,
    calibrations: Mapping[str, SymbolCalibration],
    specs: Sequence[CandidateSpec],
    friction_model: FrictionModel,
    order_notional: float,
    gate: ConfidenceGate,
) -> ValidationSelection:
    """Select a candidate without consulting test data."""
    if not specs:
        raise ValueError("specs must not be empty")
    metrics: dict[CandidateSpec, Mapping[str, Any]] = {}
    for spec in specs:
        result = evaluate_candidate(
            validation_data,
            calibrations=calibrations,
            spec=spec,
            friction_model=friction_model,
            order_notional=order_notional,
            gate=gate,
        )
        metrics[spec] = result["metrics"]

    qualified = [spec for spec in specs if bool(metrics[spec]["gate_passed"])]
    pool = qualified or list(specs)

    def rank(spec: CandidateSpec) -> tuple[float, float, float, int, float, int]:
        item = metrics[spec]
        lcb = item["daily_return_lcb"]
        return (
            float(lcb) if lcb is not None else float("-inf"),
            float(item["daily_mean_net_return"]),
            float(item["net_total_return"]),
            int(item["event_count"]),
            -spec.entry_quantile,
            -spec.hold_bars,
        )

    selected = max(pool, key=rank)
    return ValidationSelection(
        selected=selected,
        gate_passed=bool(metrics[selected]["gate_passed"]),
        candidate_metrics=metrics,
    )


def run_leakage_safe_fold(
    *,
    train_data: Mapping[str, pd.DataFrame],
    validation_data: Mapping[str, pd.DataFrame],
    test_data: Mapping[str, pd.DataFrame],
    friction_model: FrictionModel,
    order_notional: float,
    gate: ConfidenceGate | None = None,
    specs: Sequence[CandidateSpec] | None = None,
) -> dict[str, Any]:
    """Fit train, select validation, then audit test with a frozen selection."""
    chosen_specs = tuple(specs or candidate_grid())
    confidence_gate = gate or ConfidenceGate()
    symbol_sets = [set(part) for part in (train_data, validation_data, test_data)]
    if not symbol_sets[0] or symbol_sets[0] != symbol_sets[1] or symbol_sets[0] != symbol_sets[2]:
        raise ValueError("train, validation, and test symbol sets must be identical")
    calibrations = {
        symbol: fit_symbol_calibration(frame, symbol=symbol, specs=chosen_specs)
        for symbol, frame in train_data.items()
    }
    selection = select_on_validation(
        validation_data,
        calibrations=calibrations,
        specs=chosen_specs,
        friction_model=friction_model,
        order_notional=order_notional,
        gate=confidence_gate,
    )
    selected = selection.selected
    return {
        "calibrations": {
            symbol: calibration.to_dict()
            for symbol, calibration in calibrations.items()
        },
        "selection": selection.to_dict(),
        "train": evaluate_candidate(
            train_data,
            calibrations=calibrations,
            spec=selected,
            friction_model=friction_model,
            order_notional=order_notional,
            gate=confidence_gate,
        ),
        "validation": evaluate_candidate(
            validation_data,
            calibrations=calibrations,
            spec=selected,
            friction_model=friction_model,
            order_notional=order_notional,
            gate=confidence_gate,
        ),
        "test": evaluate_candidate(
            test_data,
            calibrations=calibrations,
            spec=selected,
            friction_model=friction_model,
            order_notional=order_notional,
            gate=confidence_gate,
        ),
        "test_is_selection_input": False,
        "gate": confidence_gate.to_dict(),
    }
