from __future__ import annotations

import numpy as np
import pandas as pd

from friction.friction_model import FrictionModel
from policies.cost_aware import (
    CandidateCalibration,
    CandidateSpec,
    ConfidenceGate,
    SymbolCalibration,
    fit_symbol_calibration,
    round_trip_cost,
    run_leakage_safe_fold,
    simulate_candidate,
    summarize_trades,
)


def _frame(*, days: int = 6, bars: int = 30, rebound: float = 0.01) -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2026-01-05", periods=days, freq="B"):
        close = 100.0
        for bar in range(bars):
            shock = -0.03 if bar in (3, 15) else 0.0
            if bar in (4, 16):
                close *= 1.0 + rebound
            rows.append({
                "Timestamp": day + pd.Timedelta(hours=9, minutes=5 * bar),
                "Close": close,
                "ExecPrice": close,
                "Adv20": 1_000_000_000.0,
                "log_ret_12": shock,
                "vwap_dev": shock * 0.8,
                "log_ret_1": shock * 0.5,
            })
    return pd.DataFrame(rows)


def _cheap_friction() -> FrictionModel:
    return FrictionModel(
        fee_rate=0.0001,
        spread_rate=0.0001,
        slippage_rate=0.0,
        sell_tax_rate=0.0002,
    )


def test_fit_is_symbol_specific_and_uses_only_passed_training_frame() -> None:
    spec = CandidateSpec(0.05, 3)
    train_a = _frame()
    train_b = train_a.copy()
    train_b["log_ret_12"] *= 10.0

    calibration_a = fit_symbol_calibration(train_a, symbol="AAA", specs=[spec])
    calibration_b = fit_symbol_calibration(train_b, symbol="BBB", specs=[spec])

    assert calibration_a.symbol == "AAA"
    assert calibration_b.scales["log_ret_12"] != calibration_a.scales["log_ret_12"]
    assert set(calibration_a.centers) == {"log_ret_12", "vwap_dev", "log_ret_1"}
    assert calibration_a.candidates[spec].threshold > 0.0


def test_round_trip_cost_uses_integer_shares_and_reduces_return() -> None:
    frame = _frame(days=1)
    entry = frame.iloc[0].copy()
    exit_row = frame.iloc[1].copy()
    entry["Close"] = entry["ExecPrice"] = 100.0
    exit_row["Close"] = exit_row["ExecPrice"] = 101.0
    result = round_trip_cost(
        entry, exit_row, friction_model=_cheap_friction(), order_notional=1_050.0
    )

    assert result is not None
    assert result["shares"] == 10
    assert np.isclose(result["gross_return"], 0.01)
    assert result["net_return"] < result["gross_return"]
    assert result["buy_cost"] > 0.0
    assert result["sell_cost"] > result["buy_cost"]


def test_simulation_entries_are_non_overlapping() -> None:
    frame = _frame(days=1, bars=20)
    spec = CandidateSpec(0.05, 3)
    calibration = SymbolCalibration(
        symbol="AAA",
        centers={name: 0.0 for name in ("log_ret_12", "vwap_dev", "log_ret_1")},
        scales={name: 0.01 for name in ("log_ret_12", "vwap_dev", "log_ret_1")},
        candidates={
            spec: CandidateCalibration(
                threshold=1.0, expected_gross_return=0.02, training_events=20
            )
        },
        training_rows=100,
    )
    trades = simulate_candidate(
        frame,
        calibration=calibration,
        spec=spec,
        friction_model=_cheap_friction(),
        order_notional=10_000.0,
    )

    assert trades
    assert all(
        current["exit_index"] < following["entry_index"]
        for current, following in zip(trades, trades[1:])
    )
    assert all(
        pd.Timestamp(trade["entry_timestamp"]).date()
        == pd.Timestamp(trade["exit_timestamp"]).date()
        for trade in trades
    )


def test_confidence_gate_fails_closed_for_too_few_events() -> None:
    gate = ConfidenceGate(
        minimum_events=3,
        minimum_days=2,
        minimum_blocks=1,
        block_days=1,
        bootstrap_samples=100,
    )
    metrics = summarize_trades(
        [{
            "entry_date": "2026-01-05",
            "net_return": 0.01,
            "gross_return": 0.012,
            "round_trip_cost_rate": 0.002,
        }],
        gate=gate,
    )

    assert metrics["gate_passed"] is False
    assert "insufficient_events" in metrics["gate_failure_reasons"]
    assert "insufficient_days" in metrics["gate_failure_reasons"]


def test_test_prices_cannot_change_validation_selection() -> None:
    spec_short = CandidateSpec(0.05, 1)
    spec_long = CandidateSpec(0.05, 3)
    train = {"AAA": _frame(days=8, rebound=0.02)}
    validation = {"AAA": _frame(days=6, rebound=0.015)}
    test_up = {"AAA": _frame(days=4, rebound=0.05)}
    test_down = {"AAA": _frame(days=4, rebound=-0.05)}
    gate = ConfidenceGate(
        minimum_events=1,
        minimum_days=1,
        minimum_blocks=1,
        block_days=1,
        bootstrap_samples=100,
    )
    common = dict(
        train_data=train,
        validation_data=validation,
        friction_model=_cheap_friction(),
        order_notional=10_000.0,
        gate=gate,
        specs=[spec_short, spec_long],
    )

    first = run_leakage_safe_fold(test_data=test_up, **common)
    second = run_leakage_safe_fold(test_data=test_down, **common)

    assert first["selection"] == second["selection"]
    assert first["test_is_selection_input"] is False
    assert first["test"]["metrics"] != second["test"]["metrics"]
