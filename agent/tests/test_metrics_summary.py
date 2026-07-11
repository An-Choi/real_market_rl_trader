from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import build_daily_frame, summarize_backtest


def _results(
    exec_ts: list[str],
    val_ts: list[str],
    values: list[float],
    units: list[int],
    actions: list[int] | None = None,
    trade_values: list[float] | None = None,
    val_prices: list[float] | None = None,
) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(exec_ts),
        "valuation_timestamp": pd.to_datetime(val_ts),
        "portfolio_value": values,
        "units_held": units,
        "action": actions or [0] * n,
        "trade_value": trade_values or [0.0] * n,
        "valuation_price": val_prices or [100.0] * n,
    })


def _two_day_results() -> pd.DataFrame:
    """day1: 10000→10100, boundary(익일 첫 bar 평가): →10300, day2: →10200."""
    return _results(
        exec_ts=["2025-06-02 09:00", "2025-06-02 09:05", "2025-06-03 09:00"],
        val_ts=["2025-06-02 09:05", "2025-06-03 09:00", "2025-06-03 09:05"],
        values=[10100.0, 10300.0, 10200.0],
        units=[1, 1, 1],
    )


def test_daily_close_equity_and_boundary_attribution() -> None:
    daily = build_daily_frame(_two_day_results(), initial_value=10000.0)
    # E_{d1}=10100 (경계 step은 valuation date=06-03 → 익일 귀속)
    assert daily["close_equity"].tolist() == [10100.0, 10200.0]
    assert daily["daily_return"].iloc[0] == pytest.approx(0.01)
    assert daily["daily_return"].iloc[1] == pytest.approx(10200.0 / 10100.0 - 1)


def test_transition_intraday_settlement_decomposition() -> None:
    daily = build_daily_frame(
        _two_day_results(), initial_value=10000.0, terminal_liquidation_cost=100.0
    )
    # day1: transition 0, intraday = 10100/10000-1, settlement 0
    assert daily["r_transition"].iloc[0] == 0.0
    assert daily["r_intraday"].iloc[0] == pytest.approx(0.01)
    assert daily["r_settlement"].iloc[0] == 0.0
    # day2: B_d=10300(경계), E_raw=10200, settled=10100
    assert daily["r_transition"].iloc[1] == pytest.approx(10300.0 / 10100.0 - 1)
    assert daily["r_intraday"].iloc[1] == pytest.approx(10200.0 / 10300.0 - 1)
    assert daily["r_settlement"].iloc[1] == pytest.approx(10100.0 / 10200.0 - 1)
    # 곱셈 항등식: (1+trans)(1+intra)(1+settle) == 1+r_d
    for _, row in daily.iterrows():
        product = (
            (1 + row["r_transition"]) * (1 + row["r_intraday"]) * (1 + row["r_settlement"])
        )
        assert product == pytest.approx(1 + row["daily_return"])


def test_settlement_applies_to_all_metrics() -> None:
    results = _two_day_results()
    with_cost = summarize_backtest(
        results, initial_value=10000.0, terminal_liquidation_cost=100.0
    )
    without = summarize_backtest(results, initial_value=10000.0)
    # 마지막 r_D에 반영 → total/final은 물론 일별 시리즈 기반 지표도 달라진다
    assert with_cost["total_return"] == pytest.approx(10100.0 / 10000.0 - 1)
    assert with_cost["final_portfolio_value"] == pytest.approx(10100.0)
    assert with_cost["terminal_liquidation_cost"] == 100.0
    assert with_cost["sharpe_ratio"] != without["sharpe_ratio"]
    # 일별 복리 == total_return
    daily = build_daily_frame(
        results, initial_value=10000.0, terminal_liquidation_cost=100.0
    )
    compounded = float((1 + daily["daily_return"]).prod() - 1)
    assert with_cost["total_return"] == pytest.approx(compounded)


def test_settlement_flips_win_rate_and_hits_mdd() -> None:
    # day2 원수익 +10 → 정산 100으로 음수 반전
    results = _results(
        exec_ts=["2025-06-02 09:00", "2025-06-03 09:00"],
        val_ts=["2025-06-02 09:05", "2025-06-03 09:05"],
        values=[10100.0, 10110.0],
        units=[1, 1],
    )
    flipped = summarize_backtest(
        results, initial_value=10000.0, terminal_liquidation_cost=100.0
    )
    clean = summarize_backtest(results, initial_value=10000.0)
    assert clean["win_rate"] == 1.0
    assert flipped["win_rate"] == 0.5
    assert clean["max_drawdown"] == 0.0
    assert flipped["max_drawdown"] == pytest.approx(10010.0 / 10100.0 - 1)


def test_overnight_hold_rate_and_open_at_end() -> None:
    results = _two_day_results()
    metrics = summarize_backtest(results, initial_value=10000.0)
    assert metrics["overnight_hold_rate"] == 1.0  # 경계 1개, units>0
    assert metrics["open_at_end"] == 1.0

    flat = _results(
        exec_ts=["2025-06-02 09:00", "2025-06-02 09:05"],
        val_ts=["2025-06-02 09:05", "2025-06-02 09:10"],
        values=[10000.0, 10000.0],
        units=[0, 0],
    )
    flat_metrics = summarize_backtest(flat, initial_value=10000.0)
    assert flat_metrics["overnight_hold_rate"] == 0.0  # 경계 없음 → 0.0
    assert flat_metrics["open_at_end"] == 0.0


def test_market_return_uses_bar0_close() -> None:
    results = _results(
        exec_ts=["2025-06-02 09:00", "2025-06-02 09:05"],
        val_ts=["2025-06-02 09:05", "2025-06-02 09:10"],
        values=[10000.0, 10000.0],
        units=[0, 0],
        val_prices=[101.0, 103.0],
    )
    metrics = summarize_backtest(
        results, initial_value=10000.0, initial_market_price=100.0
    )
    assert metrics["market_return"] == pytest.approx(0.03)  # 103/100 − 1 (bar0 기준)


def test_partial_last_day_uses_same_formula() -> None:
    # max_steps로 날짜 중간에서 끊겨 마지막 날 행이 1개(경계 step)뿐이어도 동일 산식
    results = _results(
        exec_ts=["2025-06-02 09:00", "2025-06-02 09:05", "2025-06-02 09:10"],
        val_ts=["2025-06-02 09:05", "2025-06-02 09:10", "2025-06-03 09:00"],
        values=[10100.0, 10150.0, 10250.0],
        units=[1, 1, 1],
    )
    daily = build_daily_frame(
        results, initial_value=10000.0, terminal_liquidation_cost=50.0
    )
    assert daily["close_equity"].tolist() == [10150.0, 10200.0]
    assert daily["daily_return"].iloc[1] == pytest.approx(10200.0 / 10150.0 - 1)
    assert daily["r_transition"].iloc[1] == pytest.approx(10250.0 / 10150.0 - 1)
    assert daily["r_intraday"].iloc[1] == 0.0  # 부분일: 장중 성분 없음
    assert daily["r_settlement"].iloc[1] == pytest.approx(10200.0 / 10250.0 - 1)


def test_empty_results() -> None:
    metrics = summarize_backtest(pd.DataFrame(), initial_value=10000.0)
    assert metrics["total_return"] == 0.0
    assert "forced_clear_count" not in metrics
