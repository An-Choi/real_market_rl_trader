"""체결 계약 v3: 관찰 봉이 아니라 ExecPrice(다음 1분봉 Open/동시호가)로 체결."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.trading_env import TradingEnvironment


def _data(closes: list[float], exec_prices: list[float]) -> pd.DataFrame:
    n = len(closes)
    ts = pd.date_range("2025-06-02 10:05", periods=n, freq="5min", tz="Asia/Seoul")
    return pd.DataFrame({
        "Timestamp": ts,
        "Close": np.array(closes, dtype=float),
        "ExecPrice": np.array(exec_prices, dtype=float),
        "ma_5": np.array(closes, dtype=float),
    })


def make_env(data: pd.DataFrame) -> TradingEnvironment:
    return TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
    )


def test_missing_exec_price_column_raises() -> None:
    bad = _data([100.0, 100.0], [100.0, 100.0]).drop(columns=["ExecPrice"])
    with pytest.raises(ValueError, match="ExecPrice"):
        make_env(bad)


def test_buy_fills_at_exec_price_not_observed_close() -> None:
    # 관찰 봉 Close 100, 다음 1분봉 Open(ExecPrice) 102 → 체결은 102.
    env = make_env(_data([100.0, 100.0, 100.0], [102.0, 102.0, 102.0]))
    env.reset(seed=0)
    env.step(1)  # Target 20%
    friction_rate = 0.002
    notional = 2_000.0 / (1.0 + 0.20 * friction_rate)
    assert env.shares_held == pytest.approx(notional / 102.0)
    assert env.cash == pytest.approx(10_000.0 - notional * (1.0 + friction_rate))


def test_clear_fills_at_exec_price() -> None:
    # 매수 후 Clear: 매도 대금 = 보유주 × 그 시점 ExecPrice(98), Close(100) 아님.
    env = make_env(_data([100.0] * 4, [100.0, 98.0, 98.0, 98.0]))
    env.reset(seed=0)
    env.step(1)
    cash_before = env.cash
    shares = env.shares_held
    env.step(0)                      # Target 0% — ExecPrice 98로 매도
    proceeds = shares * 98.0
    friction = proceeds * (0.001 + 0.0005 + 0.0005 + 0.002)  # fee+spread+slip+sell tax
    assert env.cash == pytest.approx(cash_before + proceeds - friction)


def test_nan_exec_price_falls_back_to_close() -> None:
    # 경매 print도 다음 1분봉도 없는 말단(NaN) → Close로 fallback.
    env = make_env(_data([100.0, 100.0, 100.0], [np.nan, np.nan, np.nan]))
    env.reset(seed=0)
    env.step(1)
    notional = 2_000.0 / (1.0 + 0.20 * 0.002)
    assert env.shares_held == pytest.approx(notional / 100.0)


def test_estimate_liquidation_cost_uses_exec_price() -> None:
    env = make_env(_data([100.0, 100.0, 100.0], [102.0, 102.0, 102.0]))
    env.reset(seed=0)
    env.step(1)
    shares = env.shares_held
    # 현재 시점(step 1) 청산 추정도 ExecPrice(102) 기준 매도 friction.
    expected = shares * 102.0 * (0.001 + 0.0005 + 0.0005 + 0.002)
    assert env.estimate_liquidation_cost() == pytest.approx(expected)


def test_observation_excludes_exec_price() -> None:
    # ExecPrice는 결정 시점 이후 미래 정보 — 관찰에 절대 포함 금지.
    env = make_env(_data([100.0, 100.0], [102.0, 102.0]))
    obs, _ = env.reset(seed=0)
    assert len(obs) == 1 + 4                                   # feature 1개 + 포트폴리오 4개
    assert 102.0 not in np.asarray(obs)
