"""체결 계약 v3: 관찰 봉이 아니라 ExecPrice(다음 1분봉 Open/동시호가)로 체결."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.trading_env import TradingEnvironment
from friction.friction_model import FrictionModel


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
    env.step(1)  # Add 1 Unit = 2,000 고정 notional
    assert env.shares_held == 19  # whole-share fill; Close(100) would buy 20
    trade_value = 19 * 102.0
    assert env.cash == pytest.approx(10_000.0 - trade_value - trade_value * 0.002)


def test_clear_fills_at_exec_price() -> None:
    # 매수 후 Clear: 매도 대금 = 보유주 × 그 시점 ExecPrice(98), Close(100) 아님.
    env = make_env(_data([100.0] * 4, [100.0, 98.0, 98.0, 98.0]))
    env.reset(seed=0)
    env.step(1)                      # 100에 20주 매수
    cash_before = env.cash
    shares = env.shares_held
    env.step(3)                      # Clear at ExecPrice 98
    proceeds = shares * 98.0
    friction = proceeds * (0.001 + 0.0005 + 0.0005 + 0.002)  # fee+spread+slip+sell tax
    assert env.cash == pytest.approx(cash_before + proceeds - friction)


def test_nan_exec_price_falls_back_to_close() -> None:
    # 경매 print도 다음 1분봉도 없는 말단(NaN) → Close로 fallback.
    env = make_env(_data([100.0, 100.0, 100.0], [np.nan, np.nan, np.nan]))
    env.reset(seed=0)
    env.step(1)
    assert env.shares_held == pytest.approx(2_000.0 / 100.0)


def test_terminal_settlement_includes_exec_price_gap_and_friction() -> None:
    env = make_env(_data([100.0, 100.0, 100.0], [102.0, 102.0, 102.0]))
    env.reset(seed=0)
    env.step(1)                      # 102에 19주 매수
    shares = env.shares_held
    execution_value = shares * 102.0
    marked_value = shares * 100.0
    friction = execution_value * (0.001 + 0.0005 + 0.0005 + 0.002)
    expected = marked_value - execution_value + friction
    assert env.estimate_terminal_settlement_adjustment() == pytest.approx(expected)
    assert env.estimate_liquidation_cost() == pytest.approx(expected)


def test_zero_share_add_is_a_complete_no_op() -> None:
    env = make_env(_data([300_000.0] * 3, [300_000.0] * 3))
    env.reset(seed=0)
    before = (env.cash, env.units_held, env.shares_held, env.cost_basis)
    assert env.action_masks().tolist() == [True, False, False, False]
    _, _, _, _, info = env.step(1)
    after = (env.cash, env.units_held, env.shares_held, env.cost_basis)
    assert after == before
    assert info["trade_value"] == 0.0
    assert env._share_lots == []
    assert env._lot_costs == []


def test_sell_liquidity_uses_actual_reduce_or_clear_order_size() -> None:
    data = _data([100.0] * 5, [100.0] * 5)
    data["Adv20"] = 1_000_000.0
    friction = FrictionModel(
        fee_rate=0.0,
        spread_rate=0.0,
        slippage_rate=0.001,
        execution_uncertainty_rate=0.0,
        sell_tax_rate=0.0,
    )

    reduce_env = TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.2,
        max_units=5,
        friction_model=friction,
    )
    reduce_env.reset(seed=0)
    reduce_env.step(1)
    reduce_env.step(1)
    _, _, _, _, reduce_info = reduce_env.step(2)

    clear_env = TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.2,
        max_units=5,
        friction_model=friction,
    )
    clear_env.reset(seed=0)
    clear_env.step(1)
    clear_env.step(1)
    _, _, _, _, clear_info = clear_env.step(3)

    # 2,000/ADV => score .5 => friction 4; 4,000/ADV => score .25 => 16.
    assert reduce_info["trade_value"] == -2_000.0
    assert reduce_info["friction_cost"] == pytest.approx(4.0)
    assert clear_info["trade_value"] == -4_000.0
    assert clear_info["friction_cost"] == pytest.approx(16.0)


def test_observation_excludes_exec_price() -> None:
    # ExecPrice는 결정 시점 이후 미래 정보 — 관찰에 절대 포함 금지.
    env = make_env(_data([100.0, 100.0], [102.0, 102.0]))
    obs, _ = env.reset(seed=0)
    assert len(obs) == 1 + 5                                   # feature 1개 + 포트폴리오 5개
    assert 102.0 not in np.asarray(obs)


def test_observation_liquidity_reads_adv20_column() -> None:
    # unit 2,000 / Adv20 2,000,000 = 1e-3 → (log10(1e-3)+4)/4 = 0.25
    data = _data([100.0, 100.0], [102.0, 102.0])
    data["Adv20"] = 2_000_000.0
    env = make_env(data)
    obs, _ = env.reset(seed=0)
    assert obs[-1] == np.float32(0.25)


def test_observation_liquidity_defaults_zero_without_adv20() -> None:
    env = make_env(_data([100.0, 100.0], [102.0, 102.0]))
    obs, _ = env.reset(seed=0)
    assert obs[-1] == np.float32(0.0)
