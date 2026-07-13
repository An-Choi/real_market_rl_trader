from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.trading_env import TradingEnvironment
from friction.friction_model import FrictionModel


def _data(prices: list[float]) -> pd.DataFrame:
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    return pd.DataFrame({"Timestamp": ts, "Close": prices, "feature": np.zeros(len(prices))})


def _zero_friction() -> FrictionModel:
    return FrictionModel(
        fee_rate=0.0,
        spread_rate=0.0,
        slippage_rate=0.0,
        execution_uncertainty_rate=0.0,
        sell_tax_rate=0.0,
    )


def test_turnover_penalty_is_optional_reward_term() -> None:
    env = TradingEnvironment(
        market_data=_data([100.0, 100.0, 100.0]),
        feature_columns=["feature"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        friction_model=_zero_friction(),
        turnover_penalty_rate=0.5,
    )

    env.reset(seed=0)
    _, reward, _, _, info = env.step(1)

    assert info["portfolio_return"] == pytest.approx(0.0)
    assert info["turnover_penalty"] == pytest.approx(0.1)
    assert reward == pytest.approx(-0.1)


def test_drawdown_penalty_uses_running_peak() -> None:
    env = TradingEnvironment(
        market_data=_data([100.0, 90.0, 90.0]),
        feature_columns=["feature"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        friction_model=_zero_friction(),
        drawdown_penalty_rate=2.0,
    )

    env.reset(seed=0)
    _, reward, _, _, info = env.step(1)

    assert info["portfolio_return"] == pytest.approx(-0.02)
    assert info["drawdown_penalty"] == pytest.approx(0.04)
    assert reward == pytest.approx(-0.06)


def test_log_return_reward_is_scaled_and_decomposed() -> None:
    env = TradingEnvironment(
        market_data=_data([100.0, 110.0, 110.0]),
        feature_columns=["feature"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        friction_model=_zero_friction(),
        reward_return_mode="log_return",
        reward_scale=100.0,
        benchmark_relative_rate=0.05,
    )

    env.reset(seed=0)
    _, reward, _, _, info = env.step(1)

    expected_base = np.log(10_200.0 / 10_000.0)
    expected_benchmark = np.log(110.0 / 100.0)
    expected_relative = (expected_base - expected_benchmark) * 0.05
    assert info["base_return"] == pytest.approx(expected_base)
    assert info["benchmark_return"] == pytest.approx(expected_benchmark)
    assert info["benchmark_simple_return"] == pytest.approx(0.1)
    assert info["execution_date"] == "2025-06-02"
    assert info["valuation_date"] == "2025-06-02"
    assert info["benchmark_relative_reward"] == pytest.approx(expected_relative)
    assert reward == pytest.approx((expected_base + expected_relative) * 100.0)
