from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.trading_env import TradingEnvironment


@pytest.fixture
def flat_data() -> pd.DataFrame:
    # 가격 100 고정, feature 1개. Timestamp는 Task 3에서 쓰므로 단일일치로만.
    n = 10
    ts = pd.date_range("2025-06-02 09:00", periods=n, freq="1min", tz="Asia/Seoul")
    return pd.DataFrame(
        {
            "Timestamp": ts,
            "Close": np.full(n, 100.0),
            "ma_5": np.full(n, 100.0),
        }
    )


def make_env(data: pd.DataFrame) -> TradingEnvironment:
    return TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
    )


def test_add_increments_units(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)  # Add 1 Unit
    assert info["units_held"] == 1


def test_add_caps_at_max_units(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    for _ in range(7):  # 5번이면 max, 이후 no-op
        _, _, _, _, info = env.step(1)
    assert info["units_held"] == 5


def test_clear_zeroes_units(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    env.step(1)
    env.step(1)
    _, _, _, _, info = env.step(2)  # Clear
    assert info["units_held"] == 0


def test_long_only_no_negative_units(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(2)  # Clear with no position → no-op
    assert info["units_held"] == 0


def test_invalid_action_raises(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(3)


def test_friction_charged_on_add(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)  # Add → buys 1 Unit (2000 notional)
    assert info["friction_cost"] > 0


def test_clear_friction_exceeds_add_friction(flat_data: pd.DataFrame) -> None:
    """매도(Clear) step 비용이 매수(Add) step 비용보다 거래세만큼 크다."""
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, add_info = env.step(1)   # Add 1 Unit (buy)
    _, _, _, _, clear_info = env.step(2)  # Clear (sell)
    assert clear_info["friction_cost"] > add_info["friction_cost"]
