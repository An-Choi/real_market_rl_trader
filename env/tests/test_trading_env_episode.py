from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.trading_env import TradingEnvironment


@pytest.fixture
def two_day_data() -> pd.DataFrame:
    # 2거래일 × 5봉, 가격 100→104 상승.
    frames = []
    for day in ["2025-06-02", "2025-06-03"]:
        ts = pd.date_range(f"{day} 09:00", periods=5, freq="1min", tz="Asia/Seoul")
        frames.append(
            pd.DataFrame(
                {
                    "Timestamp": ts,
                    "Close": np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
                    "ma_5": np.full(5, 100.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_env(data: pd.DataFrame) -> TradingEnvironment:
    return TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
    )


def test_reset_with_date_uses_only_that_day(two_day_data: pd.DataFrame) -> None:
    env = make_env(two_day_data)
    env.reset(seed=0, options={"date": "2025-06-02"})
    steps = 0
    truncated = False
    while not truncated:
        _, _, _, truncated, _ = env.step(0)  # Hold
        steps += 1
    assert steps == 4  # 5봉 → 4 step (마지막 bar는 valuation 전용)


def test_episode_does_not_cross_into_next_day(two_day_data: pd.DataFrame) -> None:
    env = make_env(two_day_data)
    env.reset(seed=0, options={"date": "2025-06-02"})
    prices = []
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(0)
        prices.append(info["portfolio_value"])
    # episode_days=1이면 다음날(06-03) 데이터에 접근하지 않는다
    assert len(prices) == 4


def test_no_settlement_or_forced_trade_at_truncation(two_day_data: pd.DataFrame) -> None:
    env = make_env(two_day_data)
    env.reset(seed=0, options={"date": "2025-06-02"})
    env.step(1)  # Add 1 Unit
    truncated = False
    info = {}
    while not truncated:
        _, _, _, truncated, info = env.step(0)  # Hold until end
    assert info["units_held"] == 1          # 강제청산 없음
    assert "forced_clear" not in info


def test_reset_without_date_is_deterministic(two_day_data: pd.DataFrame) -> None:
    env = make_env(two_day_data)
    env.reset(seed=42)
    d1 = env._active_date
    env.reset(seed=42)
    d2 = env._active_date
    assert d1 == d2
