from __future__ import annotations

import pandas as pd
import pytest

from env.trading_env import ACTION_BUY, ACTION_HOLD, ACTION_SELL, TradingEnvironment


@pytest.fixture
def env() -> TradingEnvironment:
    market_data = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0, 103.0],
            "feature": [0.0, 0.1, 0.2, 0.3],
        }
    )
    return TradingEnvironment(
        market_data=market_data,
        feature_columns=["feature"],
        max_position=1.0,
    )


def test_sell_from_flat_keeps_position_flat(env: TradingEnvironment) -> None:
    env.reset(seed=0)

    _, _, _, _, info = env.step(ACTION_SELL)

    assert info["position"] == 0.0
    assert env.position == 0.0


def test_buy_enters_long_position(env: TradingEnvironment) -> None:
    env.reset(seed=0)

    _, _, _, _, info = env.step(ACTION_BUY)

    assert info["position"] == pytest.approx(env.max_position)
    assert env.position == pytest.approx(env.max_position)


def test_sell_after_buy_exits_to_flat(env: TradingEnvironment) -> None:
    env.reset(seed=0)
    env.step(ACTION_BUY)

    _, _, _, _, info = env.step(ACTION_SELL)

    assert info["position"] == 0.0
    assert env.position == 0.0


def test_hold_keeps_current_position(env: TradingEnvironment) -> None:
    env.reset(seed=0)
    env.step(ACTION_BUY)

    _, _, _, _, info = env.step(ACTION_HOLD)

    assert info["position"] == pytest.approx(env.max_position)
    assert env.position == pytest.approx(env.max_position)
