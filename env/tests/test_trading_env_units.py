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


def test_friction_counted_once_in_reward(flat_data: pd.DataFrame) -> None:
    """비용은 cash에서 이미 차감되므로 reward에 단 한 번만 반영된다.

    가격이 100으로 고정이라 시장 PnL이 0이므로, 매수 step의 reward는
    온전히 friction 드래그(-friction_cost / previous_value)와 같아야 한다.
    """
    env = make_env(flat_data)
    env.reset(seed=0)
    previous_value = env.portfolio_value
    _, reward, _, _, info = env.step(1)  # Add 1 Unit (buy), 가격 변동 없음
    expected = -info["friction_cost"] / previous_value
    assert reward == pytest.approx(expected)


def test_multi_add_mark_to_market_uses_actual_shares() -> None:
    """서로 다른 가격에 Add한 Unit의 mark-to-market은 실제 보유 주식 수 기준이어야 한다.

    1 Unit = 고정 2000 notional. 100에 1 Unit(=20주), 200에 1 Unit(=10주)을 사면
    총 30주. 현재가 200이면 보유분 시가는 30 * 200 = 6000이어야 한다.
    friction을 0으로 두어 거래비용 드래그를 배제하고 순수 valuation만 검증한다.
    """
    prices = [100.0, 200.0, 200.0]
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    data = pd.DataFrame(
        {
            "Timestamp": ts,
            "Close": prices,
            "ma_5": prices,
        }
    )
    from friction.friction_model import FrictionModel

    env = TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
        friction_model=FrictionModel(
            fee_rate=0.0, spread_rate=0.0, slippage_rate=0.0, sell_tax_rate=0.0
        ),
    )
    env.reset(seed=0)
    env.step(1)  # step 0: Add 1 Unit @100 → 20 shares
    env.step(1)  # step 1: Add 1 Unit @200 → +10 shares (now 30 shares); valued @200

    held_value = env.portfolio_value - env.cash
    assert held_value == pytest.approx(6000.0)


def test_clear_realizes_mark_to_market_proceeds() -> None:
    """Clear는 원금 notional이 아니라 현재가 기준 매도대금을 현금화해야 한다.

    1 Unit = 2000 notional을 100에 매수(=20주) 후 가격이 110이 된 시점에 Clear하면
    매도대금은 20 * 110 = 2200이어야 한다. friction=0이면 청산 후 cash는
    10_000 - 2000(매수) + 2200(매도) = 10_200, portfolio_value도 10_200이어야 한다.
    """
    prices = [100.0, 110.0, 110.0]
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    data = pd.DataFrame({"Timestamp": ts, "Close": prices, "ma_5": prices})
    from friction.friction_model import FrictionModel

    env = TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
        friction_model=FrictionModel(
            fee_rate=0.0, spread_rate=0.0, slippage_rate=0.0, sell_tax_rate=0.0
        ),
    )
    env.reset(seed=0)
    env.step(1)  # Add 1 Unit @100 → 20 shares, cash 8000
    _, _, _, _, info = env.step(2)  # Clear @110 → proceeds 2200

    assert env.cash == pytest.approx(10_200.0)
    assert info["portfolio_value"] == pytest.approx(10_200.0)


def test_clear_sell_tax_on_market_proceeds() -> None:
    """매도 거래세는 원금 notional이 아니라 현재가 기준 매도대금에 부과된다.

    100에 1 Unit(20주) 매수 후 110에 Clear하면 매도대금은 2200. 다른 비용을 0으로
    두면 Clear step의 friction_cost는 정확히 2200 * sell_tax_rate여야 한다.
    """
    prices = [100.0, 110.0, 110.0]
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    data = pd.DataFrame({"Timestamp": ts, "Close": prices, "ma_5": prices})
    from friction.friction_model import FrictionModel

    sell_tax_rate = 0.002
    env = TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
        friction_model=FrictionModel(
            fee_rate=0.0, spread_rate=0.0, slippage_rate=0.0, sell_tax_rate=sell_tax_rate
        ),
    )
    env.reset(seed=0)
    env.step(1)  # Add @100
    _, _, _, _, info = env.step(2)  # Clear @110, proceeds 2200

    assert info["friction_cost"] == pytest.approx(2200.0 * sell_tax_rate)


def test_last_executable_bar_add_executes() -> None:
    """마지막 실행 bar(B−2)의 Add도 그대로 실행된다 — 강제청산 없음.

    spec r5 §3.2: env는 어떤 정산·강제 거래도 하지 않는다.
    """
    prices = [100.0, 101.0, 102.0]
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    data = pd.DataFrame({"Timestamp": ts, "Close": prices, "ma_5": prices})

    env = TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
    )
    env.reset(seed=0)
    env.step(0)  # bar0 실행 (3 bars → 2 steps)
    _, _, terminated, truncated, info = env.step(1)  # 마지막 실행 bar: Add

    assert truncated is True and terminated is False
    assert info["units_held"] == 1


def test_clear_sell_tax_exact_amount(flat_data: pd.DataFrame) -> None:
    """매도(Clear) step에서 거래세가 정확히 abs(trade_value) * sell_tax_rate 만큼만
    매수 대비 추가된다.

    매수/매도 거래 규모가 동일(1 Unit = 2000 notional)하고 다른 비용은
    좌우 대칭이므로, 두 step friction 차이는 정확히 매도 거래세여야 한다.
    """
    env = make_env(flat_data)
    env.reset(seed=0)
    sell_tax_rate = env.friction_model.sell_tax_rate
    assert sell_tax_rate > 0  # 이 검증이 의미 있으려면 세율이 0이 아니어야 함

    _, _, _, _, add_info = env.step(1)   # Add 1 Unit (buy)
    _, _, _, _, clear_info = env.step(2)  # Clear (sell), 동일 규모

    unit_notional = env.initial_cash * env.unit_fraction  # 1 Unit 거래 금액
    expected_tax = unit_notional * sell_tax_rate
    diff = clear_info["friction_cost"] - add_info["friction_cost"]
    assert diff == pytest.approx(expected_tax)
