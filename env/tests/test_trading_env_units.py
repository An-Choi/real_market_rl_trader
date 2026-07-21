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
            "ExecPrice": np.full(n, 100.0),
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


def test_target_20pct_selects_one_allocation_step(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)  # Target 20%
    assert info["units_held"] == 1
    assert info["actual_allocation"] == pytest.approx(0.20)


def test_target_100pct_reaches_full_allocation(flat_data: pd.DataFrame) -> None:
    from friction.friction_model import FrictionModel

    env = TradingEnvironment(
        market_data=flat_data, feature_columns=["ma_5"],
        initial_cash=10_000.0, unit_fraction=0.20, max_units=5,
        friction_model=FrictionModel(fee_rate=0.0, spread_rate=0.0,
                                     slippage_rate=0.0, sell_tax_rate=0.0),
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(5)
    assert info["units_held"] == 5
    assert env.cash == pytest.approx(0.0)
    assert info["actual_allocation"] == pytest.approx(1.0)


def test_target_100pct_accounts_for_friction_without_negative_cash(
    flat_data: pd.DataFrame,
) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(5)
    assert info["units_held"] == 5
    assert env.cash >= 0.0
    assert info["actual_allocation"] == pytest.approx(1.0)


def test_all_target_allocations_are_valid(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    assert env.action_masks().tolist() == [True] * 6
    env.step(5)
    assert env.action_masks().tolist() == [True] * 6


@pytest.mark.parametrize("action", range(6))
def test_each_action_reaches_its_target_weight(
    flat_data: pd.DataFrame,
    action: int,
) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(action)
    assert info["actual_allocation"] == pytest.approx(action / 5)


def test_repeating_same_target_is_no_op(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    env.step(3)
    shares = env.shares_held
    cash = env.cash
    _, _, _, _, info = env.step(3)
    assert info["trade_value"] == 0.0
    assert info["friction_cost"] == 0.0
    assert env.shares_held == shares
    assert env.cash == cash


def test_lower_target_partially_sells_to_requested_weight(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    env.step(5)
    _, _, _, _, info = env.step(2)
    assert info["trade_value"] < 0.0
    assert info["actual_allocation"] == pytest.approx(0.40)


def test_cash_never_negative_under_random_actions(flat_data: pd.DataFrame) -> None:
    """불변식: 어떤 행동 시퀀스에서도 cash ≥ 0."""
    rng = np.random.default_rng(3)
    env = make_env(flat_data)
    env.reset(seed=3)
    for _ in range(9):
        _, _, terminated, truncated, _ = env.step(int(rng.integers(0, 6)))
        assert env.cash >= 0.0
        if terminated or truncated:
            break


def test_target_weights_do_not_depend_on_legacy_unit_fraction(flat_data: pd.DataFrame) -> None:
    env = TradingEnvironment(
        market_data=flat_data, feature_columns=["ma_5"],
        initial_cash=10_000.0, unit_fraction=0.199, max_units=5,
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(5)
    assert info["units_held"] == 5
    assert env.cash >= 0.0
    assert info["actual_allocation"] == pytest.approx(1.0)


def test_target_zero_clears_position(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    env.step(2)
    _, _, _, _, info = env.step(0)  # Target 0%
    assert info["units_held"] == 0


def test_long_only_no_negative_units(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)  # Target 0% while flat → no-op
    assert info["units_held"] == 0


def test_invalid_action_raises(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(6)


def test_friction_charged_when_increasing_target(flat_data: pd.DataFrame) -> None:
    env = make_env(flat_data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)
    assert info["friction_cost"] > 0


def test_friction_counted_once_in_reward(flat_data: pd.DataFrame) -> None:
    """비용은 cash에서 이미 차감되므로 reward에 단 한 번만 반영된다.

    가격이 100으로 고정이라 시장 PnL이 0이므로, 매수 step의 reward는
    온전히 friction 드래그(-friction_cost / previous_value)와 같아야 한다.
    """
    env = make_env(flat_data)
    env.reset(seed=0)
    previous_value = env.portfolio_value
    _, reward, _, _, info = env.step(1)
    expected = -info["friction_cost"] / previous_value
    assert reward == pytest.approx(expected)


def test_target_change_uses_current_portfolio_value() -> None:
    """가격 변동 후 40% 목표는 최초 자본이 아닌 현재 자산 기준이어야 한다."""
    prices = [100.0, 200.0, 200.0]
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    data = pd.DataFrame(
        {
            "Timestamp": ts,
            "Close": prices,
            "ExecPrice": prices,
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
    env.step(1)  # 20% target at 100
    _, _, _, _, info = env.step(2)  # 40% target at 200

    assert info["actual_allocation"] == pytest.approx(0.40)


def test_clear_realizes_mark_to_market_proceeds() -> None:
    """Clear는 원금 notional이 아니라 현재가 기준 매도대금을 현금화해야 한다.

    1 Unit = 2000 notional을 100에 매수(=20주) 후 가격이 110이 된 시점에 Clear하면
    매도대금은 20 * 110 = 2200이어야 한다. friction=0이면 청산 후 cash는
    10_000 - 2000(매수) + 2200(매도) = 10_200, portfolio_value도 10_200이어야 한다.
    """
    prices = [100.0, 110.0, 110.0]
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    data = pd.DataFrame({"Timestamp": ts, "Close": prices, "ExecPrice": prices, "ma_5": prices})
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
    env.step(1)  # Target 20% @100
    _, _, _, _, info = env.step(0)  # Target 0% @110

    assert env.cash == pytest.approx(10_200.0)
    assert info["portfolio_value"] == pytest.approx(10_200.0)


def test_clear_sell_tax_on_market_proceeds() -> None:
    """매도 거래세는 원금 notional이 아니라 현재가 기준 매도대금에 부과된다.

    100에 1 Unit(20주) 매수 후 110에 Clear하면 매도대금은 2200. 다른 비용을 0으로
    두면 Clear step의 friction_cost는 정확히 2200 * sell_tax_rate여야 한다.
    """
    prices = [100.0, 110.0, 110.0]
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    data = pd.DataFrame({"Timestamp": ts, "Close": prices, "ExecPrice": prices, "ma_5": prices})
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
    _, _, _, _, info = env.step(0)  # Target 0% @110, proceeds 2200

    assert info["friction_cost"] == pytest.approx(2200.0 * sell_tax_rate)


def test_last_executable_bar_add_executes() -> None:
    """마지막 실행 bar(B−2)의 Add도 그대로 실행된다 — 강제청산 없음.

    spec r5 §3.2: env는 어떤 정산·강제 거래도 하지 않는다.
    """
    prices = [100.0, 101.0, 102.0]
    ts = pd.date_range("2025-06-02 09:00", periods=len(prices), freq="1min", tz="Asia/Seoul")
    data = pd.DataFrame({"Timestamp": ts, "Close": prices, "ExecPrice": prices, "ma_5": prices})

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


def test_target_zero_sell_tax_exact_amount(flat_data: pd.DataFrame) -> None:
    """0% 목표 매도의 friction에는 매도 거래세가 정확히 포함된다."""
    env = make_env(flat_data)
    env.reset(seed=0)
    sell_tax_rate = env.friction_model.sell_tax_rate
    assert sell_tax_rate > 0  # 이 검증이 의미 있으려면 세율이 0이 아니어야 함

    env.step(1)
    _, _, _, _, clear_info = env.step(0)
    sold_notional = abs(clear_info["trade_value"])
    symmetric_cost = sold_notional * (
        env.friction_model.fee_rate
        + env.friction_model.spread_rate
        + env.friction_model.slippage_rate
    )
    expected = symmetric_cost + sold_notional * sell_tax_rate
    assert clear_info["friction_cost"] == pytest.approx(expected)
