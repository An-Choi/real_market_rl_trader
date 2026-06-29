from __future__ import annotations

import pytest

from friction.friction_model import FrictionModel


@pytest.fixture
def model() -> FrictionModel:
    # 명시적 값으로 고정 (기본값과 동일하지만 테스트 의도를 분명히)
    return FrictionModel(
        fee_rate=0.001,
        spread_rate=0.0005,
        slippage_rate=0.0005,
        execution_uncertainty_rate=0.0,
        sell_tax_rate=0.002,
    )


def test_sell_tax_zero_on_buy(model: FrictionModel) -> None:
    assert model.calculate_sell_tax(10_000.0, side="buy") == 0.0


def test_sell_tax_applied_on_sell(model: FrictionModel) -> None:
    # 0.002 * 10_000 = 20.0
    assert model.calculate_sell_tax(10_000.0, side="sell") == pytest.approx(20.0)


def test_sell_tax_uses_absolute_value(model: FrictionModel) -> None:
    # 음수 trade_value(매도 방향)도 절대값 기준
    assert model.calculate_sell_tax(-10_000.0, side="sell") == pytest.approx(20.0)


def test_buy_total_has_no_sell_tax(model: FrictionModel) -> None:
    # buy: fee+spread+slippage = (0.001+0.0005+0.0005)*10_000 = 20.0
    total = model.calculate_total_friction(10_000.0, side="buy")
    assert total == pytest.approx(20.0)


def test_sell_total_includes_sell_tax(model: FrictionModel) -> None:
    # sell: buy비용 20.0 + 거래세 20.0 = 40.0
    total = model.calculate_total_friction(10_000.0, side="sell")
    assert total == pytest.approx(40.0)


def test_sell_minus_buy_equals_sell_tax(model: FrictionModel) -> None:
    buy = model.calculate_total_friction(10_000.0, side="buy")
    sell = model.calculate_total_friction(10_000.0, side="sell")
    assert sell - buy == pytest.approx(10_000.0 * model.sell_tax_rate)


def test_round_trip_cost_formula(model: FrictionModel) -> None:
    # 왕복: buy(fee+spread+slippage) + sell(fee+spread+slippage+tax)
    tv = 10_000.0
    buy = model.calculate_total_friction(tv, side="buy")
    sell = model.calculate_total_friction(tv, side="sell")
    expected = (
        2 * model.fee_rate * tv
        + 2 * model.spread_rate * tv
        + 2 * model.slippage_rate * tv
        + 1 * model.sell_tax_rate * tv
    )
    assert buy + sell == pytest.approx(expected)
