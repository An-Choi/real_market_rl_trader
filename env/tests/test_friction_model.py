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


# ---- 정밀 보정: 동적 스프레드(KRX 틱) + 체결일 기준 거래세 ----

from datetime import date

from friction.friction_model import krx_tick_size


def test_krx_tick_size_schedule() -> None:
    # 2023-01-25 개편 호가단위
    assert krx_tick_size(1_999) == 1
    assert krx_tick_size(2_000) == 5
    assert krx_tick_size(19_999) == 10
    assert krx_tick_size(49_999) == 50
    assert krx_tick_size(70_000) == 100
    assert krx_tick_size(199_999) == 100
    assert krx_tick_size(200_000) == 500
    assert krx_tick_size(499_999) == 500
    assert krx_tick_size(500_000) == 1_000


def test_dynamic_spread_uses_half_tick_over_price() -> None:
    model = FrictionModel(dynamic_spread=True)
    # 280,000원: 틱 500 → half-spread 250/280,000
    cost = model.calculate_spread_cost(10_000.0, price=280_000.0)
    assert cost == pytest.approx(10_000.0 * 250.0 / 280_000.0)
    # 70,000원: 틱 100 → half-spread 50/70,000
    cost = model.calculate_spread_cost(10_000.0, price=70_000.0)
    assert cost == pytest.approx(10_000.0 * 50.0 / 70_000.0)


def test_dynamic_spread_requires_price() -> None:
    model = FrictionModel(dynamic_spread=True)
    with pytest.raises(ValueError, match="price"):
        model.calculate_spread_cost(10_000.0)


def test_sell_tax_by_trade_date() -> None:
    model = FrictionModel(date_based_sell_tax=True)
    # 2025년: 농특세만 0.15% / 2026년~: 거래세 인상으로 0.20% / 2024년: 0.18%
    assert model.calculate_sell_tax(10_000.0, "sell", trade_date=date(2025, 6, 2)) == pytest.approx(15.0)
    assert model.calculate_sell_tax(10_000.0, "sell", trade_date=date(2026, 1, 2)) == pytest.approx(20.0)
    assert model.calculate_sell_tax(10_000.0, "sell", trade_date=date(2024, 7, 1)) == pytest.approx(18.0)
    assert model.calculate_sell_tax(10_000.0, "buy", trade_date=date(2026, 1, 2)) == 0.0


def test_date_based_sell_tax_requires_date() -> None:
    model = FrictionModel(date_based_sell_tax=True)
    with pytest.raises(ValueError, match="trade_date"):
        model.calculate_sell_tax(10_000.0, "sell")


def test_total_friction_dynamic_end_to_end() -> None:
    model = FrictionModel(
        fee_rate=0.00018, slippage_rate=0.0,
        dynamic_spread=True, date_based_sell_tax=True,
    )
    tv = 10_000.0
    sell = model.calculate_total_friction(
        tv, side="sell", price=280_000.0, trade_date=date(2026, 7, 21)
    )
    expected = tv * (0.00018 + 250.0 / 280_000.0 + 0.0020)
    assert sell == pytest.approx(expected)
