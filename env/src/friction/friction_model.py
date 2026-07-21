"""Trading friction model — 정적 비율 + 정밀 모드(동적 스프레드·시기별 거래세)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# KRX 호가가격단위 (2023-01-25 개편, 유가증권시장).
_KRX_TICK_SCHEDULE: tuple[tuple[float, int], ...] = (
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
)
_KRX_TICK_MAX = 1_000

# 코스피 매도 거래세(증권거래세+농특세) 연혁. (시행일, 세율) 오름차순.
_KOSPI_SELL_TAX_SCHEDULE: tuple[tuple[date, float], ...] = (
    (date(2023, 1, 1), 0.0020),
    (date(2024, 1, 1), 0.0018),
    (date(2025, 1, 1), 0.0015),
    (date(2026, 1, 1), 0.0020),  # 2026년 거래세 본세 0.05% 인상 + 농특세 0.15%
)


def krx_tick_size(price: float) -> int:
    """KRX 유가증권시장 호가단위."""
    for upper, tick in _KRX_TICK_SCHEDULE:
        if price < upper:
            return tick
    return _KRX_TICK_MAX


def _sell_tax_rate_on(trade_date: date) -> float:
    rate = _KOSPI_SELL_TAX_SCHEDULE[0][1]
    for effective, schedule_rate in _KOSPI_SELL_TAX_SCHEDULE:
        if trade_date >= effective:
            rate = schedule_rate
    return rate


@dataclass
class FrictionModel:
    """Estimate transaction costs from fee, spread, slippage, and uncertainty.

    정밀 모드: ``dynamic_spread``는 half-spread를 `0.5 × tick(price) / price`로
    계산(가격이 틱 경계를 넘나드는 데이터에서 정적 비율은 부정확),
    ``date_based_sell_tax``는 체결일 기준 세율 연혁을 적용(2025년 0.15% →
    2026년 0.20%). 두 모드는 필요한 인자(price/trade_date)가 없으면 조용히
    정적 값으로 후퇴하지 않고 실패한다(fail-closed).
    """

    fee_rate: float = 0.001
    spread_rate: float = 0.0005
    slippage_rate: float = 0.0005
    execution_uncertainty_rate: float = 0.0
    sell_tax_rate: float = 0.002
    dynamic_spread: bool = False
    date_based_sell_tax: bool = False

    def calculate_fee(self, trade_value: float) -> float:
        """Calculate exchange or broker fee."""
        return abs(trade_value) * self.fee_rate

    def calculate_spread_cost(self, trade_value: float, price: float | None = None) -> float:
        """Calculate estimated spread cost (dynamic 모드는 mid 대비 half-spread)."""
        if self.dynamic_spread:
            if price is None or price <= 0:
                raise ValueError("dynamic_spread requires a positive execution price")
            return abs(trade_value) * (0.5 * krx_tick_size(price) / price)
        return abs(trade_value) * self.spread_rate

    def calculate_slippage(
        self,
        trade_value: float,
        liquidity_score: float | None = None,
    ) -> float:
        """Calculate estimated slippage cost."""
        liquidity_adjustment = 1.0
        if liquidity_score is not None and liquidity_score > 0:
            liquidity_adjustment = 1.0 / liquidity_score
        return abs(trade_value) * self.slippage_rate * liquidity_adjustment

    def calculate_execution_uncertainty_cost(self, trade_value: float) -> float:
        """Calculate placeholder cost for execution uncertainty."""
        # TODO: Model partial fills, latency, and failed order probability.
        return abs(trade_value) * self.execution_uncertainty_rate

    def calculate_sell_tax(
        self, trade_value: float, side: str, trade_date: date | None = None
    ) -> float:
        """Korean securities transaction tax — applied on sells only."""
        if side != "sell":
            return 0.0
        if self.date_based_sell_tax:
            if trade_date is None:
                raise ValueError("date_based_sell_tax requires trade_date")
            return abs(trade_value) * _sell_tax_rate_on(trade_date)
        return abs(trade_value) * self.sell_tax_rate

    def calculate_total_friction(
        self,
        trade_value: float,
        side: str,
        liquidity_score: float | None = None,
        price: float | None = None,
        trade_date: date | None = None,
    ) -> float:
        """Calculate total estimated trading friction."""
        fee = self.calculate_fee(trade_value)
        spread = self.calculate_spread_cost(trade_value, price=price)
        slippage = self.calculate_slippage(trade_value, liquidity_score=liquidity_score)
        uncertainty = self.calculate_execution_uncertainty_cost(trade_value)
        sell_tax = self.calculate_sell_tax(trade_value, side, trade_date=trade_date)
        return fee + spread + slippage + uncertainty + sell_tax
