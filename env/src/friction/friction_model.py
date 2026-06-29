"""Trading friction model skeleton."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrictionModel:
    """Estimate transaction costs from fee, spread, slippage, and uncertainty."""

    fee_rate: float = 0.001
    spread_rate: float = 0.0005
    slippage_rate: float = 0.0005
    execution_uncertainty_rate: float = 0.0
    sell_tax_rate: float = 0.002

    def calculate_fee(self, trade_value: float) -> float:
        """Calculate exchange or broker fee."""
        # TODO: Support maker/taker fee schedules.
        return abs(trade_value) * self.fee_rate

    def calculate_spread_cost(self, trade_value: float) -> float:
        """Calculate estimated spread cost."""
        # TODO: Replace fixed spread rate with quote-derived spread.
        return abs(trade_value) * self.spread_rate

    def calculate_slippage(
        self,
        trade_value: float,
        liquidity_score: float | None = None,
    ) -> float:
        """Calculate estimated slippage cost."""
        # TODO: Model slippage as a function of order size and market liquidity.
        liquidity_adjustment = 1.0
        if liquidity_score is not None and liquidity_score > 0:
            liquidity_adjustment = 1.0 / liquidity_score
        return abs(trade_value) * self.slippage_rate * liquidity_adjustment

    def calculate_execution_uncertainty_cost(self, trade_value: float) -> float:
        """Calculate placeholder cost for execution uncertainty."""
        # TODO: Model partial fills, latency, and failed order probability.
        return abs(trade_value) * self.execution_uncertainty_rate

    def calculate_sell_tax(self, trade_value: float, side: str) -> float:
        """Korean securities transaction tax — applied on sells only."""
        if side == "sell":
            return abs(trade_value) * self.sell_tax_rate
        return 0.0

    def calculate_total_friction(
        self,
        trade_value: float,
        side: str,
        liquidity_score: float | None = None,
    ) -> float:
        """Calculate total estimated trading friction."""
        fee = self.calculate_fee(trade_value)
        spread = self.calculate_spread_cost(trade_value)
        slippage = self.calculate_slippage(trade_value, liquidity_score=liquidity_score)
        uncertainty = self.calculate_execution_uncertainty_cost(trade_value)
        sell_tax = self.calculate_sell_tax(trade_value, side)
        return fee + spread + slippage + uncertainty + sell_tax
