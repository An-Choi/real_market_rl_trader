"""Reward calculation helpers for trading environments."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward penalties.

    The base reward is cost-deducted portfolio return because execution costs
    have already been applied to cash before portfolio value is calculated.
    """

    inventory_penalty_rate: float = 0.0
    turnover_penalty_rate: float = 0.0
    drawdown_penalty_rate: float = 0.0
    downside_penalty_rate: float = 0.0
    benchmark_relative_rate: float = 0.0
    reward_scale: float = 1.0
    return_mode: str = "simple_return"


@dataclass(frozen=True)
class RewardTerms:
    """Reward decomposition for logging and ablation."""

    reward: float
    portfolio_return: float
    base_return: float
    benchmark_return: float
    benchmark_simple_return: float
    benchmark_relative_reward: float
    inventory_penalty: float
    turnover_penalty: float
    drawdown_penalty: float
    downside_penalty: float


def calculate_reward_terms(
    *,
    previous_value: float,
    current_value: float,
    previous_peak_value: float,
    previous_market_price: float,
    current_market_price: float,
    units_held: int,
    max_units: int,
    trade_value: float,
    config: RewardConfig,
) -> RewardTerms:
    """Calculate scaled return reward with explicit, ablatable shaping terms."""
    safe_previous = max(previous_value, 1e-9)
    portfolio_return = (current_value - previous_value) / safe_previous
    benchmark_simple_return = (
        current_market_price - previous_market_price
    ) / max(previous_market_price, 1e-9)
    if config.return_mode == "log_return":
        base_return = math.log(max(current_value, 1e-9) / safe_previous)
        benchmark_return = math.log(
            max(current_market_price, 1e-9) / max(previous_market_price, 1e-9)
        )
    elif config.return_mode == "simple_return":
        base_return = portfolio_return
        benchmark_return = benchmark_simple_return
    else:
        raise ValueError(f"unsupported reward return_mode: {config.return_mode}")

    benchmark_relative_reward = (
        base_return - benchmark_return
    ) * config.benchmark_relative_rate
    exposure = units_held / max(max_units, 1)
    inventory_penalty = exposure * config.inventory_penalty_rate
    turnover_penalty = (abs(trade_value) / safe_previous) * config.turnover_penalty_rate
    previous_peak = max(previous_peak_value, 1e-9)
    current_peak = max(previous_peak, current_value)
    previous_drawdown = max(0.0, (previous_peak - previous_value) / previous_peak)
    current_drawdown = max(0.0, (current_peak - current_value) / current_peak)
    drawdown_increase = max(0.0, current_drawdown - previous_drawdown)
    drawdown_penalty = drawdown_increase * config.drawdown_penalty_rate
    downside_penalty = max(0.0, -base_return) * config.downside_penalty_rate
    unscaled_reward = (
        base_return
        + benchmark_relative_reward
        - inventory_penalty
        - turnover_penalty
        - drawdown_penalty
        - downside_penalty
    )
    reward = unscaled_reward * config.reward_scale

    return RewardTerms(
        reward=float(reward),
        portfolio_return=float(portfolio_return),
        base_return=float(base_return),
        benchmark_return=float(benchmark_return),
        benchmark_simple_return=float(benchmark_simple_return),
        benchmark_relative_reward=float(benchmark_relative_reward),
        inventory_penalty=float(inventory_penalty),
        turnover_penalty=float(turnover_penalty),
        drawdown_penalty=float(drawdown_penalty),
        downside_penalty=float(downside_penalty),
    )
