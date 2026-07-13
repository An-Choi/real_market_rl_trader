"""Baseline trading agent skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_BASELINES = ("buy_and_hold", "random", "ma_crossover")


class BuyAndHoldAgent:
    """Baseline that scales to full allocation, then holds to the end."""

    def __init__(self, target_units: int = 5) -> None:
        if target_units <= 0:
            raise ValueError("target_units must be positive")
        self.target_units = target_units
        self.units_requested = 0

    def reset(self) -> None:
        """Reset state for a fresh evaluation episode."""
        self.units_requested = 0

    def predict(self, observation: Any, market_row: pd.Series | None = None) -> tuple[int, dict]:
        """Return Buy once, then Hold."""
        if self.units_requested < self.target_units:
            self.units_requested += 1
            return 1, {}
        return 0, {}


@dataclass
class MovingAverageCrossoverAgent:
    """Baseline that trades from fast/slow moving average crossover."""

    fast_window: int = 5
    slow_window: int = 20
    price_col: str = "Close"

    def __post_init__(self) -> None:
        self._prices: list[float] = []

    def reset(self) -> None:
        """Reset rolling price history for a fresh evaluation episode."""
        self._prices = []

    def predict(self, observation: Any, market_row: pd.Series | None = None) -> tuple[int, dict]:
        """Return Buy when fast MA is above slow MA, Sell when below."""
        if market_row is None:
            return 0, {"reason": "missing_market_row"}
        if self.price_col not in market_row:
            return 0, {"reason": "missing_price"}

        self._prices.append(float(market_row[self.price_col]))
        if len(self._prices) < self.slow_window:
            return 0, {"reason": "warming_up"}

        fast_ma = float(np.mean(self._prices[-self.fast_window:]))
        slow_ma = float(np.mean(self._prices[-self.slow_window:]))
        if fast_ma > slow_ma:
            return 1, {}
        if fast_ma < slow_ma:
            return 2, {}
        return 0, {}


class RandomAgent:
    """Random discrete-action baseline."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, observation: Any, market_row: pd.Series | None = None) -> tuple[int, dict]:
        """Sample Hold, Buy, or Sell uniformly."""
        # TODO: Support action probabilities from config.
        return int(self.rng.integers(0, 3)), {}


@dataclass
class RuleBasedRegimeAgent:
    """Simple rule-based agent using regime feature proxies."""

    return_col: str = "return_1"
    volatility_col: str = "volatility_20"
    max_volatility: float = 0.03

    def predict(self, observation: Any, market_row: pd.Series | None = None) -> tuple[int, dict]:
        """Trade in the direction of return when volatility is acceptable."""
        # TODO: Replace heuristic thresholds with configurable regime labels.
        if market_row is None:
            return 0, {"reason": "missing_market_row"}
        if market_row[self.volatility_col] > self.max_volatility:
            return 0, {"reason": "high_volatility"}
        if market_row[self.return_col] > 0:
            return 1, {}
        if market_row[self.return_col] < 0:
            return 2, {}
        return 0, {}


def make_baseline_agent(
    name: str,
    *,
    seed: int | None = None,
    max_units: int = 5,
) -> Any:
    """Create a supported rule-based baseline policy by experiment name."""
    if name == "buy_and_hold":
        return BuyAndHoldAgent(target_units=max_units)
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "ma_crossover":
        return MovingAverageCrossoverAgent()
    raise ValueError(f"Unknown baseline agent: {name}")
