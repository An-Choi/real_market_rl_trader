"""Baseline trading agent skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class BuyAndHoldAgent:
    """Baseline that buys once and then holds."""

    def __init__(self) -> None:
        self.has_bought = False

    def predict(self, observation: Any, market_row: pd.Series | None = None) -> tuple[int, dict]:
        """Return Buy once, then Hold."""
        # TODO: Add reset hook for repeated backtests.
        if not self.has_bought:
            self.has_bought = True
            return 1, {}
        return 0, {}


@dataclass
class MovingAverageCrossoverAgent:
    """Baseline that trades from fast/slow moving average crossover."""

    fast_ma_col: str = "ma_5"
    slow_ma_col: str = "ma_20"

    def predict(self, observation: Any, market_row: pd.Series | None = None) -> tuple[int, dict]:
        """Return Buy when fast MA is above slow MA, Sell when below."""
        # TODO: Add crossover state to avoid repeated target-position signals.
        if market_row is None:
            return 0, {"reason": "missing_market_row"}
        if market_row[self.fast_ma_col] > market_row[self.slow_ma_col]:
            return 1, {}
        if market_row[self.fast_ma_col] < market_row[self.slow_ma_col]:
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
