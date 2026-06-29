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

    def reset(self) -> None:
        """Reset state before a new backtest run."""
        self.has_bought = False

    def predict(self, observation: Any, market_row: pd.Series | None = None) -> tuple[int, dict]:
        """Return Buy once, then Hold."""
        if not self.has_bought:
            self.has_bought = True
            return 1, {}
        return 0, {}


@dataclass
class MovingAverageCrossoverAgent:
    """Baseline that trades from fast/slow moving average crossover."""

    fast_ma_col: str = "ma_5"
    slow_ma_col: str = "ma_20"

    def reset(self) -> None:
        """Reset state before a new backtest run."""
        return None

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
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset the RNG so repeated backtests are reproducible."""
        self.rng = np.random.default_rng(self.seed)

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

    def reset(self) -> None:
        """Reset state before a new backtest run."""
        return None

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


def make_baseline_agent(name: str, seed: int | None = None) -> Any:
    """Create a baseline agent by config-friendly name."""
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"buy_and_hold", "buyhold"}:
        return BuyAndHoldAgent()
    if normalized in {"moving_average_crossover", "ma_crossover"}:
        return MovingAverageCrossoverAgent()
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized in {"rule_based_regime", "regime"}:
        return RuleBasedRegimeAgent()
    raise ValueError(f"Unknown baseline agent: {name}")
