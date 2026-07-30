"""Baseline trading agent skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_BASELINES = (
    "cash",
    "static_20pct",
    "static_40pct",
    "static_60pct",
    "static_80pct",
    "buy_and_hold",
    "volatility_scaled",
    "random",
    "ma_crossover",
)


@dataclass(frozen=True)
class StaticAllocationAgent:
    """Baseline that continuously targets one fixed allocation."""

    target_units: int

    def __post_init__(self) -> None:
        if self.target_units < 0:
            raise ValueError("target_units must be non-negative")

    def reset(self) -> None:
        """Static allocation has no internal state."""

    def predict(
        self,
        observation: Any,
        market_row: pd.Series | None = None,
    ) -> tuple[int, dict]:
        """Return the configured target on every step."""
        return self.target_units, {}


class BuyAndHoldAgent(StaticAllocationAgent):
    """Baseline that selects 100% allocation and holds to the end."""

    def __init__(self, target_units: int = 5) -> None:
        if target_units <= 0:
            raise ValueError("target_units must be positive")
        super().__init__(target_units=target_units)


@dataclass
class MovingAverageCrossoverAgent:
    """Baseline that trades from fast/slow moving average crossover."""

    fast_window: int = 5
    slow_window: int = 20
    price_col: str = "Close"
    target_units: int = 5

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
            return self.target_units, {}
        if fast_ma < slow_ma:
            return 0, {}
        return 0, {}


class RandomAgent:
    """Random discrete-action baseline."""

    def __init__(self, seed: int | None = None, action_count: int = 6) -> None:
        self.rng = np.random.default_rng(seed)
        self.action_count = action_count

    def predict(self, observation: Any, market_row: pd.Series | None = None) -> tuple[int, dict]:
        """Sample one target allocation uniformly."""
        # TODO: Support action probabilities from config.
        return int(self.rng.integers(0, self.action_count)), {}


@dataclass
class VolatilityScaledAgent:
    """Causal inverse-volatility allocation using a rolling median target."""

    volatility_col: str = "realized_vol_12"
    window: int = 20
    max_units: int = 5

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.max_units <= 0:
            raise ValueError("max_units must be positive")
        self._volatility: list[float] = []

    def reset(self) -> None:
        """Reset causal volatility history for a fresh evaluation."""
        self._volatility = []

    def predict(
        self,
        observation: Any,
        market_row: pd.Series | None = None,
    ) -> tuple[int, dict]:
        """Reduce exposure when current volatility exceeds its recent median."""
        if market_row is None or self.volatility_col not in market_row:
            return 0, {"reason": "missing_volatility"}
        volatility = float(market_row[self.volatility_col])
        if not np.isfinite(volatility) or volatility < 0:
            return 0, {"reason": "invalid_volatility"}

        self._volatility.append(volatility)
        history = self._volatility[-self.window:]
        positive = [value for value in history if value > 0]
        if not positive:
            return self.max_units, {"reason": "zero_volatility"}

        target_volatility = float(np.median(positive))
        allocation = min(target_volatility / max(volatility, 1e-12), 1.0)
        target_units = int(np.clip(np.rint(allocation * self.max_units), 0, self.max_units))
        return target_units, {}


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
            return 5, {}
        if market_row[self.return_col] < 0:
            return 0, {}
        return 0, {}


def make_baseline_agent(
    name: str,
    *,
    seed: int | None = None,
    max_units: int = 5,
) -> Any:
    """Create a supported rule-based baseline policy by experiment name."""
    if name == "cash":
        return StaticAllocationAgent(target_units=0)
    if name.startswith("static_") and name.endswith("pct"):
        percentage = int(name.removeprefix("static_").removesuffix("pct"))
        target_units = round((percentage / 100) * max_units)
        return StaticAllocationAgent(target_units=target_units)
    if name == "buy_and_hold":
        return BuyAndHoldAgent(target_units=max_units)
    if name == "volatility_scaled":
        return VolatilityScaledAgent(max_units=max_units)
    if name == "random":
        return RandomAgent(seed=seed, action_count=max_units + 1)
    if name == "ma_crossover":
        return MovingAverageCrossoverAgent(target_units=max_units)
    raise ValueError(f"Unknown baseline agent: {name}")
