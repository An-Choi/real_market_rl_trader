"""Backtest engine skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BacktestEngine:
    """Run an agent through a trading environment and collect results."""

    agent: Any
    environment: Any
    results: list[dict] = field(default_factory=list)

    def run(self, max_steps: int | None = None) -> pd.DataFrame:
        """Execute a backtest loop."""
        # TODO: Add vectorized evaluation, logging, and benchmark comparison.
        observation, info = self.environment.reset()
        done = False
        step_count = 0
        self.results = []

        while not done:
            market_row = self.environment.get_current_market_row()
            action = self._predict_action(observation, market_row=market_row)
            observation, reward, terminated, truncated, info = self.environment.step(action)

            self.results.append(
                {
                    "step": step_count,
                    "action": action,
                    "reward": reward,
                    "portfolio_value": info["portfolio_value"],
                    "position": info["position"],
                    "friction_cost": info["friction_cost"],
                }
            )

            step_count += 1
            done = terminated or truncated
            if max_steps is not None and step_count >= max_steps:
                break

        return self.collect_results()

    def collect_results(self) -> pd.DataFrame:
        """Return collected backtest results as a DataFrame."""
        # TODO: Include trade blotter and benchmark series.
        return pd.DataFrame(self.results)

    def _predict_action(self, observation: Any, market_row: pd.Series) -> int:
        """Call an agent predict method while supporting simple baselines."""
        try:
            action_output = self.agent.predict(observation, market_row=market_row)
        except TypeError:
            action_output = self.agent.predict(observation)

        if isinstance(action_output, tuple):
            return int(action_output[0])
        return int(action_output)
