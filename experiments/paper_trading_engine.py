"""Paper trading engine skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PaperTradingEngine:
    """Route live-like market snapshots through an agent without real orders."""

    agent: Any
    feature_engineer: Any | None = None
    latest_market_data: list[dict] = field(default_factory=list)

    def on_market_data(self, market_snapshot: dict) -> int:
        """Receive a new market snapshot and return the selected action."""
        # TODO: Add validation, feature warm-up, and latency measurement.
        self.latest_market_data.append(market_snapshot)
        observation = self._build_observation()
        action_output = self.agent.predict(observation)
        action = int(action_output[0]) if isinstance(action_output, tuple) else int(action_output)
        self._send_paper_order(action, market_snapshot)
        return action

    def _build_observation(self) -> Any:
        """Build the latest observation for the agent."""
        # TODO: Share observation construction with TradingEnvironment.
        market_data = pd.DataFrame(self.latest_market_data)
        if self.feature_engineer is not None:
            market_data = self.feature_engineer.transform(market_data)
        return market_data.tail(1)

    def _send_paper_order(self, action: int, market_snapshot: dict) -> None:
        """Placeholder for simulated order routing."""
        # TODO: Connect to exchange sandbox or broker paper trading API.
        _ = (action, market_snapshot)

    def start(self) -> None:
        """Start the paper trading loop."""
        # TODO: Wire streaming market data source here.
        raise NotImplementedError("Streaming paper trading loop is not implemented yet.")
