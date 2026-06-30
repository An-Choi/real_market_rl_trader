"""Gymnasium-style trading environment skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from friction.friction_model import FrictionModel


ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2


@dataclass
class TradeExecution:
    """Container for a simulated trade execution."""

    action: int
    price: float
    trade_value: float
    friction_cost: float


class TradingEnvironment(gym.Env):
    """Discrete-action market environment for RL trading experiments."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: pd.DataFrame,
        feature_columns: list[str],
        price_col: str = "Close",
        initial_cash: float = 10_000.0,
        max_position: float = 1.0,
        friction_model: FrictionModel | None = None,
        risk_penalty_rate: float = 0.0,
    ) -> None:
        """Initialize the trading environment.

        The initial action space is discrete:
        0 = Hold, 1 = Buy, 2 = Sell.
        """
        super().__init__()
        self.market_data = market_data.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.price_col = price_col
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.friction_model = friction_model or FrictionModel()
        self.risk_penalty_rate = risk_penalty_rate

        self.action_space = spaces.Discrete(3)
        observation_size = len(self.feature_columns) + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.trade_history: list[TradeExecution] = []

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment state."""
        # TODO: Support randomized episode starts for training.
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.trade_history = []
        return self._get_observation(), {"portfolio_value": self.portfolio_value}

    def step(self, action: int):
        """Advance one market step after executing an action."""
        # TODO: Add episode truncation, execution delay, and position size actions.
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_value = self.portfolio_value
        execution = self._execute_trade(action)

        self.current_step += 1
        self.portfolio_value = self._calculate_portfolio_value()
        reward = self._calculate_reward(
            previous_value=previous_value,
            current_value=self.portfolio_value,
            friction_cost=execution.friction_cost,
        )

        terminated = self.current_step >= len(self.market_data) - 1
        truncated = False
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "friction_cost": execution.friction_cost,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build the observation from market features and portfolio state."""
        row = self.market_data.iloc[self.current_step]
        features = (
            pd.to_numeric(row[self.feature_columns], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        portfolio_state = np.array(
            [
                self.cash / self.initial_cash,
                self.position / max(self.max_position, 1e-9),
                self.portfolio_value / self.initial_cash,
            ],
            dtype=np.float32,
        )
        return np.concatenate([features, portfolio_state]).astype(np.float32)

    def _calculate_reward(
        self,
        previous_value: float,
        current_value: float,
        friction_cost: float,
    ) -> float:
        """Calculate reward as return minus friction and risk penalties."""
        # TODO: Add drawdown penalty, inventory penalty, and risk-adjusted rewards.
        portfolio_return = (current_value - previous_value) / max(previous_value, 1e-9)
        friction_penalty = friction_cost / max(previous_value, 1e-9)
        risk_penalty = abs(self.position) * self.risk_penalty_rate
        return float(portfolio_return - friction_penalty - risk_penalty)

    def _execute_trade(self, action: int) -> TradeExecution:
        """Execute a discrete target-position trade."""
        price = float(self.market_data.iloc[self.current_step][self.price_col])
        target_position = self._action_to_target_position(action)
        trade_units = target_position - self.position
        trade_value = trade_units * price
        side = "sell" if trade_units < 0 else "buy"
        liquidity_score = self._current_liquidity_score()
        friction_cost = self.friction_model.calculate_total_friction(
            trade_value=trade_value,
            side=side,
            liquidity_score=liquidity_score,
        )

        self.cash -= trade_value
        self.cash -= friction_cost
        self.position = target_position

        execution = TradeExecution(
            action=action,
            price=price,
            trade_value=trade_value,
            friction_cost=friction_cost,
        )
        self.trade_history.append(execution)
        return execution

    def _action_to_target_position(self, action: int) -> float:
        """Map discrete action to target position."""
        if action == ACTION_BUY:
            return self.max_position
        if action == ACTION_SELL:
            return -self.max_position
        return self.position

    def _calculate_portfolio_value(self) -> float:
        """Calculate marked-to-market portfolio value."""
        current_price = float(self.market_data.iloc[self.current_step][self.price_col])
        return self.cash + (self.position * current_price)

    def _current_liquidity_score(self) -> float | None:
        """Return current liquidity score when available."""
        if "liquidity_score" not in self.market_data.columns:
            return None
        return float(self.market_data.iloc[self.current_step]["liquidity_score"])

    def get_current_market_row(self) -> pd.Series:
        """Expose the current market row for baseline agents and diagnostics."""
        return self.market_data.iloc[self.current_step]
