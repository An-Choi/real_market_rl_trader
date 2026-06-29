from __future__ import annotations

import math

import numpy as np
import pandas as pd

from env.trading_env import TradingEnvironment, TradeExecution
from env.virtual_agents import (
    MeanReversionTrader,
    NoiseTrader,
    TrendFollower,
    VirtualAgent,
)
from friction.friction_model import FrictionModel


class HybridMarketEnv(TradingEnvironment):
    """OHLCV backbone + VirtualAgent price impact (deterministic via reset(seed=))."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        feature_columns: list[str],
        price_col: str = "Close",
        initial_cash: float = 10_000.0,
        max_position: float = 1.0,
        friction_model: FrictionModel | None = None,
        risk_penalty_rate: float = 0.0,
        noise_strength: float = 0.001,
        trend_strength: float = 0.002,
        mean_reversion_strength: float = 0.001,
        max_agent_impact: float = 0.05,
    ) -> None:
        super().__init__(
            market_data=market_data,
            feature_columns=feature_columns,
            price_col=price_col,
            initial_cash=initial_cash,
            max_position=max_position,
            friction_model=friction_model,
            risk_penalty_rate=risk_penalty_rate,
        )
        self.virtual_agents: list[VirtualAgent] = [
            NoiseTrader(noise_strength=noise_strength),
            TrendFollower(trend_strength=trend_strength, price_col=price_col),
            MeanReversionTrader(
                mean_reversion_strength=mean_reversion_strength,
                price_col=price_col,
            ),
        ]
        self.max_agent_impact = float(max_agent_impact)

        self.simulated_price: float = math.nan
        self.real_price: float = math.nan
        self.agent_impact: float = math.nan
        self.execution_price: float = math.nan
        self.execution_step: int | None = None
        self.execution_real_price: float = math.nan
        self.execution_agent_impact: float = math.nan
        self.valuation_step: int | None = None
        self._impact_cache: dict[int, tuple[float, float, float]] = {}

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        if seed is not None:
            agent_rng = np.random.default_rng(seed)
            for agent in self.virtual_agents:
                sub_seed = int(agent_rng.integers(0, 2**32 - 1))
                agent.reset(seed=sub_seed)
        self.simulated_price = math.nan
        self.real_price = math.nan
        self.agent_impact = math.nan
        self.execution_price = math.nan
        self.execution_step = None
        self.execution_real_price = math.nan
        self.execution_agent_impact = math.nan
        self.valuation_step = None
        self._impact_cache = {}
        return obs, info

    def _compute_impact_at(self, step: int) -> tuple[float, float, float]:
        cached = self._impact_cache.get(step)
        if cached is not None:
            return cached
        real_price = float(self.market_data.iloc[step][self.price_col])
        history = self.market_data.iloc[: step + 1]
        raw_impact = sum(
            agent.compute_impact(history, step) for agent in self.virtual_agents
        )
        clipped = float(
            np.clip(raw_impact, -self.max_agent_impact, self.max_agent_impact)
        )
        simulated_price = real_price * (1.0 + clipped)
        result = (real_price, simulated_price, clipped)
        self._impact_cache[step] = result
        return result

    def _execute_trade(self, action: int) -> TradeExecution:
        real_price, simulated_price, agent_impact = self._compute_impact_at(
            self.current_step
        )
        target_position = self._action_to_target_position(action)
        trade_units = target_position - self.position
        trade_value = trade_units * simulated_price
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

        self.execution_step = self.current_step
        self.execution_real_price = real_price
        self.execution_price = simulated_price
        self.execution_agent_impact = agent_impact
        execution = TradeExecution(
            action=action,
            price=simulated_price,
            trade_value=trade_value,
            friction_cost=friction_cost,
        )
        self.trade_history.append(execution)
        return execution

    def _calculate_portfolio_value(self) -> float:
        real_price, simulated_price, agent_impact = self._compute_impact_at(
            self.current_step
        )
        self.real_price = real_price
        self.simulated_price = simulated_price
        self.agent_impact = agent_impact
        self.valuation_step = self.current_step
        return self.cash + (self.position * simulated_price)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        info["execution_step"] = self.execution_step
        info["valuation_step"] = self.valuation_step
        info["execution_real_price"] = self.execution_real_price
        info["execution_simulated_price"] = self.execution_price
        info["execution_agent_impact"] = self.execution_agent_impact
        info["valuation_real_price"] = self.real_price
        info["valuation_simulated_price"] = self.simulated_price
        info["valuation_agent_impact"] = self.agent_impact
        info["real_price"] = self.real_price
        info["simulated_price"] = self.simulated_price
        info["agent_impact"] = self.agent_impact
        info["execution_price"] = self.execution_price
        return obs, reward, terminated, truncated, info
