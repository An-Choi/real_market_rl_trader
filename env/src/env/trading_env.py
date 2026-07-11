"""Gymnasium-style trading environment skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from friction.friction_model import FrictionModel
from env.reward import RewardConfig, RewardTerms, calculate_reward_terms


ACTION_HOLD = 0
ACTION_ADD = 1
ACTION_CLEAR = 2


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
        unit_fraction: float = 0.20,
        max_units: int = 5,
        friction_model: FrictionModel | None = None,
        risk_penalty_rate: float = 0.0,
        turnover_penalty_rate: float = 0.0,
        drawdown_penalty_rate: float = 0.0,
        downside_penalty_rate: float = 0.0,
        benchmark_relative_rate: float = 0.0,
        reward_scale: float = 1.0,
        reward_return_mode: str = "simple_return",
        episode_days: int = 1,
    ) -> None:
        """Real-OHLCV intraday Unit-scaling environment.

        Discrete actions: 0=Hold, 1=Add 1 Unit, 2=Clear.
        """
        super().__init__()
        self.market_data = market_data.reset_index(drop=True)
        ts = pd.to_datetime(self.market_data["Timestamp"])
        self._date_groups = {
            d: np.asarray(grp)
            for d, grp in self.market_data.groupby(ts.dt.date).groups.items()
        }
        self._available_dates = sorted(self._date_groups.keys())
        if isinstance(episode_days, bool) or not isinstance(episode_days, int) or episode_days <= 0:
            raise ValueError(f"episode_days must be a positive int: {episode_days!r}")
        self.episode_days = episode_days
        self._active_date = None
        self._episode_dates: list = []
        self._day_start_offsets: np.ndarray | None = None
        self._active_index: np.ndarray | None = None
        self.feature_columns = feature_columns
        self.price_col = price_col
        self.initial_cash = initial_cash
        self.unit_fraction = unit_fraction
        self.max_units = max_units
        self.friction_model = friction_model or FrictionModel()
        self.risk_penalty_rate = risk_penalty_rate
        self.reward_config = RewardConfig(
            inventory_penalty_rate=risk_penalty_rate,
            turnover_penalty_rate=turnover_penalty_rate,
            drawdown_penalty_rate=drawdown_penalty_rate,
            downside_penalty_rate=downside_penalty_rate,
            benchmark_relative_rate=benchmark_relative_rate,
            reward_scale=reward_scale,
            return_mode=reward_return_mode,
        )

        self.action_space = spaces.Discrete(3)
        observation_size = len(self.feature_columns) + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.cash = self.initial_cash
        self.units_held = 0
        self.shares_held = 0.0
        self.entry_step: int | None = None
        self.entry_price: float | None = None
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.trade_history: list[TradeExecution] = []

    def reset(self, seed: int | None = None, options: dict | None = None):
        """연속 episode_days 거래일 윈도우의 시작으로 reset한다."""
        super().reset(seed=seed)
        options = options or {}
        episode_days = options.get("episode_days", self.episode_days)
        if isinstance(episode_days, bool) or not isinstance(episode_days, int) or episode_days <= 0:
            raise ValueError(f"episode_days must be a positive int: {episode_days!r}")

        start = options.get("start_date", options.get("date"))
        n_dates = len(self._available_dates)
        if start is not None:
            if isinstance(start, str):
                start = pd.Timestamp(start).date()
            if start not in self._date_groups:
                raise ValueError(f"unknown trading date: {start}")
            start_idx = self._available_dates.index(start)
            # 명시 지정은 남은 일수로 truncate 수용
            effective_days = min(episode_days, n_dates - start_idx)
        else:
            effective_days = min(episode_days, n_dates)
            last_start = n_dates - effective_days
            start_idx = int(self.np_random.integers(0, last_start + 1))

        self._episode_dates = self._available_dates[start_idx:start_idx + effective_days]
        self._active_date = self._episode_dates[0]
        index_parts = [self._date_groups[d] for d in self._episode_dates]
        self._active_index = np.concatenate(index_parts)
        lengths = [len(part) for part in index_parts]
        self._day_start_offsets = np.cumsum([0] + lengths[:-1])
        if len(self._active_index) < 2:
            raise ValueError(
                f"episode window needs at least 2 bars, got {len(self._active_index)}"
            )

        self.current_step = 0
        self.cash = self.initial_cash
        self.units_held = 0
        self.shares_held = 0.0
        self.entry_step = None
        self.entry_price = None
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.trade_history = []
        return self._get_observation(), {
            "portfolio_value": self.portfolio_value,
            "units_held": self.units_held,
            "date": str(self._active_date),
            "episode_days": effective_days,
            "initial_market_price": float(self._row_at(0)[self.price_col]),
        }

    @property
    def available_dates(self) -> tuple:
        """Trading dates available for deterministic full-split evaluation."""
        return tuple(self._available_dates)

    def _row_at(self, local_step: int) -> pd.Series:
        """Map a local (within-day) step to the global market row."""
        global_idx = int(self._active_index[local_step])
        return self.market_data.iloc[global_idx]

    def step(self, action: int):
        """Advance one step; force-clear and terminate at the day's last bar."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        n = len(self._active_index)
        is_last = self.current_step >= n - 1
        forced_clear = False

        previous_value = self.portfolio_value
        previous_peak_value = self.peak_portfolio_value
        previous_market_price = float(self._row_at(self.current_step)[self.price_col])
        if is_last:
            # 마감 강제청산: agent action(Add/Hold)을 무시하고 무조건 Clear한다.
            # 보유분이 없으면 Clear는 no-op이라 안전하며, 새 진입도 막힌다.
            forced_clear = self.units_held > 0
            execution = self._execute_trade(ACTION_CLEAR)
        else:
            execution = self._execute_trade(action)

        if not is_last:
            self.current_step += 1
        self.portfolio_value = self._calculate_portfolio_value()
        current_market_price = float(self._row_at(self.current_step)[self.price_col])
        reward_terms = self._calculate_reward(
            previous_value=previous_value,
            current_value=self.portfolio_value,
            previous_peak_value=previous_peak_value,
            previous_market_price=previous_market_price,
            current_market_price=current_market_price,
            trade_value=execution.trade_value,
        )
        self.peak_portfolio_value = max(previous_peak_value, self.portfolio_value)
        reward = reward_terms.reward

        terminated = is_last
        truncated = False
        info = {
            "portfolio_value": self.portfolio_value,
            "units_held": self.units_held,
            "friction_cost": execution.friction_cost,
            "trade_value": execution.trade_value,
            "forced_clear": forced_clear,
            "portfolio_return": reward_terms.portfolio_return,
            "base_return": reward_terms.base_return,
            "benchmark_return": reward_terms.benchmark_return,
            "benchmark_relative_reward": reward_terms.benchmark_relative_reward,
            "inventory_penalty": reward_terms.inventory_penalty,
            "turnover_penalty": reward_terms.turnover_penalty,
            "drawdown_penalty": reward_terms.drawdown_penalty,
            "downside_penalty": reward_terms.downside_penalty,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Observation = features + Unit portfolio state (4)."""
        row = self._row_at(self.current_step)
        features = (
            pd.to_numeric(row[self.feature_columns], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        price = float(row[self.price_col])
        n = len(self._active_index)
        units_held_frac = self.units_held / max(self.max_units, 1)
        held_value = self._held_market_value(price)
        cost_basis = self.units_held * self.initial_cash * self.unit_fraction
        unrealized_pnl_norm = (held_value - cost_basis) / max(self.initial_cash, 1e-9)
        if self.units_held > 0 and self.entry_step is not None:
            holding_duration_norm = (self.current_step - self.entry_step) / max(n - 1, 1)
        else:
            holding_duration_norm = 0.0
        tod_frac = self.current_step / max(n - 1, 1)
        portfolio_state = np.array(
            [units_held_frac, unrealized_pnl_norm, holding_duration_norm, tod_frac],
            dtype=np.float32,
        )
        return np.concatenate([features, portfolio_state]).astype(np.float32)

    def _calculate_reward(
        self,
        previous_value: float,
        current_value: float,
        previous_peak_value: float,
        previous_market_price: float,
        current_market_price: float,
        trade_value: float,
    ) -> RewardTerms:
        """Calculate reward as cost-deducted realized return minus risk penalty.

        거래비용은 _execute_trade에서 이미 cash(→ portfolio_value)에 차감되므로
        portfolio_return 자체가 비용 반영 후 수익률이다. friction을 reward에서
        다시 빼면 이중 차감이 되므로 friction_cost는 별도 패널티로 쓰지 않는다.
        """
        return calculate_reward_terms(
            previous_value=previous_value,
            current_value=current_value,
            previous_peak_value=previous_peak_value,
            previous_market_price=previous_market_price,
            current_market_price=current_market_price,
            units_held=self.units_held,
            max_units=self.max_units,
            trade_value=trade_value,
            config=self.reward_config,
        )

    def _execute_trade(self, action: int) -> TradeExecution:
        """Execute a Unit-scaling trade at the current real Close.

        보유분은 실제 주식수(`shares_held`)로 추적한다. 매수는 1 Unit = 고정
        notional만큼 현금을 쓰고 `notional/price`주를 취득, 매도는 보유 주식을
        현재가로 현금화한다. 거래비용(거래세 포함)은 실제 거래 현금흐름
        (`trade_value`) 기준으로 부과된다.
        """
        price = float(self._row_at(self.current_step)[self.price_col])
        target_units = self._action_to_target_units(action)
        unit_notional = self.initial_cash * self.unit_fraction
        prev_units = self.units_held
        delta_units = target_units - prev_units

        if delta_units > 0:
            # 매수: 고정 notional 지출, shares 누적.
            trade_value = delta_units * unit_notional
            side = "buy"
            self.cash -= trade_value
            self.shares_held += trade_value / price
            if self.entry_step is None:
                self.entry_step = self.current_step
        elif delta_units < 0:
            # 매도(Clear): 보유 주식을 현재가로 현금화. (long-only라 항상 전량 청산.)
            shares_sold = self.shares_held
            trade_value = -shares_sold * price
            side = "sell"
            self.cash += shares_sold * price
            self.shares_held = 0.0
        else:
            trade_value = 0.0
            side = "buy"

        liquidity_score = self._current_liquidity_score()
        friction_cost = (
            self.friction_model.calculate_total_friction(
                trade_value=trade_value,
                side=side,
                liquidity_score=liquidity_score,
            )
            if delta_units != 0
            else 0.0
        )
        self.cash -= friction_cost
        self.units_held = target_units
        if target_units == 0:
            self.entry_step = None

        execution = TradeExecution(
            action=action,
            price=price,
            trade_value=trade_value,
            friction_cost=friction_cost,
        )
        self.trade_history.append(execution)
        return execution

    def _action_to_target_units(self, action: int) -> int:
        """Map discrete action to target Unit count (long-only, 0..max_units)."""
        if action == ACTION_ADD:
            return min(self.units_held + 1, self.max_units)
        if action == ACTION_CLEAR:
            return 0
        return self.units_held

    def _held_market_value(self, price: float) -> float:
        """Current market value of held shares (actual shares * current price)."""
        return self.shares_held * price

    def _calculate_portfolio_value(self) -> float:
        """Mark-to-market: cash + held notional scaled by price change."""
        price = float(self._row_at(self.current_step)[self.price_col])
        return self.cash + self._held_market_value(price)

    def _current_liquidity_score(self) -> float | None:
        """Return current liquidity score when available."""
        if "liquidity_score" not in self.market_data.columns:
            return None
        return float(self._row_at(self.current_step)["liquidity_score"])

    def get_current_market_row(self) -> pd.Series:
        """Current market row within the active trading day."""
        return self._row_at(self.current_step)
