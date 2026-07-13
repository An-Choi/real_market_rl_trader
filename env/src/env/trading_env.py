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
        duration_horizon_bars: int | None = None,
        nominal_bars_per_day: int = 64,
        feature_schema_version: int | None = None,
    ) -> None:
        """Real-OHLCV intraday Unit-scaling environment.

        Discrete actions: 0=Hold, 1=Add 1 Unit, 2=Clear.
        feature_schema_version: feature 계산 파이프라인의 schema version 선언 — artifact 호환성 검증용.
        """
        super().__init__()
        if feature_schema_version is not None and (
            isinstance(feature_schema_version, bool)
            or not isinstance(feature_schema_version, int)
            or feature_schema_version <= 0
        ):
            raise ValueError(
                f"feature_schema_version must be a positive int or None: {feature_schema_version!r}"
            )
        self.feature_schema_version = feature_schema_version
        self.market_data = market_data.reset_index(drop=True)
        ts = pd.to_datetime(self.market_data["Timestamp"])
        self._date_groups = {
            d: np.asarray(grp)
            for d, grp in self.market_data.groupby(ts.dt.date).groups.items()
        }
        self._available_dates = sorted(self._date_groups.keys())
        if isinstance(episode_days, bool) or not isinstance(episode_days, int) or episode_days <= 0:
            raise ValueError(f"episode_days must be a positive integer: {episode_days!r}")
        self.episode_days = episode_days
        if (
            isinstance(nominal_bars_per_day, bool)
            or not isinstance(nominal_bars_per_day, int)
            or nominal_bars_per_day <= 0
        ):
            raise ValueError(
                f"nominal_bars_per_day must be a positive int: {nominal_bars_per_day!r}"
            )
        self.nominal_bars_per_day = nominal_bars_per_day
        if duration_horizon_bars is None:
            duration_horizon_bars = episode_days * self.nominal_bars_per_day
        if (
            isinstance(duration_horizon_bars, bool)
            or not isinstance(duration_horizon_bars, int)
            or duration_horizon_bars <= 0
        ):
            raise ValueError(
                f"duration_horizon_bars must be a positive int: {duration_horizon_bars!r}"
            )
        self.duration_horizon_bars = duration_horizon_bars
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
            raise ValueError(f"episode_days must be a positive integer: {episode_days!r}")

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
            # 데이터가 episode_days보다 짧으면 가용 전체로 truncate (짧은 fixture 허용)
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
            "end_date": str(self._episode_dates[-1]),
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
        """bar t에서 실행 → t+1로 전진해 평가·관측. 마지막 bar는 valuation 전용.

        B개 bar짜리 episode는 B−1 step이며 마지막 step은 truncated=True를
        반환한다(terminated 아님 — continuing task, SB3가 V(s)로 bootstrap).
        강제청산·정산은 env에 없다.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_value = self.portfolio_value
        previous_peak_value = self.peak_portfolio_value
        execution_row = self._row_at(self.current_step)
        previous_market_price = float(execution_row[self.price_col])
        execution_timestamp = pd.Timestamp(execution_row["Timestamp"])
        execution = self._execute_trade(action)

        self.current_step += 1
        self.portfolio_value = self._calculate_portfolio_value()
        valuation_row = self._row_at(self.current_step)
        current_market_price = float(valuation_row[self.price_col])
        valuation_timestamp = pd.Timestamp(valuation_row["Timestamp"])
        reward_terms = self._calculate_reward(
            previous_value=previous_value,
            current_value=self.portfolio_value,
            previous_peak_value=previous_peak_value,
            previous_market_price=previous_market_price,
            current_market_price=current_market_price,
            trade_value=execution.trade_value,
        )
        self.peak_portfolio_value = max(previous_peak_value, self.portfolio_value)

        truncated = self.current_step >= len(self._active_index) - 1
        info = {
            "portfolio_value": self.portfolio_value,
            "units_held": self.units_held,
            "friction_cost": execution.friction_cost,
            "trade_value": execution.trade_value,
            "execution_date": str(execution_timestamp.date()),
            "valuation_date": str(valuation_timestamp.date()),
            "valuation_timestamp": valuation_row["Timestamp"],
            "valuation_price": current_market_price,
            "held_market_value": self._held_market_value(current_market_price),
            "portfolio_return": reward_terms.portfolio_return,
            "base_return": reward_terms.base_return,
            "benchmark_return": reward_terms.benchmark_return,
            "benchmark_simple_return": reward_terms.benchmark_simple_return,
            "benchmark_relative_reward": reward_terms.benchmark_relative_reward,
            "inventory_penalty": reward_terms.inventory_penalty,
            "turnover_penalty": reward_terms.turnover_penalty,
            "drawdown_penalty": reward_terms.drawdown_penalty,
            "downside_penalty": reward_terms.downside_penalty,
        }
        return self._get_observation(), reward_terms.reward, False, truncated, info

    def estimate_liquidation_cost(self) -> float:
        """현재 보유분을 현재 bar에 가상 매도할 때의 friction (metrics 정산용).

        실제 거래·reward에는 쓰지 않는다 — 평가 회계 전용.
        """
        price = float(self._row_at(self.current_step)[self.price_col])
        held_value = self._held_market_value(price)
        if held_value <= 0:
            return 0.0
        return self.friction_model.calculate_total_friction(
            trade_value=held_value,
            side="sell",
            liquidity_score=self._current_liquidity_score(),
        )

    def _get_observation(self) -> np.ndarray:
        """Observation = features + Unit portfolio state (4).

        holding_duration_norm은 고정 horizon(duration_horizon_bars) 기준 —
        runtime episode 길이와 무관해 학습/평가/서빙에서 동일 스케일(causal).
        tod_frac은 당일 기준, 분모는 그날 실제 bar 수가 아닌 고정 상수
        nominal_bars_per_day — 당일 뒤쪽 bar 결손 여부(미래 정보)에 의존하지
        않는다(causal).
        """
        row = self._row_at(self.current_step)
        features = (
            pd.to_numeric(row[self.feature_columns], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        price = float(row[self.price_col])
        units_held_frac = self.units_held / max(self.max_units, 1)
        held_value = self._held_market_value(price)
        cost_basis = self.units_held * self.initial_cash * self.unit_fraction
        unrealized_pnl_norm = (held_value - cost_basis) / max(self.initial_cash, 1e-9)
        if self.units_held > 0 and self.entry_step is not None:
            holding_duration_norm = min(
                (self.current_step - self.entry_step) / self.duration_horizon_bars, 1.0
            )
        else:
            holding_duration_norm = 0.0

        day_idx = int(
            np.searchsorted(self._day_start_offsets, self.current_step, side="right")
        ) - 1
        day_start = int(self._day_start_offsets[day_idx])
        tod_frac = min(
            (self.current_step - day_start) / max(self.nominal_bars_per_day - 1, 1), 1.0
        )

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
