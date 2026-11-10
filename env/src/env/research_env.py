"""Explicit research contracts for action/temporal-observation ablations.

These environments are offline research models; their artifacts are deliberately
separate from the production v5 unit/serving contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium import spaces

from env.trading_env import TradingEnvironment, TradeExecution
from env.observation import adv_value_from_row, liquidity_score_from_adv
from env.reward import calculate_reward_terms

TARGET_WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ACCOUNT_FIELDS = (
    "exposure", "cash_fraction", "equity_return", "unrealized_pnl",
    "holding_duration", "time_of_day", "liquidity_pressure", "drawdown",
    "unit_occupancy",  # only the incremental policy has lots/unit capacity
)
CONTRACT_VERSION = "research-exposure-v1"


def market_window(data, columns, index, length, normalizer=None):
    """Causal intraday window; left padding is zero with an explicit valid mask.

    Reset context at each day, avoiding unknown calendar gaps and overnight
    discontinuities. This conservative first contract does not carry windows
    across days. Future rows and execution prices are never included.
    """
    timestamps = pd.to_datetime(data.Timestamp)
    day = timestamps.iloc[index].date()
    start = index
    while start > max(0, index - length + 1) and timestamps.iloc[start - 1].date() == day:
        # Unknown intraday gaps reset context instead of hiding missing time.
        if timestamps.iloc[start] - timestamps.iloc[start - 1] != pd.Timedelta(minutes=5):
            break
        start -= 1
    values = data.iloc[start:index + 1][columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("research features must be finite")
    if normalizer is not None:
        values = normalizer.transform_observation(values)
    result = np.zeros((length, len(columns)), dtype=np.float32)
    valid = np.zeros(length, dtype=np.float32)
    result[-len(values):] = values
    valid[-len(values):] = 1
    return np.concatenate((result.ravel(), valid))


class ResearchTradingEnvironment(TradingEnvironment):
    """Common observation and observed-price quantity planning for A/B/C/D."""

    def __init__(self, *args, window_length=1, normalizer=None,
                 rebalance_tolerance=0.01, **kwargs):
        if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length < 1:
            raise ValueError("window_length must be a positive integer")
        if not np.isfinite(rebalance_tolerance) or not 0 <= rebalance_tolerance < 0.1:
            raise ValueError("rebalance_tolerance must be in [0, .1)")
        super().__init__(*args, **kwargs)
        if self.exec_price_col in self.feature_columns:
            raise ValueError("execution price cannot be a research observation feature")
        if self.market_data.empty or not self.market_data.Timestamp.is_monotonic_increasing:
            raise ValueError("research market data must be non-empty and chronological")
        if self.market_data.Timestamp.duplicated().any():
            raise ValueError("duplicate research bars")
        marks = self.market_data[self.price_col].to_numpy(float)
        execution = self.market_data[self.exec_price_col].to_numpy(float)
        if not np.isfinite(marks).all() or (marks <= 0).any():
            raise ValueError("research mark prices must be finite and positive")
        if np.isinf(execution).any() or (execution[np.isfinite(execution)] <= 0).any():
            raise ValueError("research execution prices must be positive or missing")
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.window_length = window_length
        self.normalizer = normalizer
        self.rebalance_tolerance = rebalance_tolerance
        n = window_length * (len(self.feature_columns) + 1) + len(ACCOUNT_FIELDS)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
        # Precompute immutable market windows: environment stepping must not
        # repeatedly parse the entire dataframe during PPO rollouts.
        dates = pd.to_datetime(self.market_data.Timestamp)
        features = self.market_data[self.feature_columns].to_numpy(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("research features must be finite")
        if normalizer is not None:
            features = normalizer.transform_observation(features)
        self._market_windows = np.zeros((len(features), window_length * (len(self.feature_columns) + 1)), np.float32)
        start = 0
        for i in range(len(features)):
            if i and (dates.iloc[i].date() != dates.iloc[i - 1].date()
                      or dates.iloc[i] - dates.iloc[i - 1] != pd.Timedelta(minutes=5)):
                start = i
            lo = max(start, i - window_length + 1)
            count = i - lo + 1
            window = self._market_windows[i, :window_length * len(self.feature_columns)].reshape(window_length, -1)
            window[-count:] = features[lo:i + 1]
            self._market_windows[i, -count:] = 1

    @property
    def action_schema(self):
        return "incremental-4"

    def contract(self):
        return {"version": CONTRACT_VERSION, "action_schema": self.action_schema,
                "n_actions": int(self.action_space.n), "window_length": self.window_length,
                "window_policy": "intraday-reset-zero-pad-with-validity",
                "feature_columns": self.feature_columns,
                "account_fields": list(ACCOUNT_FIELDS),
                "observation_shape": list(self.observation_space.shape),
                "target_weights": list(TARGET_WEIGHTS) if self.action_schema == "target-weight-6" else None,
                "quantity_planning": "observable-close; cash-limited partial fill at ExecPrice",
                "missing_execution": "no fill, no cost; never substitute same-bar close",
                "rebalance_tolerance": self.rebalance_tolerance,
                "sizing_basis": "initial-cash-unit" if self.action_schema == "incremental-4" else "current-marked-equity",
                "terminal_settlement": "current-observed-close-plus-sell-friction"}

    def _get_observation(self):
        row = self._row_at(self.current_step)
        price = float(row[self.price_col])
        equity = max(self.cash + self.shares_held * price, 1e-9)
        held = self.shares_held * price
        adv = adv_value_from_row(row)
        liquidity = 0.0 if adv is None or adv <= 0 else np.clip(
            (np.log10(self.initial_cash * self.unit_fraction / adv) + 4) / 4, -1, 1)
        age = 0 if self.entry_step is None else self.current_step - self.entry_step
        ts = pd.Timestamp(row.Timestamp)
        # Fixed clock semantics also handle delayed-open days without using
        # future information about how many bars today will eventually contain.
        tod = np.clip((ts.hour * 60 + ts.minute - 9 * 60) / 380, 0, 1)
        state = np.asarray([
            held / equity, self.cash / equity, equity / self.initial_cash - 1,
            (held - self.cost_basis) / self.initial_cash,
            min(age / self.duration_horizon_bars, 1), tod, liquidity,
            max(0, 1 - equity / max(self.peak_portfolio_value, equity)),
            self.units_held / self.max_units,
        ], dtype=np.float32)
        index = int(self._active_index[self.current_step])
        return np.concatenate((self._market_windows[index], state))

    def _friction(self, quantity, price):
        value = abs(quantity) * price
        return self.friction_model.calculate_total_friction(
            trade_value=value, side="buy" if quantity > 0 else "sell",
            liquidity_score=self._current_liquidity_score(value), price=price,
            trade_date=pd.Timestamp(self._row_at(self.current_step).Timestamp).date()) if quantity else 0.0

    def _fill(self, requested):
        """Only reduce planned buy size at fill-time; never exploit future prices
        to increase the quantity. Execution is an idealized cash-limited fill.
        """
        row = self._row_at(self.current_step)
        price = float(row[self.exec_price_col])
        if not np.isfinite(price):
            # A missing auction/next-minute print is an unavailable fill, not
            # permission to transact at the earlier observation price.
            return 0, float(row[self.price_col]), 0.0
        quantity = int(requested)
        if quantity > 0:
            lo, hi = 0, min(quantity, int(self.cash // price))
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if mid * price + self._friction(mid, price) <= self.cash:
                    lo = mid
                else:
                    hi = mid - 1
            quantity = lo
        else:
            quantity = max(quantity, -int(self.shares_held))
        cost = self._friction(quantity, price)
        old_shares = self.shares_held
        self.cash -= quantity * price + cost
        self.shares_held += quantity
        if quantity > 0:
            self.cost_basis += quantity * price
            if old_shares == 0:
                self.entry_step = self.current_step
        elif quantity < 0:
            self.cost_basis *= self.shares_held / old_shares
        if self.shares_held == 0:
            self.cost_basis = 0.0
            self.entry_step = None
        if self.cash < -1e-6:
            raise RuntimeError("negative research cash")
        return quantity, price, cost

    def _execute_trade(self, action):
        # Plan at the observable Close; base legacy env sizes with ExecPrice.
        price = float(self._row_at(self.current_step)[self.price_col])
        requested = 0
        if action == 1 and self.units_held < self.max_units and self.action_masks()[1]:
            requested = int(self.initial_cash * self.unit_fraction // price)
        elif action == 2 and self._share_lots:
            requested = -self._share_lots[-1]
        elif action == 3:
            requested = -int(self.shares_held)
        previous_basis = self.cost_basis
        quantity, fill_price, cost = self._fill(requested)
        if quantity > 0:
            self._share_lots.append(quantity)
            self._lot_costs.append(quantity * fill_price)
        elif quantity < 0:
            if action == 3:
                self._share_lots.clear()
                self._lot_costs.clear()
            else:
                self._share_lots.pop()
                self._lot_costs.pop()
            # Preserve the legacy LIFO lot convention for incremental actions.
            self.cost_basis = float(sum(self._lot_costs))
        else:
            self.cost_basis = previous_basis
        self.units_held = len(self._share_lots)
        execution = TradeExecution(action, fill_price, quantity * fill_price, cost)
        self.trade_history.append(execution)
        return execution

    def _calculate_reward(self, **kwargs):
        # Identical exposure-based penalties for both action representations.
        price = float(self._row_at(self.current_step)[self.price_col])
        equity = max(self.cash + self.shares_held * price, 1e-9)
        return calculate_reward_terms(
            **kwargs, units_held=self.shares_held * price / equity,
            max_units=1, config=self.reward_config)

    def step(self, action):
        if self._active_index is None or self.current_step >= len(self._active_index) - 1:
            raise RuntimeError("reset before stepping a finished research episode")
        unavailable = not np.isfinite(float(self._row_at(self.current_step)[self.exec_price_col]))
        obs, reward, terminated, truncated, info = super().step(action)
        held = info["held_market_value"]
        info.update({"cash": self.cash, "shares": self.shares_held,
                     "cost_basis": self.cost_basis, "flat": self.shares_held == 0,
                     "exposure": held / max(info["portfolio_value"], 1e-9),
                     "fill_unavailable": unavailable,
                     "requested_action": int(action), "executed": info["trade_value"] != 0})
        return obs, reward, terminated, truncated, info

    def estimate_terminal_settlement_adjustment(self):
        # Settle at last observed mark, without peeking beyond split boundary.
        price = float(self._row_at(self.current_step)[self.price_col])
        return self._friction(-int(self.shares_held), price)


class TargetExposureEnvironment(ResearchTradingEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(len(TARGET_WEIGHTS))

    @property
    def action_schema(self):
        return "target-weight-6"

    def action_masks(self):
        # All bounded target intents are valid. Cash-limited execution is part
        # of the contract, so even 100% remains feasible after paying friction.
        return np.ones(len(TARGET_WEIGHTS), dtype=bool)

    def _execute_trade(self, action):
        row = self._row_at(self.current_step)
        price = float(row[self.price_col])
        equity = self.cash + self.shares_held * price
        target = TARGET_WEIGHTS[action]
        exposure = self.shares_held * price / max(equity, 1e-9)
        desired = int(target * equity // price)
        requested = desired - int(self.shares_held)
        if target != 0 and abs(target - exposure) <= self.rebalance_tolerance:
            requested = 0
        quantity, fill_price, cost = self._fill(requested)
        # Units/lots are intentionally unused in this distinct contract.
        execution = TradeExecution(action, fill_price, quantity * fill_price, cost)
        self.trade_history.append(execution)
        return execution
