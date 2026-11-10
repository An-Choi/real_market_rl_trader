"""Finite-horizon, duration-aware target-exposure research environment."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from env.research_env import TargetExposureEnvironment, TradeExecution
from env.reward import calculate_reward_terms


class ControlledExposureEnvironment(TargetExposureEnvironment):
    """One target decision, followed by passive holding for up to k bars.

    Rewards are discounted *within* the decision. The paired rollout buffer
    applies gamma**duration to the next state. The final observed mark is
    liquidated with friction during training as well as evaluation.
    """

    def __init__(self, *args, decision_interval=1, bar_gamma=.999,
                 drawdown_threshold=.02, risk_budget=.001,
                 risk_multiplier=0., reserve_exit_cost=True, settle_daily=False, **kwargs):
        if isinstance(decision_interval, bool) or not isinstance(decision_interval, int) or decision_interval < 1:
            raise ValueError("decision_interval must be a positive integer")
        if not np.isfinite(bar_gamma) or not 0 < bar_gamma <= 1:
            raise ValueError("bar_gamma must be in (0, 1]")
        if not np.isfinite(drawdown_threshold) or not 0 <= drawdown_threshold < 1:
            raise ValueError("drawdown_threshold must be in [0, 1)")
        if not np.isfinite(risk_budget) or not 0 <= risk_budget < 1:
            raise ValueError("risk_budget must be in [0, 1)")
        self.decision_interval = decision_interval
        if not isinstance(reserve_exit_cost, bool):
            raise ValueError("reserve_exit_cost must be a boolean")
        self.reserve_exit_cost = reserve_exit_cost
        if not isinstance(settle_daily, bool):
            raise ValueError("settle_daily must be a boolean")
        self.settle_daily = settle_daily
        self.bar_gamma = bar_gamma
        self.drawdown_threshold, self.risk_budget = drawdown_threshold, risk_budget
        self.set_risk_multiplier(risk_multiplier)
        self._allow_rebalance = True
        super().__init__(*args, **kwargs)
        self._liquidation_value = self._liquidation_peak = self.initial_cash
        self.observation_space = spaces.Box(-np.inf, np.inf,
            (self.observation_space.shape[0] + 4,), np.float32)

    def reset(self, *args, **kwargs):
        self._liquidation_value = self._liquidation_peak = self.initial_cash
        return super().reset(*args, **kwargs)

    def set_risk_multiplier(self, value):
        if not np.isfinite(value) or not 0 <= value <= 10:
            raise ValueError("risk_multiplier must be in [0, 10]")
        self.risk_multiplier = float(value)

    def contract(self):
        contract = super().contract()
        contract.update(version="controlled-exposure-v1",
            decision_interval=self.decision_interval, bar_gamma=self.bar_gamma,
            account_fields=contract["account_fields"] + ["episode_progress", "risk_multiplier_div_10",
                "liquidation_reserve_fraction", "liquidation_drawdown"],
            transition="one rebalance then passive hold; stop at intraday context discontinuity",
            reward_valuation="liquidation-NAV" if self.reserve_exit_cost else "marked-NAV",
            reward="discounted per-bar NAV return minus multiplier * drawdown excess; scaled",
            risk_cost="max(liquidation-NAV drawdown - drawdown_threshold, 0) per valuation bar",
            risk_constraint="per-episode gamma-weighted mean excess, estimated from training rollouts",
            drawdown_threshold=self.drawdown_threshold, risk_budget=self.risk_budget,
            episode_days=self.episode_days,
            evaluation_accounting="train-matched finite episodes; compound settled cash between episodes",
            terminal_settlement="last observed close plus sell friction; absorbing finite horizon")
        # The adaptive multiplier is state, not an immutable environment rule.
        if self.settle_daily:
            contract.update(daily_settlement=True,
                risk_peak="persists across daily cash settlements within the multi-day episode",
                overnight="flat cash; no new entry on the final bar of a session")
        return contract

    def _session_boundary(self):
        return (self._active_index is not None and self.current_step < len(self._active_index)-1
            and self._row_at(self.current_step).Timestamp.date() != self._row_at(self.current_step+1).Timestamp.date())

    def action_masks(self):
        masks = super().action_masks()
        if self.settle_daily and self._session_boundary():
            masks[:] = False
            masks[0] = True
        return masks

    def _get_observation(self):
        base = super()._get_observation()
        progress = self.current_step / max(len(self._active_index) - 1, 1)
        reserve = self.estimate_terminal_settlement_adjustment()
        nav = self.portfolio_value - reserve
        peak = max(getattr(self, "_liquidation_peak", self.initial_cash), nav)
        return np.concatenate((base, np.asarray([progress, self.risk_multiplier / 10,
            reserve / max(self.portfolio_value, 1e-9), max(0., 1 - nav / max(peak, 1e-9))], np.float32)))

    def _execute_trade(self, action):
        if self._allow_rebalance and not (self.settle_daily and self._session_boundary()):
            return super()._execute_trade(action)
        price = float(self._row_at(self.current_step)[self.price_col])
        execution = TradeExecution(action, price, 0., 0.)
        self.trade_history.append(execution)
        return execution

    def step(self, action, *, rebalance=True):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        records, reward, discounted_risk = [], 0., 0.
        for offset in range(self.decision_interval):
            previous_price = float(self._row_at(self.current_step)[self.price_col])
            previous_nav, previous_nav_peak = self._liquidation_value, self._liquidation_peak
            previous_marked, previous_marked_peak = self.portfolio_value, self.peak_portfolio_value
            self._allow_rebalance = bool(rebalance) and offset == 0
            _, bar_reward, _, done, info = super().step(action)
            info["absolute_trade_value"] = abs(info["trade_value"])
            info["executed_trades"] = int(info["executed"])
            info["terminal_settlement_cost"] = 0.
            daily_close = self.settle_daily and self._session_boundary()
            if done or daily_close:
                # Training cannot avoid the closing transaction cost by buying
                # just before a time limit or bootstrapping beyond this episode.
                before = self.portfolio_value
                closing_value = self.shares_held * info["valuation_price"]
                closing_cost = self.estimate_terminal_settlement_adjustment()
                self.cash = before - closing_cost
                self.shares_held = self.cost_basis = 0.
                self.entry_step = None
                self.portfolio_value = self.cash
                info["portfolio_return"] = (1 + info["portfolio_return"]) * self.cash / before - 1
                info.update(portfolio_value=self.cash, cash=self.cash, shares=0.,
                            cost_basis=0., held_market_value=0., flat=True, exposure=0.,
                            friction_cost=info["friction_cost"] + closing_cost,
                            terminal_settlement_cost=closing_cost,
                            absolute_trade_value=info["absolute_trade_value"] + closing_value,
                            executed_trades=info["executed_trades"] + int(closing_value > 0))
            # Reserve the observable cost of closing the current position now.
            # Buying therefore recognizes both entry and eventual exit costs
            # immediately, without charging that exit cost twice at settlement.
            nav = self.portfolio_value - self.estimate_terminal_settlement_adjustment()
            terms = calculate_reward_terms(
                previous_value=previous_nav if self.reserve_exit_cost else previous_marked,
                current_value=nav if self.reserve_exit_cost else self.portfolio_value,
                previous_peak_value=previous_nav_peak if self.reserve_exit_cost else previous_marked_peak,
                previous_market_price=previous_price,
                current_market_price=info["valuation_price"],
                units_held=info["exposure"], max_units=1,
                trade_value=info["absolute_trade_value"], config=self.reward_config)
            bar_reward = terms.reward
            self._liquidation_value = nav
            self._liquidation_peak = max(previous_nav_peak, nav)
            drawdown = max(0., 1 - nav / max(self._liquidation_peak, 1e-9))
            risk = max(0., drawdown - self.drawdown_threshold)
            penalty = self.reward_config.reward_scale * self.risk_multiplier * risk
            bar_reward -= penalty
            info.update(risk_cost=risk, drawdown=drawdown,
                        risk_discount_weight=self.bar_gamma ** (self.current_step - 1),
                        liquidation_value=nav, marked_base_return=info["base_return"],
                        base_return=terms.base_return,
                        risk_penalty=penalty, risk_multiplier=self.risk_multiplier,
                        bar_reward=float(bar_reward), decision_start=offset == 0,
                        decision_rebalance=bool(rebalance) and offset == 0)
            records.append(info)
            reward += self.bar_gamma ** offset * bar_reward
            discounted_risk += self.bar_gamma ** offset * risk
            if done or daily_close:
                break
            # Resume decisions at the next observed day/gap; never retain a
            # hidden pending rebalance across unknown intraday discontinuities.
            current = self._row_at(self.current_step).Timestamp
            previous = self._row_at(self.current_step - 1).Timestamp
            if current.date() != previous.date() or (current - previous).total_seconds() != 300:
                break
        self._allow_rebalance = True
        aggregate = dict(records[-1])
        aggregate.update(bar_records=records, elapsed_bars=len(records),
                         risk_cost_sum=sum(r["risk_cost"] for r in records),
                         weighted_risk_cost_sum=sum(r["risk_cost"] * r["risk_discount_weight"] for r in records),
                         risk_discount_weight_sum=sum(r["risk_discount_weight"] for r in records),
                         discounted_risk_cost=discounted_risk)
        # Absorbing horizon: SB3 must not time-limit-bootstrap after liquidation.
        return self._get_observation(), float(reward), bool(done), False, aggregate
