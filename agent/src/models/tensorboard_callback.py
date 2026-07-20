"""TensorBoard metrics for trading-policy training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import warnings
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


REWARD_INFO_KEYS = (
    "base_return",
    "benchmark_return",
    "benchmark_relative_reward",
    "inventory_penalty",
    "turnover_penalty",
    "drawdown_penalty",
    "downside_penalty",
)


@dataclass
class DailyMetrics:
    """Metrics attributed to one market trading date."""

    date: str
    strategy_factor: float = 1.0
    benchmark_factor: float = 1.0
    portfolio_value_close: float = 0.0
    exposures: list[float] = field(default_factory=list)
    friction_sum: float = 0.0
    action_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.int64)
    )
    forced_clear_count: int = 0
    shaped_rewards: list[float] = field(default_factory=list)
    reward_terms: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in REWARD_INFO_KEYS}
    )


class TradingMetricsTensorBoardCallback(BaseCallback):
    """Aggregate portfolio, action, cost, and reward metrics per PPO rollout."""

    def __init__(self, *, initial_cash: float, max_units: int) -> None:
        super().__init__(verbose=0)
        self.initial_cash = float(initial_cash)
        self.max_units = max(int(max_units), 1)
        self._strategy_factor = 1.0
        self._benchmark_factor = 1.0
        self._episode_strategy_factors: dict[int, float] = defaultdict(lambda: 1.0)
        self._episode_benchmark_factors: dict[int, float] = defaultdict(lambda: 1.0)
        self._total_action_counts = np.zeros(3, dtype=np.int64)
        self._total_forced_clears = 0
        self._daily_states: dict[int, DailyMetrics] = {}
        self._reset_rollout_metrics()

    def _reset_rollout_metrics(self) -> None:
        self._portfolio_values: list[float] = []
        self._exposures: list[float] = []
        self._friction_costs: list[float] = []
        self._shaped_rewards: list[float] = []
        self._reward_terms: dict[str, list[float]] = {
            key: [] for key in REWARD_INFO_KEYS
        }
        self._rollout_action_counts = np.zeros(3, dtype=np.int64)
        self._rollout_forced_clears = 0
        self._completed_episode_returns: list[float] = []
        self._completed_benchmark_returns: list[float] = []

    def _on_rollout_start(self) -> None:
        self._reset_rollout_metrics()

    def _on_step(self) -> bool:
        infos = list(self.locals.get("infos", []))
        actions = np.asarray(self.locals.get("actions", []), dtype=np.int64).reshape(-1)
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)

        for env_idx, info in enumerate(infos):
            action: int | None = None
            if env_idx < len(actions):
                action = int(actions[env_idx])
                if 0 <= action < 3:
                    self._rollout_action_counts[action] += 1
                    self._total_action_counts[action] += 1
            if env_idx < len(rewards):
                self._shaped_rewards.append(float(rewards[env_idx]))

            portfolio_value = float(info.get("portfolio_value", self.initial_cash))
            self._portfolio_values.append(portfolio_value)
            self._exposures.append(
                float(info.get("units_held", 0)) / self.max_units
            )
            self._friction_costs.append(float(info.get("friction_cost", 0.0)))
            for key in REWARD_INFO_KEYS:
                self._reward_terms[key].append(float(info.get(key, 0.0)))

            if bool(info.get("forced_clear", False)):
                self._rollout_forced_clears += 1
                self._total_forced_clears += 1

            portfolio_return = float(info.get("portfolio_return", 0.0))
            benchmark_return = float(info.get("benchmark_simple_return", 0.0))
            shaped_reward = float(rewards[env_idx]) if env_idx < len(rewards) else 0.0
            self._update_daily_metrics(
                env_idx=env_idx,
                info=info,
                action=action,
                shaped_reward=shaped_reward,
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                done=env_idx < len(dones) and bool(dones[env_idx]),
            )
            self._episode_strategy_factors[env_idx] *= max(1.0 + portfolio_return, 0.0)
            self._episode_benchmark_factors[env_idx] *= max(1.0 + benchmark_return, 0.0)

            if env_idx < len(dones) and dones[env_idx]:
                strategy_factor = self._episode_strategy_factors.pop(env_idx)
                benchmark_factor = self._episode_benchmark_factors.pop(env_idx)
                self._strategy_factor *= strategy_factor
                self._benchmark_factor *= benchmark_factor
                self._completed_episode_returns.append(strategy_factor - 1.0)
                self._completed_benchmark_returns.append(benchmark_factor - 1.0)

        return True

    def _new_daily_state(self, date: str) -> DailyMetrics:
        return DailyMetrics(date=date, portfolio_value_close=self.initial_cash)

    def _update_daily_metrics(
        self,
        *,
        env_idx: int,
        info: dict[str, Any],
        action: int | None,
        shaped_reward: float,
        portfolio_return: float,
        benchmark_return: float,
        done: bool,
    ) -> None:
        execution_date = str(
            info.get("execution_date") or info.get("valuation_date") or "unknown"
        )
        valuation_date = str(info.get("valuation_date") or execution_date)

        execution_state = self._daily_states.get(env_idx)
        if execution_state is None:
            execution_state = self._new_daily_state(execution_date)
            self._daily_states[env_idx] = execution_state
        elif execution_state.date != execution_date:
            self._flush_daily_state(execution_state)
            execution_state = self._new_daily_state(execution_date)
            self._daily_states[env_idx] = execution_state

        if action is not None and 0 <= action < 3:
            execution_state.action_counts[action] += 1
        execution_state.friction_sum += float(info.get("friction_cost", 0.0))
        if bool(info.get("forced_clear", False)):
            execution_state.forced_clear_count += 1

        if valuation_date != execution_date:
            self._flush_daily_state(execution_state)
            valuation_state = self._new_daily_state(valuation_date)
            self._daily_states[env_idx] = valuation_state
        else:
            valuation_state = execution_state

        valuation_state.strategy_factor *= max(1.0 + portfolio_return, 0.0)
        valuation_state.benchmark_factor *= max(1.0 + benchmark_return, 0.0)
        valuation_state.portfolio_value_close = float(
            info.get("portfolio_value", self.initial_cash)
        )
        valuation_state.exposures.append(
            float(info.get("units_held", 0)) / self.max_units
        )
        valuation_state.shaped_rewards.append(shaped_reward)
        for key in REWARD_INFO_KEYS:
            valuation_state.reward_terms[key].append(float(info.get(key, 0.0)))

        if done:
            self._flush_daily_state(valuation_state)
            self._daily_states.pop(env_idx, None)

    @staticmethod
    def _date_number(date: str) -> float:
        try:
            return float(int(date.replace("-", "")[:8]))
        except ValueError:
            return 0.0

    def _flush_daily_state(self, state: DailyMetrics) -> None:
        self._record("daily/date", self._date_number(state.date))
        self._record("daily/portfolio_return", state.strategy_factor - 1.0)
        self._record("daily/benchmark_return", state.benchmark_factor - 1.0)
        self._record(
            "daily/excess_return",
            state.strategy_factor / max(state.benchmark_factor, 1e-9) - 1.0,
        )
        self._record("daily/portfolio_value_close", state.portfolio_value_close)
        self._record("daily/friction_sum", state.friction_sum)
        self._record("daily/forced_clear_count", state.forced_clear_count)
        if state.exposures:
            self._record("daily/exposure_mean", float(np.mean(state.exposures)))

        action_total = int(state.action_counts.sum())
        if action_total:
            for action, label in enumerate(("hold", "add", "clear")):
                self._record(
                    f"daily/{label}_rate",
                    state.action_counts[action] / action_total,
                )
        if state.shaped_rewards:
            self._record("daily/reward_shaped_mean", float(np.mean(state.shaped_rewards)))
        for key, values in state.reward_terms.items():
            if values:
                self._record(f"daily/reward_{key}_mean", float(np.mean(values)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Tried to write empty key-value dict",
                category=UserWarning,
            )
            self.logger.dump(self.num_timesteps)

    def _record(self, name: str, value: float) -> None:
        self.logger.record(name, float(value), exclude="stdout")

    def _record_mean(self, name: str, values: list[float]) -> None:
        if values:
            self._record(name, float(np.mean(values)))

    def _on_rollout_end(self) -> None:
        self._record_mean("portfolio/value_mean", self._portfolio_values)
        if self._portfolio_values:
            self._record("portfolio/value_last", self._portfolio_values[-1])
            self._record(
                "returns/current_episode_return",
                self._portfolio_values[-1] / max(self.initial_cash, 1e-9) - 1.0,
            )
        self._record_mean("portfolio/exposure_mean", self._exposures)

        self._record_mean("cost/friction_mean", self._friction_costs)
        self._record("cost/friction_sum", sum(self._friction_costs))
        self._record("trading/forced_clear_count", self._rollout_forced_clears)
        self._record("trading/forced_clear_total", self._total_forced_clears)

        rollout_actions = int(self._rollout_action_counts.sum())
        total_actions = int(self._total_action_counts.sum())
        for action, label in enumerate(("hold", "add", "clear")):
            if rollout_actions:
                self._record(
                    f"actions/{label}_rate",
                    self._rollout_action_counts[action] / rollout_actions,
                )
            if total_actions:
                self._record(
                    f"actions/{label}_rate_total",
                    self._total_action_counts[action] / total_actions,
                )

        self._record_mean("reward/shaped_mean", self._shaped_rewards)
        for key, values in self._reward_terms.items():
            self._record_mean(f"reward/{key}_mean", values)

        self._record_mean(
            "returns/completed_episode_mean", self._completed_episode_returns
        )
        self._record_mean(
            "returns/benchmark_episode_mean", self._completed_benchmark_returns
        )
        self._record("returns/cumulative_return", self._strategy_factor - 1.0)
        self._record(
            "returns/benchmark_cumulative_return", self._benchmark_factor - 1.0
        )
        benchmark_factor = max(self._benchmark_factor, 1e-9)
        self._record(
            "returns/excess_cumulative_return",
            self._strategy_factor / benchmark_factor - 1.0,
        )

    def _on_training_end(self) -> None:
        for state in list(self._daily_states.values()):
            self._flush_daily_state(state)
        self._daily_states.clear()
