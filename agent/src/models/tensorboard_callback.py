"""Compact TensorBoard metrics for trading-policy training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


REWARD_INFO_KEYS = (
    "base_return",
    "benchmark_relative_reward",
    "inventory_penalty",
    "turnover_penalty",
    "drawdown_penalty",
    "downside_penalty",
)


def active_reward_info_keys(environment_config: dict[str, Any]) -> tuple[str, ...]:
    """Return only reward terms that can be nonzero under the current config."""
    keys = ["base_return"]
    rate_by_key = {
        "benchmark_relative_reward": "benchmark_relative_rate",
        "inventory_penalty": "risk_penalty_rate",
        "turnover_penalty": "turnover_penalty_rate",
        "drawdown_penalty": "drawdown_penalty_rate",
        "downside_penalty": "downside_penalty_rate",
    }
    keys.extend(
        key
        for key, rate_name in rate_by_key.items()
        if float(environment_config.get(rate_name, 0.0)) != 0.0
    )
    return tuple(keys)


class TradingMetricsTensorBoardCallback(BaseCallback):
    """Record one compact decision dashboard per PPO rollout.

    SB3 already logs optimizer and timing metrics. This callback adds only the
    trading metrics needed to judge a run: return versus market, exposure,
    action mix, friction, and the reward terms that are actually enabled.
    """

    def __init__(
        self,
        *,
        initial_cash: float,
        max_units: int,
        reward_info_keys: Iterable[str] = ("base_return",),
    ) -> None:
        super().__init__(verbose=0)
        self.initial_cash = float(initial_cash)
        self.max_units = max(int(max_units), 1)
        requested = tuple(reward_info_keys)
        unknown = sorted(set(requested).difference(REWARD_INFO_KEYS))
        if unknown:
            raise ValueError(f"unknown reward info keys: {unknown}")
        self.reward_info_keys = requested
        self._episode_strategy_factors: dict[int, float] = defaultdict(lambda: 1.0)
        self._episode_benchmark_factors: dict[int, float] = defaultdict(lambda: 1.0)
        self._reset_rollout_metrics()

    def _reset_rollout_metrics(self) -> None:
        self._exposures: list[float] = []
        self._friction_costs: list[float] = []
        self._shaped_rewards: list[float] = []
        self._reward_terms = {key: [] for key in self.reward_info_keys}
        self._action_counts = np.zeros(3, dtype=np.int64)
        self._forced_clears = 0
        self._completed_episode_returns: list[float] = []
        self._completed_benchmark_returns: list[float] = []
        self._completed_excess_returns: list[float] = []

    def _on_rollout_start(self) -> None:
        self._reset_rollout_metrics()

    def _on_step(self) -> bool:
        infos = list(self.locals.get("infos", []))
        actions = np.asarray(self.locals.get("actions", []), dtype=np.int64).reshape(-1)
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)

        for env_idx, info in enumerate(infos):
            if env_idx < len(actions):
                action = int(actions[env_idx])
                if 0 <= action < 3:
                    self._action_counts[action] += 1
            if env_idx < len(rewards):
                self._shaped_rewards.append(float(rewards[env_idx]))
            self._exposures.append(float(info.get("units_held", 0)) / self.max_units)
            self._friction_costs.append(float(info.get("friction_cost", 0.0)))
            for key in self.reward_info_keys:
                self._reward_terms[key].append(float(info.get(key, 0.0)))
            if bool(info.get("forced_clear", False)):
                self._forced_clears += 1

            portfolio_return = float(info.get("portfolio_return", 0.0))
            benchmark_return = float(info.get("benchmark_simple_return", 0.0))
            self._episode_strategy_factors[env_idx] *= max(1.0 + portfolio_return, 0.0)
            self._episode_benchmark_factors[env_idx] *= max(1.0 + benchmark_return, 0.0)
            if env_idx < len(dones) and dones[env_idx]:
                strategy_factor = self._episode_strategy_factors.pop(env_idx)
                benchmark_factor = self._episode_benchmark_factors.pop(env_idx)
                self._completed_episode_returns.append(strategy_factor - 1.0)
                self._completed_benchmark_returns.append(benchmark_factor - 1.0)
                self._completed_excess_returns.append(
                    strategy_factor / max(benchmark_factor, 1e-9) - 1.0
                )
        return True

    def _record(self, name: str, value: float) -> None:
        self.logger.record(name, float(value), exclude="stdout")

    def _record_mean(self, name: str, values: list[float]) -> None:
        if values:
            self._record(name, float(np.mean(values)))

    def _on_rollout_end(self) -> None:
        self._record_mean(
            "performance/episode_return_mean", self._completed_episode_returns
        )
        self._record_mean(
            "performance/benchmark_episode_return_mean",
            self._completed_benchmark_returns,
        )
        self._record_mean(
            "performance/episode_excess_return_mean",
            self._completed_excess_returns,
        )
        self._record_mean("risk/exposure_mean", self._exposures)
        self._record("trading/friction_total", sum(self._friction_costs))
        self._record("trading/forced_clear_count", self._forced_clears)

        action_total = int(self._action_counts.sum())
        if action_total:
            for action, label in enumerate(("hold", "add", "clear")):
                self._record(
                    f"policy/{label}_rate", self._action_counts[action] / action_total
                )
        self._record_mean("reward/shaped_mean", self._shaped_rewards)
        for key, values in self._reward_terms.items():
            self._record_mean(f"reward/{key}_mean", values)
