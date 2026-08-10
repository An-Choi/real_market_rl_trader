"""Full-split validation and best-parameter selection during training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class ValidationSnapshot:
    """Metrics from one deterministic pass over the validation split."""

    timestep: int
    total_return: float
    final_portfolio_value: float
    max_drawdown: float
    turnover: float
    trade_count: int
    hold_action_rate: float
    add_action_rate: float
    clear_action_rate: float


class FullSplitValidationCallback(BaseCallback):
    """Keep the parameters with the best full-validation terminal return."""

    def __init__(
        self,
        evaluation_envs: dict[str, Any],
        *,
        eval_freq: int,
        use_action_masks: bool,
        seed: int = 0,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if eval_freq <= 0:
            raise ValueError("validation eval_freq must be positive")
        if not evaluation_envs:
            raise ValueError("evaluation_envs must not be empty")
        self.evaluation_envs = dict(evaluation_envs)
        self.eval_freq = int(eval_freq)
        self.use_action_masks = bool(use_action_masks)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.best_score = float("-inf")
        self.best_timestep: int | None = None
        self.best_snapshot: dict[str, ValidationSnapshot] | None = None
        self.latest_snapshot: dict[str, ValidationSnapshot] | None = None
        self.evaluation_count = 0
        self._last_evaluation_timestep: int | None = None
        self._best_parameters: dict[str, Any] | None = None

    def _on_training_start(self) -> None:
        # Step zero is a useful control: a later checkpoint must genuinely beat
        # the initialized policy on held-out data to replace it.
        self._evaluate_and_maybe_update()

    def _on_step(self) -> bool:
        last_timestep = self._last_evaluation_timestep or 0
        if self.num_timesteps - last_timestep >= self.eval_freq:
            self._evaluate_and_maybe_update()
        return True

    def _on_training_end(self) -> None:
        if self._last_evaluation_timestep != self.num_timesteps:
            self._evaluate_and_maybe_update()
        if self._best_parameters is not None:
            self.model.set_parameters(self._best_parameters, exact_match=True)

    def _evaluate_and_maybe_update(self) -> None:
        snapshots = self._run_all_splits()
        mean_return = float(
            np.mean([snap.total_return for snap in snapshots.values()])
        )
        self.latest_snapshot = snapshots
        self.evaluation_count += 1
        self._last_evaluation_timestep = self.num_timesteps

        if mean_return > self.best_score:
            self.best_score = mean_return
            self.best_timestep = int(self.num_timesteps)
            self.best_snapshot = snapshots
            self._best_parameters = copy.deepcopy(self.model.get_parameters())
            if self.verbose:
                print(
                    f"Validation best updated at {self.num_timesteps} steps: {mean_return:.4%}"
                )
        self._record(snapshots, mean_return)

    def _run_all_splits(self) -> dict[str, ValidationSnapshot]:
        return {
            symbol: self._run_full_split_for(env)
            for symbol, env in self.evaluation_envs.items()
        }

    def _run_full_split_for(self, env: Any) -> ValidationSnapshot:
        dates = tuple(self._env_attr(env, "available_dates"))
        if not dates:
            raise ValueError("validation environment has no trading dates")
        observation, reset_info = env.reset(
            seed=self.seed,
            options={"start_date": dates[0], "episode_days": len(dates)},
        )
        initial_value = float(reset_info["portfolio_value"])
        portfolio_values = [initial_value]
        traded_notional = 0.0
        trade_count = 0
        action_counts = np.zeros(3, dtype=np.int64)

        done = False
        while not done:
            predict_kwargs: dict[str, Any] = {"deterministic": self.deterministic}
            if self.use_action_masks:
                predict_kwargs["action_masks"] = self._env_attr(env, "action_masks")()
            action, _ = self.model.predict(observation, **predict_kwargs)
            action_int = int(action)
            observation, _, terminated, truncated, info = env.step(action_int)
            portfolio_values.append(float(info["portfolio_value"]))
            action_counts[action_int] += 1
            trade_value = abs(float(info.get("trade_value", 0.0)))
            traded_notional += trade_value
            trade_count += int(trade_value > 0.0)
            done = bool(terminated or truncated)

        liquidation_cost = float(self._env_attr(env, "estimate_liquidation_cost")())
        final_value = portfolio_values[-1] - liquidation_cost
        portfolio_values[-1] = final_value
        values = np.asarray(portfolio_values, dtype=np.float64)
        running_peaks = np.maximum.accumulate(values)
        max_drawdown = float(np.min(values / np.maximum(running_peaks, 1e-9) - 1.0))
        action_total = max(int(action_counts.sum()), 1)
        return ValidationSnapshot(
            timestep=int(self.num_timesteps),
            total_return=float(final_value / initial_value - 1.0),
            final_portfolio_value=float(final_value),
            max_drawdown=max_drawdown,
            turnover=float(traded_notional / max(initial_value, 1e-9)),
            trade_count=trade_count,
            hold_action_rate=float(action_counts[0] / action_total),
            add_action_rate=float(action_counts[1] / action_total),
            clear_action_rate=float(action_counts[2] / action_total),
        )

    def _env_attr(self, env: Any, name: str) -> Any:
        try:
            return env.get_wrapper_attr(name)
        except AttributeError:
            return getattr(env.unwrapped, name)

    def _record_value(self, key: str, value: float) -> None:
        self.logger.record(key, float(value))

    def _record(
        self, snapshots: dict[str, ValidationSnapshot], mean_return: float
    ) -> None:
        for symbol, snap in snapshots.items():
            self._record_value(f"validation/return_{symbol}", snap.total_return)
        self._record_value("validation/mean_total_return", mean_return)
        self._record_value("validation/best_mean_total_return", self.best_score)

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable model-selection metadata."""

        def _pack(
            snapshots: dict[str, ValidationSnapshot] | None,
        ) -> dict[str, Any] | None:
            if snapshots is None:
                return None
            return {
                "mean_total_return": float(
                    np.mean([s.total_return for s in snapshots.values()])
                ),
                "per_symbol": {sym: asdict(s) for sym, s in snapshots.items()},
            }

        return {
            "metric": "mean_total_return",
            "eval_freq": self.eval_freq,
            "evaluation_count": self.evaluation_count,
            "best_timestep": self.best_timestep,
            "best": _pack(self.best_snapshot),
            "latest": _pack(self.latest_snapshot),
        }
