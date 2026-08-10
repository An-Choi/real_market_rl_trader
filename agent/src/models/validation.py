"""Full-split validation and best-parameter selection during training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


DEFAULT_ROBUST_SELECTION_WEIGHTS = {
    "median_return": 1.0,
    "worst_return": 0.5,
    "max_drawdown": 0.5,
}


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


def summarize_validation_snapshots(
    snapshots: dict[str, ValidationSnapshot],
    *,
    selection: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Aggregate validation windows and calculate the checkpoint score.

    ``mean_total_return`` preserves the legacy selector. ``robust_return`` uses
    the median window return and penalizes only a negative worst window plus
    average drawdown. This prevents one unusually strong symbol/window from
    dominating checkpoint selection.
    """
    if not snapshots:
        raise ValueError("validation snapshots must not be empty")
    config = dict(selection or {})
    metric = str(config.get("metric", "mean_total_return"))
    returns = np.asarray(
        [snapshot.total_return for snapshot in snapshots.values()], dtype=np.float64
    )
    drawdowns = np.asarray(
        [snapshot.max_drawdown for snapshot in snapshots.values()], dtype=np.float64
    )
    hold_rates = np.asarray(
        [snapshot.hold_action_rate for snapshot in snapshots.values()], dtype=np.float64
    )
    stats = {
        "mean_total_return": float(np.mean(returns)),
        "median_total_return": float(np.median(returns)),
        "worst_total_return": float(np.min(returns)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "mean_hold_action_rate": float(np.mean(hold_rates)),
    }
    if metric == "mean_total_return":
        score = stats["mean_total_return"]
    elif metric == "robust_return":
        weights = dict(DEFAULT_ROBUST_SELECTION_WEIGHTS)
        configured_weights = config.get("weights", {})
        unknown_weights = set(configured_weights).difference(weights)
        if unknown_weights:
            raise ValueError(
                f"unknown validation selection weights: {sorted(unknown_weights)}"
            )
        weights.update(configured_weights)
        for key, value in weights.items():
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"validation selection weight {key} must be finite and >= 0")
        score = (
            weights["median_return"] * stats["median_total_return"]
            + weights["worst_return"] * min(stats["worst_total_return"], 0.0)
            + weights["max_drawdown"] * stats["mean_max_drawdown"]
        )
    else:
        raise ValueError(f"unsupported validation selection metric: {metric}")
    return {"selection_score": float(score), **stats}


class FullSplitValidationCallback(BaseCallback):
    """Keep parameters with the best configured multi-window validation score."""

    def __init__(
        self,
        evaluation_envs: dict[str, Any],
        *,
        eval_freq: int,
        use_action_masks: bool,
        seed: int = 0,
        deterministic: bool = True,
        verbose: int = 0,
        selection: dict[str, Any] | None = None,
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
        self.selection = dict(selection or {})
        # Validate the metric eagerly instead of failing after a long rollout.
        summarize_validation_snapshots(
            {"validation": ValidationSnapshot(0, 0, 0, 0, 0, 0, 1, 0, 0)},
            selection=self.selection,
        )
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
        metrics = summarize_validation_snapshots(snapshots, selection=self.selection)
        score = metrics["selection_score"]
        self.latest_snapshot = snapshots
        self.evaluation_count += 1
        self._last_evaluation_timestep = self.num_timesteps

        if score > self.best_score:
            self.best_score = score
            self.best_timestep = int(self.num_timesteps)
            self.best_snapshot = snapshots
            self._best_parameters = copy.deepcopy(self.model.get_parameters())
            if self.verbose:
                print(
                    f"Validation best updated at {self.num_timesteps} steps: "
                    f"score={score:.4f}, median={metrics['median_total_return']:.2%}, "
                    f"worst={metrics['worst_total_return']:.2%}, "
                    f"mean_mdd={metrics['mean_max_drawdown']:.2%}"
                )
        self._record(metrics)

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

    def _record(self, metrics: dict[str, float]) -> None:
        # Keep TensorBoard focused on checkpoint decisions. Per-window detail is
        # preserved in artifact metadata and the walk-forward JSON report.
        for key in (
            "selection_score",
            "mean_total_return",
            "median_total_return",
            "worst_total_return",
            "mean_max_drawdown",
            "mean_hold_action_rate",
        ):
            self._record_value(f"validation/{key}", metrics[key])
        self._record_value("validation/best_selection_score", self.best_score)

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable model-selection metadata."""

        def _pack(
            snapshots: dict[str, ValidationSnapshot] | None,
        ) -> dict[str, Any] | None:
            if snapshots is None:
                return None
            return {
                **summarize_validation_snapshots(snapshots, selection=self.selection),
                "per_symbol": {sym: asdict(s) for sym, s in snapshots.items()},
            }

        return {
            "metric": str(self.selection.get("metric", "mean_total_return")),
            "selection": self.selection,
            "eval_freq": self.eval_freq,
            "evaluation_count": self.evaluation_count,
            "best_timestep": self.best_timestep,
            "best": _pack(self.best_snapshot),
            "latest": _pack(self.latest_snapshot),
        }
