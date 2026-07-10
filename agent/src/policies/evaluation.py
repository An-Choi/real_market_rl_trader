"""Policy evaluation helpers for backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from data.feature_engineer import FeatureEngineer
from env.trading_env import TradingEnvironment
from evaluation.metrics import summarize_backtest
from friction.friction_model import FrictionModel
from models import load_artifact
from policies.baseline_agents import SUPPORTED_BASELINES, make_baseline_agent


BACKTEST_INFO_KEYS = (
    "trade_value",
    "forced_clear",
    "portfolio_return",
    "base_return",
    "benchmark_return",
    "benchmark_relative_reward",
    "inventory_penalty",
    "turnover_penalty",
    "drawdown_penalty",
    "downside_penalty",
    "real_price",
    "simulated_price",
    "agent_impact",
    "execution_price",
    "execution_step",
    "valuation_step",
    "execution_real_price",
    "execution_simulated_price",
    "execution_agent_impact",
    "valuation_real_price",
    "valuation_simulated_price",
    "valuation_agent_impact",
)


@dataclass
class BacktestEngine:
    """Run an agent through a trading environment and collect results."""

    agent: Any
    environment: Any
    results: list[dict] = field(default_factory=list)

    def run(
        self,
        max_steps: int | None = None,
        seed: int | None = None,
        dates: list[Any] | tuple[Any, ...] | None = None,
    ) -> pd.DataFrame:
        """Execute every trading day in the supplied evaluation split."""
        evaluation_dates = dates
        if evaluation_dates is None:
            evaluation_dates = getattr(self.environment, "available_dates", (None,))
        self.results = []

        global_step = 0
        for episode_index, episode_date in enumerate(evaluation_dates):
            if hasattr(self.agent, "reset"):
                self.agent.reset()
            reset_options = None if episode_date is None else {"date": episode_date}
            episode_seed = None if seed is None else seed + episode_index
            observation, reset_info = self.environment.reset(
                seed=episode_seed,
                options=reset_options,
            )
            done = False
            episode_step = 0

            while not done:
                market_row = self.environment.get_current_market_row()
                timestamp = market_row.get("Timestamp")
                action = self._predict_action(observation, market_row=market_row)
                observation, reward, terminated, truncated, info = self.environment.step(action)

                result_row = {
                    "step": global_step,
                    "episode_step": episode_step,
                    "episode": episode_index,
                    "episode_date": str(episode_date),
                    "timestamp": timestamp,
                    "initial_portfolio_value": reset_info.get(
                        "portfolio_value", self.environment.initial_cash
                    ),
                    "action": action,
                    "reward": reward,
                    "portfolio_value": info["portfolio_value"],
                    "units_held": info["units_held"],
                    "friction_cost": info["friction_cost"],
                }
                for key in BACKTEST_INFO_KEYS:
                    if key in info:
                        result_row[key] = info[key]
                self.results.append(result_row)

                global_step += 1
                episode_step += 1
                done = terminated or truncated
                if max_steps is not None and episode_step >= max_steps:
                    break

        return self.collect_results()

    def collect_results(self) -> pd.DataFrame:
        """Return collected backtest results as a DataFrame."""
        return pd.DataFrame(self.results)

    def _predict_action(self, observation: Any, market_row: pd.Series) -> int:
        """Call an agent predict method while supporting simple baselines."""
        try:
            action_output = self.agent.predict(observation, market_row=market_row)
        except TypeError:
            action_output = self.agent.predict(observation)

        if isinstance(action_output, tuple):
            return int(action_output[0])
        return int(action_output)


def build_backtest_environment(featured_data: Any, config: dict[str, Any]) -> TradingEnvironment:
    """Build the standard environment used for policy evaluation."""
    return TradingEnvironment(
        market_data=featured_data,
        feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
        initial_cash=config["environment"]["initial_cash"],
        unit_fraction=config["environment"]["unit_fraction"],
        max_units=config["environment"]["max_units"],
        friction_model=FrictionModel(**config["friction"]),
        risk_penalty_rate=config["environment"]["risk_penalty_rate"],
        turnover_penalty_rate=config["environment"].get("turnover_penalty_rate", 0.0),
        drawdown_penalty_rate=config["environment"].get("drawdown_penalty_rate", 0.0),
        downside_penalty_rate=config["environment"].get("downside_penalty_rate", 0.0),
        benchmark_relative_rate=config["environment"].get("benchmark_relative_rate", 0.0),
        reward_scale=config["environment"].get("reward_scale", 1.0),
        reward_return_mode=config["environment"].get(
            "reward_return_mode", "simple_return"
        ),
    )


def run_agent_backtest(
    *,
    agent: Any,
    agent_name: str,
    environment: TradingEnvironment,
    max_steps: int | None,
    seed: int,
) -> dict[str, Any]:
    """Run one policy and return a compact metric summary."""
    engine = BacktestEngine(agent=agent, environment=environment)
    results = engine.run(max_steps=max_steps, seed=seed)
    return {"agent": agent_name, "metrics": summarize_backtest(results)}


def evaluate_baseline(
    *,
    baseline_name: str,
    featured_data: Any,
    config: dict[str, Any],
    max_steps: int | None,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one named baseline policy."""
    environment = build_backtest_environment(featured_data, config)
    agent = make_baseline_agent(
        baseline_name,
        seed=seed,
        max_units=config["environment"]["max_units"],
    )
    return run_agent_backtest(
        agent=agent,
        agent_name=baseline_name,
        environment=environment,
        max_steps=max_steps,
        seed=seed,
    )


def evaluate_artifact(
    *,
    artifact_path: str | Path,
    featured_data: Any,
    config: dict[str, Any],
    max_steps: int | None,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one saved learned-policy artifact."""
    environment = build_backtest_environment(featured_data, config)
    agent, metadata = load_artifact(artifact_path, env=environment)
    return run_agent_backtest(
        agent=agent,
        agent_name=metadata.artifact_id,
        environment=environment,
        max_steps=max_steps,
        seed=seed,
    )


def compare_baselines(
    *,
    featured_data: Any,
    config: dict[str, Any],
    max_steps: int | None,
    seed: int,
    artifact_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all supported baselines and optionally one learned artifact."""
    summaries = [
        evaluate_baseline(
            baseline_name=name,
            featured_data=featured_data,
            config=config,
            max_steps=max_steps,
            seed=seed + idx,
        )
        for idx, name in enumerate(SUPPORTED_BASELINES)
    ]
    if artifact_path is not None:
        summaries.append(
            evaluate_artifact(
                artifact_path=artifact_path,
                featured_data=featured_data,
                config=config,
                max_steps=max_steps,
                seed=seed,
            )
        )
    return summaries


def summarize_seed_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated seed runs for one agent."""
    if not runs:
        return {"runs": [], "mean_metrics": {}, "std_metrics": {}}

    metric_names = sorted(set().union(*(run["metrics"].keys() for run in runs)))
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for name in metric_names:
        values = pd.Series([run["metrics"].get(name) for run in runs], dtype="float64").dropna()
        if values.empty:
            continue
        mean_metrics[name] = float(values.mean())
        std_metrics[name] = float(values.std(ddof=0))
    return {"runs": runs, "mean_metrics": mean_metrics, "std_metrics": std_metrics}


def evaluate_baseline_multi_seed(
    *,
    baseline_name: str,
    featured_data: Any,
    config: dict[str, Any],
    max_steps: int | None,
    seeds: list[int],
) -> dict[str, Any]:
    """Evaluate one baseline across multiple environment seeds."""
    runs = [
        evaluate_baseline(
            baseline_name=baseline_name,
            featured_data=featured_data,
            config=config,
            max_steps=max_steps,
            seed=seed,
        )
        for seed in seeds
    ]
    summary = summarize_seed_runs(runs)
    return {"agent": baseline_name, **summary}


def evaluate_artifact_multi_seed(
    *,
    artifact_path: str | Path,
    featured_data: Any,
    config: dict[str, Any],
    max_steps: int | None,
    seeds: list[int],
) -> dict[str, Any]:
    """Evaluate one learned artifact across multiple environment seeds."""
    runs = [
        evaluate_artifact(
            artifact_path=artifact_path,
            featured_data=featured_data,
            config=config,
            max_steps=max_steps,
            seed=seed,
        )
        for seed in seeds
    ]
    agent_name = runs[0]["agent"] if runs else str(artifact_path)
    summary = summarize_seed_runs(runs)
    return {"agent": agent_name, **summary}


def compare_baselines_multi_seed(
    *,
    featured_data: Any,
    config: dict[str, Any],
    max_steps: int | None,
    seeds: list[int],
    artifact_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate supported baselines and optionally one artifact over many seeds."""
    summaries = [
        evaluate_baseline_multi_seed(
            baseline_name=name,
            featured_data=featured_data,
            config=config,
            max_steps=max_steps,
            seeds=seeds,
        )
        for name in SUPPORTED_BASELINES
    ]
    if artifact_path is not None:
        summaries.append(
            evaluate_artifact_multi_seed(
                artifact_path=artifact_path,
                featured_data=featured_data,
                config=config,
                max_steps=max_steps,
                seeds=seeds,
            )
        )
    return summaries
