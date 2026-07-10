"""Rule-based policy baselines."""

from policies.baseline_agents import (
    SUPPORTED_BASELINES,
    BuyAndHoldAgent,
    MovingAverageCrossoverAgent,
    RandomAgent,
    RuleBasedRegimeAgent,
    make_baseline_agent,
)
from policies.evaluation import (
    BacktestEngine,
    build_backtest_environment,
    compare_baselines,
    compare_baselines_multi_seed,
    evaluate_artifact,
    evaluate_artifact_multi_seed,
    evaluate_baseline,
    evaluate_baseline_multi_seed,
    run_agent_backtest,
    summarize_seed_runs,
)

__all__ = [
    "SUPPORTED_BASELINES",
    "BacktestEngine",
    "BuyAndHoldAgent",
    "MovingAverageCrossoverAgent",
    "RandomAgent",
    "RuleBasedRegimeAgent",
    "build_backtest_environment",
    "compare_baselines",
    "compare_baselines_multi_seed",
    "evaluate_artifact",
    "evaluate_artifact_multi_seed",
    "evaluate_baseline",
    "evaluate_baseline_multi_seed",
    "make_baseline_agent",
    "run_agent_backtest",
    "summarize_seed_runs",
]
