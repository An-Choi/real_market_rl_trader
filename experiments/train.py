"""Smoke-test training/backtest pipeline for the reorganized project."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
AGENT_SRC = PROJECT_ROOT / "agent" / "src"

for path in (ENV_SRC, AGENT_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from env.trading_env import TradingEnvironment
from evaluation.metrics import summarize_backtest
from experiments.backtest_engine import BacktestEngine
from friction.friction_model import FrictionModel
from policies.baseline_agents import BuyAndHoldAgent
from utils.config_loader import load_config
from utils.logger import setup_logger


def main() -> None:
    """Run the end-to-end baseline pipeline."""
    logger = setup_logger()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")

    from data.feature_builder import build_features  # local import to keep top clean

    data_loader = DataLoader(
        raw_data_dir=PROJECT_ROOT / config["data"]["raw_dir"],
        processed_data_dir=PROJECT_ROOT / config["data"]["processed_dir"],
    )
    symbol = config["data"]["symbol"]
    featured_data = build_features(symbol, data_loader)
    feature_columns = list(FeatureEngineer.FEATURE_COLUMNS)

    friction_model = FrictionModel(**config["friction"])
    environment = TradingEnvironment(
        market_data=featured_data,
        feature_columns=feature_columns,
        initial_cash=config["environment"]["initial_cash"],
        unit_fraction=config["environment"]["unit_fraction"],
        max_units=config["environment"]["max_units"],
        friction_model=friction_model,
        risk_penalty_rate=config["environment"]["risk_penalty_rate"],
    )
    agent = BuyAndHoldAgent()
    backtest_engine = BacktestEngine(agent=agent, environment=environment)
    results = backtest_engine.run(
        max_steps=config["backtest"].get("max_steps"),
        seed=config.get("seed", 42),
    )
    metrics = summarize_backtest(results)

    logger.info("Backtest metrics: %s", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
