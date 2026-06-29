"""Smoke-test training/backtest pipeline for the reorganized project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
from policies.baseline_agents import MovingAverageCrossoverAgent
from utils.config_loader import load_config
from utils.logger import setup_logger


def build_sample_market_data(periods: int = 120) -> pd.DataFrame:
    """Build a tiny sample OHLCV dataset for local smoke testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, size=periods))
    open_ = close + rng.normal(0, 0.5, size=periods)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, size=periods)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, size=periods)
    volume = rng.integers(100_000, 500_000, size=periods)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def main() -> None:
    """Run the end-to-end baseline pipeline."""
    logger = setup_logger()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")

    data_loader = DataLoader(
        raw_data_dir=PROJECT_ROOT / config["data"]["raw_dir"],
        processed_data_dir=PROJECT_ROOT / config["data"]["processed_dir"],
    )

    sample_file = config["data"].get("sample_file", "sample_ohlcv.csv")
    raw_path = data_loader.raw_data_dir / sample_file
    if raw_path.exists():
        logger.info("Loading raw data from %s", raw_path)
        raw_data = data_loader.load_raw_csv(sample_file)
    else:
        logger.info("No raw sample found. Building synthetic OHLCV data.")
        raw_data = build_sample_market_data()

    processed_data = data_loader.prepare_processed_data(raw_data)
    feature_engineer = FeatureEngineer(drop_na=True)
    featured_data = feature_engineer.fit_transform(processed_data)

    friction_model = FrictionModel(**config["friction"])
    environment = TradingEnvironment(
        market_data=featured_data,
        feature_columns=feature_engineer.feature_columns,
        initial_cash=config["environment"]["initial_cash"],
        unit_fraction=config["environment"]["unit_fraction"],
        max_units=config["environment"]["max_units"],
        friction_model=friction_model,
        risk_penalty_rate=config["environment"]["risk_penalty_rate"],
    )
    agent = MovingAverageCrossoverAgent()
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
