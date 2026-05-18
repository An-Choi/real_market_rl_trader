"""Example pipeline wiring for the real-market RL trader skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.baseline_agents import MovingAverageCrossoverAgent
from backtest.backtest_engine import BacktestEngine
from backtest.metrics import summarize_backtest
from data.data_loader import DataLoader
from env.friction_model import FrictionModel
from env.trading_env import TradingEnvironment
from features.feature_engineer import FeatureEngineer
from utils.config_loader import load_config
from utils.logger import setup_logger


def build_sample_market_data(periods: int = 120) -> pd.DataFrame:
    """Build a tiny sample OHLCV dataset for local smoke testing."""
    # TODO: Replace this with real raw data from data/raw.
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
    """Run the end-to-end skeleton pipeline."""
    logger = setup_logger()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config" / "config.yaml")

    data_loader = DataLoader(
        raw_data_dir=project_root / config["data"]["raw_dir"],
        processed_data_dir=project_root / config["data"]["processed_dir"],
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
        max_position=config["environment"]["max_position"],
        friction_model=friction_model,
        risk_penalty_rate=config["environment"]["risk_penalty_rate"],
    )

    agent = MovingAverageCrossoverAgent()
    backtest_engine = BacktestEngine(agent=agent, environment=environment)
    results = backtest_engine.run()
    metrics = summarize_backtest(results)

    logger.info("Backtest metrics: %s", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
