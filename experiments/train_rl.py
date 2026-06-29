"""PPO smoke-training entrypoint for the hybrid market environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
from env.hybrid_market_env import HybridMarketEnv
from evaluation.metrics import summarize_backtest
from experiments.backtest_engine import BacktestEngine
from experiments.train import build_sample_market_data
from friction.friction_model import FrictionModel
from models.rl_agent import RLAgent
from policies.baseline_agents import BuyAndHoldAgent, MovingAverageCrossoverAgent
from utils.config_loader import load_config
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on HybridMarketEnv.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override config agent.total_timesteps for quick smoke runs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "runs" / "models" / "ppo_smoke",
        help="Path prefix for the saved Stable-Baselines3 model.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Stable-Baselines3 check_env before training.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Chronological fraction of featured rows used for PPO training.",
    )
    return parser.parse_args()


def load_featured_data(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    data_loader = DataLoader(
        raw_data_dir=PROJECT_ROOT / config["data"]["raw_dir"],
        processed_data_dir=PROJECT_ROOT / config["data"]["processed_dir"],
    )

    sample_file = config["data"].get("sample_file", "sample_ohlcv.csv")
    raw_path = data_loader.raw_data_dir / sample_file
    if raw_path.exists():
        raw_data = data_loader.load_raw_csv(sample_file)
    else:
        raw_data = build_sample_market_data()

    processed_data = data_loader.prepare_processed_data(raw_data)
    feature_engineer = FeatureEngineer(drop_na=True)
    featured_data = feature_engineer.fit_transform(processed_data)

    return featured_data, feature_engineer.feature_columns


def split_featured_data(
    featured_data: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    split_index = int(len(featured_data) * train_ratio)
    if split_index < 2 or len(featured_data) - split_index < 2:
        raise ValueError(
            "Not enough featured rows for train/test environments. "
            f"rows={len(featured_data)}, split_index={split_index}"
        )

    train_data = featured_data.iloc[:split_index].reset_index(drop=True)
    test_data = featured_data.iloc[split_index:].reset_index(drop=True)
    return train_data, test_data


def build_environment(
    config: dict[str, Any],
    market_data: pd.DataFrame,
    feature_columns: list[str],
) -> HybridMarketEnv:
    friction_model = FrictionModel(**config["friction"])
    va_config = config.get("virtual_agents", {})
    return HybridMarketEnv(
        market_data=market_data,
        feature_columns=feature_columns,
        initial_cash=config["environment"]["initial_cash"],
        max_position=config["environment"]["max_position"],
        friction_model=friction_model,
        risk_penalty_rate=config["environment"]["risk_penalty_rate"],
        noise_strength=va_config.get("noise_strength", 0.001),
        trend_strength=va_config.get("trend_strength", 0.002),
        mean_reversion_strength=va_config.get("mean_reversion_strength", 0.001),
        max_agent_impact=va_config.get("max_agent_impact", 0.05),
    )


def maybe_check_env(environment: HybridMarketEnv) -> None:
    from stable_baselines3.common.env_checker import check_env

    check_env(environment, warn=True)


def run_backtest(
    agent: Any,
    config: dict[str, Any],
    market_data: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> dict[str, float]:
    environment = build_environment(
        config,
        market_data=market_data,
        feature_columns=feature_columns,
    )
    backtest_engine = BacktestEngine(agent=agent, environment=environment)
    results = backtest_engine.run(
        max_steps=config["backtest"].get("max_steps"),
        seed=seed,
    )
    return summarize_backtest(results)


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    total_timesteps = args.total_timesteps or config["agent"]["total_timesteps"]
    seed = config.get("seed", 42)

    featured_data, feature_columns = load_featured_data(config)
    train_data, test_data = split_featured_data(featured_data, args.train_ratio)
    train_environment = build_environment(
        config,
        market_data=train_data,
        feature_columns=feature_columns,
    )
    if args.check_env:
        logger.info("Running Stable-Baselines3 check_env.")
        maybe_check_env(train_environment)
    logger.info(
        "Using chronological split: train_rows=%s, test_rows=%s",
        len(train_data),
        len(test_data),
    )

    agent = RLAgent(
        model_name=config["agent"].get("rl_model_name", "PPO"),
        policy="MlpPolicy",
        model_kwargs={
            "seed": seed,
        },
    )

    logger.info("Training PPO for %s timesteps.", total_timesteps)
    agent.train(train_environment, total_timesteps=total_timesteps)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_path)
    logger.info("Saved PPO model to %s", args.model_path)

    metrics_by_agent = {
        "ppo": run_backtest(
            agent,
            config=config,
            market_data=test_data,
            feature_columns=feature_columns,
            seed=seed,
        ),
        "moving_average_crossover": run_backtest(
            MovingAverageCrossoverAgent(),
            config=config,
            market_data=test_data,
            feature_columns=feature_columns,
            seed=seed,
        ),
        "buy_and_hold": run_backtest(
            BuyAndHoldAgent(),
            config=config,
            market_data=test_data,
            feature_columns=feature_columns,
            seed=seed,
        ),
    }

    logger.info("Backtest comparison: %s", metrics_by_agent)
    print(metrics_by_agent)


if __name__ == "__main__":
    main()
