"""Train a PPO agent and save a versioned model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = PROJECT_ROOT / "env" / "src"
AGENT_SRC = PROJECT_ROOT / "agent" / "src"

for path in (ENV_SRC, AGENT_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.common import load_feature_data, make_data_loader
from models.training import train_ppo_artifact
from models.walk_forward import split_by_trading_day
from utils.config_loader import load_config
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on prepared market data")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="train")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    logger = setup_logger()
    args = parse_args()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")

    symbol = args.symbol or config["data"]["symbol"]
    total_timesteps = args.total_timesteps or config["agent"]["total_timesteps"]
    seed = args.seed if args.seed is not None else config.get("seed", 42)

    data_loader = make_data_loader(project_root=PROJECT_ROOT, config=config)
    featured_data = load_feature_data(
        symbol=symbol,
        data_loader=data_loader,
        force_rebuild=args.force_rebuild,
    )
    featured_data = split_by_trading_day(featured_data, split=args.split)
    logger.info(
        "Training %s for %d timesteps on %s (%s split)",
        config["agent"]["rl_model_name"],
        total_timesteps,
        symbol,
        args.split,
    )
    artifact_path = train_ppo_artifact(
        featured_data=featured_data,
        symbol=symbol,
        config=config,
        total_timesteps=total_timesteps,
        seed=seed,
        artifacts_dir=PROJECT_ROOT / args.artifacts_dir,
    )
    logger.info("Saved artifact: %s", artifact_path)
    print(artifact_path)


if __name__ == "__main__":
    main()
