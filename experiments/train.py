"""Train a configured RL agent and save a versioned model artifact."""

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

from experiments.common import load_feature_data, make_data_loader, resolve_project_path
from models.training import train_ppo_artifact
from models.walk_forward import split_by_trading_day
from utils.config_loader import load_config
from utils.logger import setup_logger


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL policy on prepared market data")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=None,
        help="PyTorch device for PPO training; defaults to agent.device or cpu.",
    )
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="train")
    parser.add_argument(
        "--episode-days",
        type=_positive_int,
        default=None,
        help="Trading days per episode; overrides environment.episode_days.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable TensorBoard logging for PPO training.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory, resolved relative to the project root.",
    )
    parser.add_argument(
        "--tensorboard-log-name",
        type=str,
        default=None,
        help="TensorBoard run name passed to the RL implementation.",
    )
    parser.add_argument(
        "--validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Select the saved model by deterministic full-validation return.",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logger = setup_logger()
    args = parse_args()
    config = load_config(PROJECT_ROOT / "env" / "configs" / "config.yaml")
    if args.episode_days is not None:
        config["environment"]["episode_days"] = args.episode_days

    symbol = args.symbol or config["data"]["symbol"]
    total_timesteps = args.total_timesteps or config["agent"]["total_timesteps"]
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    device = args.device or config["agent"].get("device", "cpu")
    tensorboard_config = config["agent"].setdefault("tensorboard", {})
    validation_config = config["agent"].setdefault("validation", {})
    if args.tensorboard is not None:
        tensorboard_config["enabled"] = args.tensorboard
    if args.tensorboard_log_name is not None:
        tensorboard_config["log_name"] = args.tensorboard_log_name
    if args.validation is not None:
        validation_config["enabled"] = args.validation
    tensorboard_log_dir = None
    if args.tensorboard_log_dir is not None:
        tensorboard_log_dir = resolve_project_path(PROJECT_ROOT, args.tensorboard_log_dir)
    elif tensorboard_config.get("log_dir"):
        tensorboard_log_dir = resolve_project_path(
            PROJECT_ROOT,
            Path(tensorboard_config["log_dir"]),
        )

    data_loader = make_data_loader(
        project_root=PROJECT_ROOT,
        config=config,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    all_featured_data = load_feature_data(
        symbol=symbol,
        data_loader=data_loader,
        force_rebuild=args.force_rebuild,
    )
    featured_data = split_by_trading_day(all_featured_data, split=args.split)
    validation_data = None
    if args.split == "train" and validation_config.get("enabled", True):
        validation_data = split_by_trading_day(
            all_featured_data,
            split="validation",
        )
    logger.info(
        "Training %s for %d timesteps on %s (%s split, %d-day episodes)",
        config["agent"]["rl_model_name"],
        total_timesteps,
        symbol,
        args.split,
        int(config["environment"].get("episode_days", 1)),
    )
    logger.info("Training device: %s", device)
    if tensorboard_config.get("enabled", False):
        logger.info(
            "TensorBoard logs: %s (run name: %s)",
            tensorboard_log_dir or PROJECT_ROOT / "runs" / "tensorboard",
            tensorboard_config.get("log_name")
            or f"{config['agent']['rl_model_name'].lower()}_{symbol}",
        )
    artifact_path = train_ppo_artifact(
        featured_data=featured_data,
        validation_data=validation_data,
        symbol=symbol,
        config=config,
        total_timesteps=total_timesteps,
        seed=seed,
        device=device,
        artifacts_dir=PROJECT_ROOT / args.artifacts_dir,
        tensorboard_log_dir=tensorboard_log_dir,
    )
    logger.info("Saved artifact: %s", artifact_path)
    print(artifact_path)


if __name__ == "__main__":
    main()
