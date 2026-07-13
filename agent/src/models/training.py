"""Training orchestration for learned trading agents."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from data.feature_engineer import FeatureEngineer
from env.trading_env import TradingEnvironment
from friction.friction_model import FrictionModel
from models.artifact import make_training_metadata, save_artifact
from models.normalization import FeatureNormalizer, NormalizedObservationEnv
from models.rl_agent import make_rl_agent
from models.tensorboard_callback import TradingMetricsTensorBoardCallback


DEFAULT_PPO_KWARGS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [128, 128]},
    "verbose": 1,
}


def build_training_environment(
    *,
    featured_data: Any,
    environment_config: dict[str, Any],
    friction_config: dict[str, Any],
) -> TradingEnvironment:
    """Build the standard RL training environment from prepared features."""
    episode_days = int(environment_config.get("episode_days", 1))
    nominal_bars = int(environment_config.get("nominal_bars_per_day", 64))
    return TradingEnvironment(
        market_data=featured_data,
        feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
        initial_cash=environment_config["initial_cash"],
        unit_fraction=environment_config["unit_fraction"],
        max_units=environment_config["max_units"],
        friction_model=FrictionModel(**friction_config),
        risk_penalty_rate=environment_config["risk_penalty_rate"],
        turnover_penalty_rate=environment_config.get("turnover_penalty_rate", 0.0),
        drawdown_penalty_rate=environment_config.get("drawdown_penalty_rate", 0.0),
        downside_penalty_rate=environment_config.get("downside_penalty_rate", 0.0),
        benchmark_relative_rate=environment_config.get("benchmark_relative_rate", 0.0),
        reward_scale=environment_config.get("reward_scale", 1.0),
        reward_return_mode=environment_config.get("reward_return_mode", "simple_return"),
        episode_days=episode_days,
        duration_horizon_bars=episode_days * nominal_bars,
        nominal_bars_per_day=nominal_bars,
        feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
    )


def train_ppo_artifact(
    *,
    featured_data: Any,
    symbol: str,
    config: dict[str, Any],
    total_timesteps: int,
    seed: int,
    artifacts_dir: str | Path,
    model_kwargs: dict[str, Any] | None = None,
    tensorboard_log_dir: str | Path | None = None,
) -> Path:
    """Train the configured PPO agent and save a versioned artifact."""
    feature_columns = list(FeatureEngineer.FEATURE_COLUMNS)
    raw_environment = build_training_environment(
        featured_data=featured_data,
        environment_config=config["environment"],
        friction_config=config["friction"],
    )
    normalization_config = config["agent"].get("normalization", {})
    normalization_enabled = normalization_config.get("enabled", True)
    normalizer = None
    environment: Any = raw_environment
    if normalization_enabled:
        normalizer = FeatureNormalizer.fit(
            featured_data,
            feature_columns,
            clip=float(normalization_config.get("clip", 5.0)),
        )
        environment = NormalizedObservationEnv(raw_environment, normalizer)

    tensorboard_config = config["agent"].get("tensorboard", {})
    tensorboard_enabled = bool(tensorboard_config.get("enabled", False))
    configured_log_dir = (
        tensorboard_log_dir
        or tensorboard_config.get("log_dir")
        or "runs/tensorboard"
    )
    tb_log_name = str(
        tensorboard_config.get("log_name")
        or f"{config['agent']['rl_model_name'].lower()}_{symbol}"
    )

    ppo_kwargs = dict(DEFAULT_PPO_KWARGS)
    ppo_kwargs.update(config["agent"].get("ppo", {}))
    if tensorboard_enabled and configured_log_dir is not None:
        log_dir = Path(configured_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ppo_kwargs["tensorboard_log"] = str(log_dir)
    ppo_kwargs.update(model_kwargs or {})
    agent = make_rl_agent(
        model_name=config["agent"]["rl_model_name"],
        policy="MlpPolicy",
        seed=seed,
        device="cpu",
        model_kwargs=ppo_kwargs,
    )
    metrics_callback = None
    if tensorboard_enabled:
        metrics_callback = TradingMetricsTensorBoardCallback(
            initial_cash=config["environment"]["initial_cash"],
            max_units=config["environment"]["max_units"],
        )
    agent.train(
        environment,
        total_timesteps=total_timesteps,
        tb_log_name=tb_log_name if tensorboard_enabled else None,
        callback=metrics_callback,
    )

    metadata = make_training_metadata(
        agent=agent,
        symbol=symbol,
        featured_data=featured_data,
        feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
        feature_columns=feature_columns,
        env_params={
            "unit_fraction": config["environment"]["unit_fraction"],
            "max_units": config["environment"]["max_units"],
            "initial_cash": config["environment"]["initial_cash"],
            "episode_days": raw_environment.episode_days,
            "duration_horizon_bars": raw_environment.duration_horizon_bars,
            "nominal_bars_per_day": raw_environment.nominal_bars_per_day,
        },
        normalization=(
            {"type": "feature_standardization", "file": "feature_normalization.json"}
            if normalizer is not None
            else None
        ),
        training_params={
            "total_timesteps": total_timesteps,
            "seed": seed,
            "ppo": ppo_kwargs,
            "normalization": normalization_config,
            "tensorboard": {
                "enabled": tensorboard_enabled,
                "log_dir": str(configured_log_dir) if configured_log_dir is not None else None,
                "log_name": tb_log_name if tensorboard_enabled else None,
            },
            "reward": {
                key: value
                for key, value in config["environment"].items()
                if key.endswith("penalty_rate")
                or key in {
                    "benchmark_relative_rate",
                    "reward_scale",
                    "reward_return_mode",
                }
            },
        },
    )
    if normalizer is None:
        return save_artifact(agent, metadata, artifacts_dir)

    with TemporaryDirectory() as tmp_dir:
        stats_path = Path(tmp_dir) / "feature_normalization.json"
        normalizer.save(stats_path)
        return save_artifact(
            agent,
            metadata,
            artifacts_dir,
            normalization_path=stats_path,
        )
