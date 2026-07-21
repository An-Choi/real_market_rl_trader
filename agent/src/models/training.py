"""Training orchestration for learned trading agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.feature_engineer import FeatureEngineer
from env.trading_env import TradingEnvironment
from friction.friction_model import FrictionModel
from models.artifact import make_training_metadata, save_artifact
from models.normalization import FeatureNormalizer, NormalizedObservationEnv
from models.rl_agent import make_rl_agent
from models.tensorboard_callback import TradingMetricsTensorBoardCallback
from models.validation import FullSplitValidationCallback


DEFAULT_PPO_KWARGS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.999,
    "gae_lambda": 0.98,
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
    validation_data: Any | None = None,
    symbol: str,
    config: dict[str, Any],
    total_timesteps: int,
    seed: int,
    artifacts_dir: str | Path,
    device: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tensorboard_log_dir: str | Path | None = None,
) -> Path:
    """Train the configured PPO-family agent and save a versioned artifact."""
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

    validation_callback = None
    validation_config = config["agent"].get("validation", {})
    validation_enabled = bool(validation_config.get("enabled", True))
    if validation_enabled and validation_data is not None:
        raw_validation_environment = build_training_environment(
            featured_data=validation_data,
            environment_config=config["environment"],
            friction_config=config["friction"],
        )
        validation_environment: Any = raw_validation_environment
        if normalizer is not None:
            validation_environment = NormalizedObservationEnv(
                raw_validation_environment,
                normalizer,
            )
        validation_callback = FullSplitValidationCallback(
            validation_environment,
            eval_freq=int(validation_config.get("eval_freq", 25_000)),
            use_action_masks=config["agent"]["rl_model_name"] == "MaskablePPO",
            seed=int(validation_config.get("seed", seed)),
            deterministic=bool(validation_config.get("deterministic", True)),
            verbose=int(validation_config.get("verbose", 1)),
        )

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
        device=device or config["agent"].get("device", "cpu"),
        model_kwargs=ppo_kwargs,
    )
    callbacks: list[Any] = []
    if tensorboard_enabled:
        callbacks.append(
            TradingMetricsTensorBoardCallback(
                initial_cash=config["environment"]["initial_cash"],
                max_units=config["environment"]["max_units"],
            )
        )
    if validation_callback is not None:
        callbacks.append(validation_callback)
    training_callback: Any | None = None
    if len(callbacks) == 1:
        training_callback = callbacks[0]
    elif callbacks:
        from stable_baselines3.common.callbacks import CallbackList

        training_callback = CallbackList(callbacks)
    agent.train(
        environment,
        total_timesteps=total_timesteps,
        tb_log_name=tb_log_name if tensorboard_enabled else None,
        callback=training_callback,
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
            "device": device or config["agent"].get("device", "cpu"),
            "ppo": ppo_kwargs,
            "normalization": normalization_config,
            "tensorboard": {
                "enabled": tensorboard_enabled,
                "log_dir": str(configured_log_dir) if configured_log_dir is not None else None,
                "log_name": tb_log_name if tensorboard_enabled else None,
            },
            "validation": (
                validation_callback.summary()
                if validation_callback is not None
                else {"enabled": False}
            ),
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

    return save_artifact(
        agent,
        metadata,
        artifacts_dir,
        normalization_payload=json.dumps(normalizer.to_dict(), indent=2),
    )
