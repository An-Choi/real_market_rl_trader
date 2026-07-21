from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.feature_engineer import FeatureEngineer
from models import load_artifact
from models.artifact import load_metadata
from models.training import build_training_environment, train_ppo_artifact


def _featured_data(start: str, n_days: int) -> pd.DataFrame:
    frames = []
    price = 100.0
    for day in pd.bdate_range(start, periods=n_days):
        timestamps = pd.date_range(
            f"{day.date()} 09:00",
            periods=8,
            freq="5min",
            tz="Asia/Seoul",
        )
        closes = price + np.linspace(0.0, 1.0, len(timestamps))
        price = float(closes[-1])
        frame = pd.DataFrame({"Timestamp": timestamps, "Close": closes, "ExecPrice": closes})
        for index, column in enumerate(FeatureEngineer.FEATURE_COLUMNS):
            frame[column] = np.linspace(0.0, 0.1 + index * 0.01, len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _config() -> dict:
    return {
        "environment": {
            "initial_cash": 10_000.0,
            "unit_fraction": 0.20,
            "max_units": 5,
            "episode_days": 2,
            "nominal_bars_per_day": 8,
            "risk_penalty_rate": 0.0,
            "reward_scale": 100.0,
            "reward_return_mode": "log_return",
        },
        "friction": {
            "fee_rate": 0.001,
            "spread_rate": 0.0005,
            "slippage_rate": 0.0005,
            "execution_uncertainty_rate": 0.0,
            "sell_tax_rate": 0.002,
        },
        "agent": {
            "rl_model_name": "MaskablePPO",
            "normalization": {"enabled": True, "clip": 5.0},
            "tensorboard": {"enabled": False},
            "validation": {
                "enabled": True,
                "eval_freq": 8,
                "deterministic": True,
                "seed": 0,
                "verbose": 0,
            },
        },
    }


def test_maskable_training_selects_validation_checkpoint_and_reloads(
    tmp_path: Path,
) -> None:
    train_data = _featured_data("2025-06-02", 4)
    validation_data = _featured_data("2025-07-01", 2)
    config = _config()
    artifact_path = train_ppo_artifact(
        featured_data=train_data,
        validation_data=validation_data,
        symbol="005930",
        config=config,
        total_timesteps=16,
        seed=0,
        artifacts_dir=tmp_path / "artifacts",
        model_kwargs={
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "verbose": 0,
        },
    )

    metadata = load_metadata(artifact_path)
    validation = metadata.training_params["validation"]
    assert metadata.algo == "MaskablePPO"
    assert validation["evaluation_count"] >= 3
    assert validation["best_timestep"] in {0, 8, 16}
    assert validation["best"]["total_return"] >= validation["latest"]["total_return"]

    env = build_training_environment(
        featured_data=validation_data,
        environment_config=config["environment"],
        friction_config=config["friction"],
    )
    loaded_agent, _ = load_artifact(artifact_path, env=env)
    observation, _ = env.reset(seed=0)
    action, _ = loaded_agent.predict(observation, deterministic=True)
    assert env.action_masks()[action]
