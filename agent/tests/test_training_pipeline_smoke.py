from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from data.data_loader import DataLoader
from data.feature_builder import build_features
from data.feature_engineer import FeatureEngineer
from models import load_artifact
from models.training import build_training_environment, train_ppo_artifact


def _write_synthetic_minute_parquet(root: Path, symbol: str) -> None:
    days = [f"2025-06-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11)]
    rng = np.random.default_rng(7)
    frames = []
    price = 100.0

    for day in days:
        ts = pd.date_range(f"{day} 09:00", periods=390, freq="1min", tz="Asia/Seoul")
        drift = np.linspace(0.0, 0.4, 390)
        closes = price + drift + np.cumsum(rng.normal(0, 0.12, 390))
        price = float(closes[-1])
        frames.append(pd.DataFrame({
            "Timestamp": ts,
            "Open": closes,
            "High": closes + 0.3,
            "Low": closes - 0.3,
            "Close": closes,
            "Volume": rng.integers(500, 5000, 390),
            "TradingValue": np.cumsum(rng.integers(1_000_000, 9_000_000, 390)),
        }))

    out_dir = root / symbol / "1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        out_dir / "2025-06.parquet", engine="pyarrow"
    )


def test_ppo_training_pipeline_saves_loadable_artifact(tmp_path: Path) -> None:
    symbol = "005930"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_synthetic_minute_parquet(raw_dir, symbol)

    loader = DataLoader(raw_data_dir=raw_dir, processed_data_dir=processed_dir)
    featured_data = build_features(symbol, loader)
    config = {
        "environment": {
            "initial_cash": 10_000.0,
            "unit_fraction": 0.20,
            "max_units": 5,
            "risk_penalty_rate": 0.0,
        },
        "friction": {
            "fee_rate": 0.001,
            "spread_rate": 0.0005,
            "slippage_rate": 0.0005,
            "execution_uncertainty_rate": 0.0,
            "sell_tax_rate": 0.002,
        },
        "agent": {
            "rl_model_name": "PPO",
            "tensorboard": {
                "enabled": True,
                "log_dir": str(tmp_path / "tensorboard"),
                "log_name": "smoke_ppo",
            },
        },
    }
    artifact_dir = train_ppo_artifact(
        featured_data=featured_data,
        symbol=symbol,
        config=config,
        total_timesteps=8,
        seed=0,
        artifacts_dir=tmp_path / "artifacts",
        model_kwargs={"n_steps": 8, "batch_size": 4, "verbose": 0},
    )

    from models.artifact import load_metadata

    meta = load_metadata(artifact_dir)
    assert meta.artifact_format_version == 2
    assert meta.env_params["episode_days"] >= 1
    assert meta.env_params["duration_horizon_bars"] == (
        meta.env_params["episode_days"] * 64
    )

    env = build_training_environment(
        featured_data=featured_data,
        environment_config=config["environment"],
        friction_config=config["friction"],
    )
    loaded_agent, loaded_meta = load_artifact(artifact_dir, env=env)
    observation, _ = env.reset(seed=0)
    action, _ = loaded_agent.predict(observation, deterministic=True)

    assert loaded_meta.algo == "PPO"
    assert loaded_meta.feature_schema_version == FeatureEngineer.FEATURE_SCHEMA_VERSION
    assert loaded_meta.normalization == {
        "type": "feature_standardization",
        "file": "feature_normalization.json",
    }
    assert (artifact_dir / "feature_normalization.json").is_file()
    assert loaded_agent.observation_normalizer is not None
    assert loaded_agent.model.n_steps == 8
    assert loaded_agent.model.ent_coef == 0.01
    assert action in range(6)
    event_files = list((tmp_path / "tensorboard").rglob("events.out.tfevents.*"))
    assert event_files
    event_data = EventAccumulator(str(event_files[0].parent))
    event_data.Reload()
    scalar_tags = set(event_data.Tags()["scalars"])
    assert {
        "returns/cumulative_return",
        "returns/benchmark_cumulative_return",
        "portfolio/value_mean",
        "actions/target_0pct_rate",
        "cost/friction_sum",
        "trading/forced_clear_count",
        "reward/base_return_mean",
        "daily/portfolio_return",
        "daily/benchmark_return",
        "daily/friction_sum",
        "daily/target_0pct_rate",
        "daily/reward_base_return_mean",
    }.issubset(scalar_tags)
