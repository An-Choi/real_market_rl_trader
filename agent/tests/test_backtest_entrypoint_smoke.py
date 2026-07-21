from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import DataLoader
from data.feature_builder import build_features
from data.feature_engineer import FeatureEngineer
from env.trading_env import TradingEnvironment
from friction.friction_model import FrictionModel
from models import RLAgent, make_training_metadata, save_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_synthetic_minute_parquet(root: Path, symbol: str) -> None:
    days = [f"2025-06-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11)]
    rng = np.random.default_rng(11)
    frames = []
    price = 100.0

    for day in days:
        ts = pd.date_range(f"{day} 09:00", periods=390, freq="1min", tz="Asia/Seoul")
        closes = price + np.cumsum(rng.normal(0, 0.15, 390))
        price = float(closes[-1])
        frames.append(pd.DataFrame({
            "Timestamp": ts,
            "Open": closes,
            "High": closes + 0.4,
            "Low": closes - 0.4,
            "Close": closes,
            "Volume": rng.integers(500, 5000, 390),
            "TradingValue": np.cumsum(rng.integers(1_000_000, 9_000_000, 390)),
        }))

    out_dir = root / symbol / "1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        out_dir / "2025-06.parquet", engine="pyarrow"
    )


def test_backtest_entrypoint_runs_on_synthetic_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_synthetic_minute_parquet(raw_dir, "005930")

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "backtest.py"),
            "--baseline",
            "random",
            "--max-steps",
            "5",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["agent"] == "random"
    assert payload["symbol"] == "005930"
    assert payload["split"] == "test"
    assert set(payload["metrics"]) >= {
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "trade_count",
        "final_portfolio_value",
        "overnight_hold_rate",
        "open_at_end",
        "terminal_liquidation_cost",
        "market_return",
    }


def test_backtest_entrypoint_compares_baselines(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_synthetic_minute_parquet(raw_dir, "005930")

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "backtest.py"),
            "--compare-baselines",
            "--max-steps",
            "5",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["symbol"] == "005930"
    assert payload["split"] == "test"
    assert [row["agent"] for row in payload["results"]] == [
        "buy_and_hold",
        "random",
        "ma_crossover",
    ]
    assert all("total_return" in row["metrics"] for row in payload["results"])


def test_backtest_entrypoint_supports_multi_seed_baseline(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_synthetic_minute_parquet(raw_dir, "005930")

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "backtest.py"),
            "--baseline",
            "random",
            "--seeds",
            "1,2",
            "--max-steps",
            "5",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["agent"] == "random"
    assert payload["seeds"] == [1, 2]
    assert len(payload["runs"]) == 2
    assert "total_return" in payload["mean_metrics"]
    assert "total_return" in payload["std_metrics"]


def test_backtest_entrypoint_compares_baselines_with_artifact(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    symbol = "005930"
    _write_synthetic_minute_parquet(raw_dir, symbol)

    loader = DataLoader(raw_data_dir=raw_dir, processed_data_dir=processed_dir)
    featured_data = build_features(symbol, loader)
    env = TradingEnvironment(
        market_data=featured_data,
        feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
        friction_model=FrictionModel(),
        unit_fraction=0.20,  # 레포 config와 일치해야 CLI 백테스트가 artifact를 수용
        episode_days=20,
        duration_horizon_bars=1280,
        nominal_bars_per_day=64,
        feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
    )
    agent = RLAgent(model_kwargs={
        "seed": 0,
        "device": "cpu",
        "n_steps": 8,
        "batch_size": 4,
        "verbose": 0,
    })
    agent.train(env, total_timesteps=8)
    metadata = make_training_metadata(
        agent=agent,
        symbol=symbol,
        featured_data=featured_data,
        feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
        feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
        env_params={
            "unit_fraction": env.unit_fraction,
            "max_units": env.max_units,
            "initial_cash": env.initial_cash,
            "episode_days": env.episode_days,
            "duration_horizon_bars": env.duration_horizon_bars,
            "nominal_bars_per_day": env.nominal_bars_per_day,
        },
    )
    artifact_dir = save_artifact(agent, metadata, tmp_path / "artifacts")

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "backtest.py"),
            "--compare-baselines",
            "--artifact",
            str(artifact_dir),
            "--max-steps",
            "5",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)

    assert [row["agent"] for row in payload["results"]] == [
        "buy_and_hold",
        "random",
        "ma_crossover",
        metadata.artifact_id,
    ]
    assert all("total_return" in row["metrics"] for row in payload["results"])
