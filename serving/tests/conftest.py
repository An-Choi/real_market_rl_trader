"""serving 테스트 경로 설정 + 공용 합성 데이터 픽스처."""
from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _src in (_ROOT / "serving" / "src", _ROOT / "agent" / "src", _ROOT / "env" / "src"):
    _p = str(_src)
    if _p not in sys.path:
        sys.path.insert(0, _p)

TZ = "Asia/Seoul"


def make_minute_data(days: int, seed: int = 7, start: str = "2026-06-01") -> pd.DataFrame:
    """결손 없는 합성 1분봉: 평일만, 09:00–15:19 정규봉 + 15:30 동시호가 print.

    TradingValue는 당일 누적(파이프라인이 diff한다). 가격은 seeded random walk —
    결정론적이라 replay parity 테스트에 그대로 쓴다.
    """
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, periods=days, tz=TZ)
    frames = []
    price = 300_000.0
    for day in bdays:
        minutes = pd.date_range(day + pd.Timedelta(hours=9),
                                day + pd.Timedelta(hours=15, minutes=19),
                                freq="1min", tz=TZ)
        minutes = minutes.append(pd.DatetimeIndex(
            [day + pd.Timedelta(hours=15, minutes=30)], tz=TZ))
        n = len(minutes)
        steps = rng.normal(0, 120, size=n)
        closes = np.maximum(price + np.cumsum(steps), 1000.0).round(-2)
        opens = np.concatenate([[price], closes[:-1]])
        volume = rng.integers(1_000, 50_000, size=n)
        minute_value = (closes * volume).astype("int64")
        frames.append(pd.DataFrame({
            "Timestamp": minutes,
            "Open": opens, "High": np.maximum(opens, closes) + 100,
            "Low": np.minimum(opens, closes) - 100, "Close": closes,
            "Volume": volume,
            "TradingValue": np.cumsum(minute_value),
        }))
        price = float(closes[-1])
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def minute_data() -> pd.DataFrame:
    # cross-day warm-up(20거래일)을 지나 feature 행이 남도록 26거래일
    return make_minute_data(days=26)


@pytest.fixture()
def raw_data_dir(tmp_path, minute_data) -> Path:
    """data/raw/<symbol>/1m/YYYY-MM.parquet 레이아웃 재현."""
    ts = pd.to_datetime(minute_data["Timestamp"])
    for period, grp in minute_data.groupby(ts.dt.to_period("M")):
        out = tmp_path / "005930" / "1m"
        out.mkdir(parents=True, exist_ok=True)
        grp.reset_index(drop=True).to_parquet(out / f"{period}.parquet")
    return tmp_path


@pytest.fixture(scope="session")
def tiny_artifact_dir(tmp_path_factory, minute_data):
    """학습 없이 seed 고정 build()만 한 MaskablePPO artifact (format v3).

    랜덤 초기화 정책도 deterministic=True predict는 결정론적이라
    parity·inference 테스트에 충분하다.
    """
    from data.feature_engineer import FeatureEngineer
    from env.trading_env import TradingEnvironment
    from friction.friction_model import FrictionModel
    from models.artifact import (
        EXPECTED_ACTION_LABELS, ArtifactMetadata, save_artifact,
    )
    from models.rl_agent import make_rl_agent

    fe = FeatureEngineer()
    featured = fe.transform(minute_data)
    friction = FrictionModel(fee_rate=0.00018, spread_rate=0.001, slippage_rate=0.0,
                             execution_uncertainty_rate=0.0, sell_tax_rate=0.002,
                             dynamic_spread=True, date_based_sell_tax=True)
    env = TradingEnvironment(
        market_data=featured,
        feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
        initial_cash=10_000.0, unit_fraction=0.199, max_units=5,
        friction_model=friction, episode_days=1,
        duration_horizon_bars=1280, nominal_bars_per_day=64,
        feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
    )
    agent = make_rl_agent(model_name="MaskablePPO", seed=42)
    agent.build(env)
    meta = ArtifactMetadata(
        artifact_format_version=3,
        artifact_id="ppo-fs3-test",
        created_at=datetime.now(timezone.utc).isoformat(),
        algo="MaskablePPO",
        policy="MlpPolicy",
        feature_schema_version=FeatureEngineer.FEATURE_SCHEMA_VERSION,
        feature_columns=list(FeatureEngineer.FEATURE_COLUMNS),
        portfolio_state_fields=["units_held_frac", "unrealized_pnl_norm",
                                "holding_duration_norm", "tod_frac",
                                "liquidity_pressure"],
        observation_dim=len(FeatureEngineer.FEATURE_COLUMNS) + 5,
        action_space={"type": "discrete", "n": 3,
                      "labels": list(EXPECTED_ACTION_LABELS)},
        normalization=None,
        train_git_sha="test",
        train_data={"symbols": ["005930"], "start": "2026-06-01", "end": "2026-06-16"},
        env_params={"unit_fraction": 0.199, "max_units": 5, "initial_cash": 10_000.0,
                    "episode_days": 1, "duration_horizon_bars": 1280,
                    "nominal_bars_per_day": 64},
        friction_params=dataclasses.asdict(friction),
    )
    out = tmp_path_factory.mktemp("artifacts")
    return save_artifact(agent, meta, out)
