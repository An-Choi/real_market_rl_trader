"""serving 테스트 경로 설정 + 공용 합성 데이터 픽스처."""
from __future__ import annotations

import sys
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
    return make_minute_data(days=12)


@pytest.fixture()
def raw_data_dir(tmp_path, minute_data) -> Path:
    """data/raw/<symbol>/1m/YYYY-MM.parquet 레이아웃 재현."""
    ts = pd.to_datetime(minute_data["Timestamp"])
    for period, grp in minute_data.groupby(ts.dt.to_period("M")):
        out = tmp_path / "005930" / "1m"
        out.mkdir(parents=True, exist_ok=True)
        grp.reset_index(drop=True).to_parquet(out / f"{period}.parquet")
    return tmp_path
