from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import DataLoader
from data.feature_builder import build_features
from data.feature_engineer import FeatureEngineer


def _write_minute_parquet(root: Path, symbol: str) -> None:
    days = [f"2025-06-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11)]
    frames = []
    price = 100.0
    rng = np.random.default_rng(1)
    for d in days:
        ts = pd.date_range(f"{d} 09:00", periods=390, freq="1min", tz="Asia/Seoul")
        closes = price + np.cumsum(rng.normal(0, 0.2, 390))
        price = float(closes[-1])
        frames.append(pd.DataFrame({
            "Timestamp": ts, "Open": closes, "High": closes + 0.5, "Low": closes - 0.5,
            "Close": closes, "Volume": rng.integers(500, 5000, 390),
            "TradingValue": np.cumsum(rng.integers(1_000_000, 9_000_000, 390)),
        }))
    df = pd.concat(frames, ignore_index=True)
    d = root / symbol / "1m"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "2025-06.parquet", engine="pyarrow")


def _loader(tmp_path: Path) -> DataLoader:
    return DataLoader(raw_data_dir=tmp_path / "raw", processed_data_dir=tmp_path / "processed")


def test_build_creates_parquet(tmp_path: Path) -> None:
    _write_minute_parquet(tmp_path / "raw", "005930")
    out = build_features("005930", _loader(tmp_path))
    v = FeatureEngineer.FEATURE_SCHEMA_VERSION
    cached = tmp_path / "processed" / "005930" / f"features_v{v}.parquet"
    assert cached.exists()
    assert list(out.columns) == (
        ["Timestamp", "Close", "ExecPrice"] + list(FeatureEngineer.FEATURE_COLUMNS)
    )


def test_build_loads_existing(tmp_path: Path) -> None:
    _write_minute_parquet(tmp_path / "raw", "005930")
    first = build_features("005930", _loader(tmp_path))
    # raw 삭제해도 캐시에서 로드되어야 함
    (tmp_path / "raw" / "005930" / "1m" / "2025-06.parquet").unlink()
    second = build_features("005930", _loader(tmp_path))
    pd.testing.assert_frame_equal(first, second)


def test_schema_version_in_filename(tmp_path: Path) -> None:
    _write_minute_parquet(tmp_path / "raw", "005930")
    build_features("005930", _loader(tmp_path))
    v = FeatureEngineer.FEATURE_SCHEMA_VERSION
    assert (tmp_path / "processed" / "005930" / f"features_v{v}.parquet").exists()
