from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.data_loader import DataLoader


@pytest.fixture
def daily_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        "Open": [73.5, 74.0, 75.0],
        "High": [74.5, 74.8, 75.5],
        "Low": [73.0, 73.5, 74.5],
        "Close": [74.0, 74.2, 75.2],
        "Volume": [1_000_000, 1_100_000, 1_200_000],
        "TradingValue": [74_000_000_000, 81_620_000_000, 90_240_000_000],
        "Change": [0.5, 0.2, 1.0],
    })


def test_load_raw_parquet_returns_dataframe(
    tmp_path: Path, daily_frame: pd.DataFrame
) -> None:
    target = tmp_path / "005930" / "1d" / "2024-2024.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    daily_frame.to_parquet(target, engine="pyarrow", compression="snappy", index=False)

    loader = DataLoader(raw_data_dir=tmp_path, processed_data_dir=tmp_path / "processed")
    loaded = loader.load_raw_parquet(symbol="005930", interval="1d", partition="2024-2024")

    pd.testing.assert_frame_equal(loaded, daily_frame)


def test_load_raw_parquet_raises_when_missing(tmp_path: Path) -> None:
    loader = DataLoader(raw_data_dir=tmp_path, processed_data_dir=tmp_path / "processed")
    with pytest.raises(FileNotFoundError):
        loader.load_raw_parquet(symbol="005930", interval="1d", partition="2099-2099")


def test_load_raw_parquet_all_concatenates_partitions(tmp_path: Path) -> None:
    minute_jan = pd.DataFrame({
        "Timestamp": pd.to_datetime(["2024-01-05 09:00:00+09:00"]),
        "Open": [75000.0], "High": [75100.0], "Low": [74900.0], "Close": [75050.0],
        "Volume": [1000], "TradingValue": [75_050_000],
    })
    minute_feb = pd.DataFrame({
        "Timestamp": pd.to_datetime(["2024-02-05 09:00:00+09:00"]),
        "Open": [76000.0], "High": [76100.0], "Low": [75900.0], "Close": [76050.0],
        "Volume": [1200], "TradingValue": [91_260_000],
    })
    minute_dir = tmp_path / "005930" / "1m"
    minute_dir.mkdir(parents=True, exist_ok=True)
    minute_jan.to_parquet(minute_dir / "2024-01.parquet", engine="pyarrow", index=False)
    minute_feb.to_parquet(minute_dir / "2024-02.parquet", engine="pyarrow", index=False)

    loader = DataLoader(raw_data_dir=tmp_path, processed_data_dir=tmp_path / "processed")
    combined = loader.load_raw_parquet_all(symbol="005930", interval="1m")

    assert len(combined) == 2
    assert combined["Timestamp"].is_monotonic_increasing
