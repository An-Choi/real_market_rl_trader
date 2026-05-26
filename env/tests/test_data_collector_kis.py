from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pipeline.data_collector import DataCollector


@pytest.fixture
def sample_daily() -> pd.DataFrame:
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


def test_save_raw_parquet_daily_writes_partition(
    tmp_path: Path, sample_daily: pd.DataFrame
) -> None:
    collector = DataCollector(raw_data_dir=tmp_path)
    output_path = collector.save_raw_parquet(
        sample_daily, symbol="005930", interval="1d", partition="2024-2024"
    )
    assert output_path == tmp_path / "005930" / "1d" / "2024-2024.parquet"
    assert output_path.exists()

    reloaded = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(reloaded, sample_daily)


def test_save_raw_parquet_minute_uses_monthly_partition(tmp_path: Path) -> None:
    minute_df = pd.DataFrame({
        "Timestamp": pd.to_datetime([
            "2024-01-05 09:00:00+09:00",
            "2024-01-05 09:01:00+09:00",
        ]),
        "Open": [75000.0, 75050.0],
        "High": [75100.0, 75150.0],
        "Low": [74950.0, 75000.0],
        "Close": [75050.0, 75100.0],
        "Volume": [1000, 1100],
        "TradingValue": [75_050_000, 82_555_000],
    })
    collector = DataCollector(raw_data_dir=tmp_path)
    output_path = collector.save_raw_parquet(
        minute_df, symbol="005930", interval="1m", partition="2024-01"
    )
    assert output_path == tmp_path / "005930" / "1m" / "2024-01.parquet"
    assert output_path.exists()


def test_fetch_kis_daily_delegates_to_fetcher(
    tmp_path: Path, sample_daily: pd.DataFrame
) -> None:
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_daily.return_value = sample_daily

    collector = DataCollector(raw_data_dir=tmp_path)
    result = collector.fetch_kis_daily(
        fetcher=mock_fetcher,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
    )

    mock_fetcher.fetch_daily.assert_called_once_with(
        start=date(2024, 1, 1), end=date(2024, 1, 31)
    )
    pd.testing.assert_frame_equal(result, sample_daily)


def test_fetch_kis_minute_delegates_to_fetcher(tmp_path: Path) -> None:
    expected = pd.DataFrame({
        "Timestamp": pd.to_datetime(["2024-01-05 09:00:00+09:00"]),
        "Open": [75000.0], "High": [75100.0], "Low": [74900.0], "Close": [75050.0],
        "Volume": [1000], "TradingValue": [75_050_000],
    })
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_minute_range.return_value = expected

    collector = DataCollector(raw_data_dir=tmp_path)
    result = collector.fetch_kis_minute(
        fetcher=mock_fetcher,
        start=date(2024, 1, 5),
        end=date(2024, 1, 5),
    )

    mock_fetcher.fetch_minute_range.assert_called_once_with(
        start=date(2024, 1, 5), end=date(2024, 1, 5), max_pages_per_day=4
    )
    pd.testing.assert_frame_equal(result, expected)
