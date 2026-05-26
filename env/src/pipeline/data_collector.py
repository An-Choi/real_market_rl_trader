"""KIS data collection and Parquet partition save."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass
class DataCollector:
    """Collect raw OHLCV data and save to Parquet partitions."""

    raw_data_dir: Path

    def fetch_kis_daily(self, fetcher, start: date, end: date) -> pd.DataFrame:
        return fetcher.fetch_daily(start=start, end=end)

    def fetch_kis_minute(
        self, fetcher, start: date, end: date, max_pages_per_day: int = 4
    ) -> pd.DataFrame:
        return fetcher.fetch_minute_range(
            start=start, end=end, max_pages_per_day=max_pages_per_day
        )

    def save_raw_parquet(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        partition: str,
    ) -> Path:
        """Save OHLCV to data/raw/<symbol>/<interval>/<partition>.parquet (snappy)."""
        directory = self.raw_data_dir / symbol / interval
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{partition}.parquet"
        data.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        return output_path
