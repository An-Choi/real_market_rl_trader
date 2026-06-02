"""KIS data collection and Parquet partition save."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def _month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into per-calendar-month windows, clamped to the range."""
    windows: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        if cur.month == 12:
            month_last = date(cur.year, 12, 31)
        else:
            month_last = date(cur.year, cur.month + 1, 1) - timedelta(days=1)
        w_end = min(month_last, end)
        windows.append((cur, w_end))
        cur = w_end + timedelta(days=1)
    return windows


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
        tmp_path = directory / f".{partition}.{os.getpid()}.tmp.parquet"
        try:
            data.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return output_path

    def backfill_minute_monthly(
        self,
        fetcher,
        symbol: str,
        start: date,
        end: date,
        overwrite: bool = False,
        max_pages_per_day: int = 4,
    ) -> list[str]:
        """Fetch minute data month-by-month, saving each month immediately.

        Crash-safe and resumable: a month whose Parquet already exists is
        skipped unless ``overwrite`` is True. Returns saved partition labels.
        """
        saved: list[str] = []
        for w_start, w_end in _month_windows(start, end):
            partition = f"{w_start.year:04d}-{w_start.month:02d}"
            path = self.raw_data_dir / symbol / "1m" / f"{partition}.parquet"
            if path.exists() and not overwrite:
                logging.info("Skip existing minute partition: %s", path)
                continue
            df = fetcher.fetch_minute_range(
                start=w_start, end=w_end, max_pages_per_day=max_pages_per_day
            )
            if df.empty:
                continue
            self.save_raw_parquet(df, symbol=symbol, interval="1m", partition=partition)
            saved.append(partition)
            logging.info("Saved minute partition %s (%d rows)", partition, len(df))
        return saved
