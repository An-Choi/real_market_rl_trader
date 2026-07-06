"""KIS data collection and Parquet partition save."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def _month_end(day: date) -> date:
    """Return the last day of the month containing the given date."""
    if day.month == 12:
        return date(day.year, 12, 31)
    return date(day.year, day.month + 1, 1) - timedelta(days=1)


def _month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into per-calendar-month windows, clamped to the range."""
    windows: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        month_last = _month_end(cur)
        w_end = min(month_last, end)
        windows.append((cur, w_end))
        cur = w_end + timedelta(days=1)
    return windows


def _normalize_for_compare(df: pd.DataFrame, column_order: list[str], time_col: str) -> pd.DataFrame:
    """Normalize DataFrame for semantic comparison: reorder columns, sort by time, reset index."""
    return df[column_order].sort_values(time_col).reset_index(drop=True)


def _shrunk_minute_dates(existing: pd.DataFrame, new: pd.DataFrame) -> list[date]:
    """Return dates whose row count is lower in the new minute data."""
    old_counts = existing["Timestamp"].dt.date.value_counts()
    new_counts = new["Timestamp"].dt.date.value_counts()
    aligned = new_counts.reindex(old_counts.index, fill_value=0)
    return old_counts.index[(aligned < old_counts)].tolist()


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

    def save_if_changed(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        partition: str,
        time_col: str,
    ) -> Path | None:
        """Save unless the existing partition is semantically identical.

        Comparison: same column set, rows sorted by ``time_col``, index and
        column order normalized, then strict ``DataFrame.equals`` (dtype-sensitive).
        """
        path = self.raw_data_dir / symbol / interval / f"{partition}.parquet"
        new_norm = _normalize_for_compare(data, data.columns.tolist(), time_col)
        if path.exists():
            existing = pd.read_parquet(path, engine="pyarrow")
            if set(existing.columns) == set(data.columns):
                old_norm = _normalize_for_compare(existing, data.columns.tolist(), time_col)
                if old_norm.equals(new_norm):
                    logging.info("Unchanged, skip write: %s", path)
                    return None
        return self.save_raw_parquet(data, symbol=symbol, interval=interval, partition=partition)

    def refresh_daily_all(self, fetcher, symbol: str, start: date, end: date) -> str:
        """Full daily refresh into a single ``1d/all.parquet``.

        Adjusted daily prices can change retroactively on corporate actions,
        so the whole range is re-fetched and atomically replaced every run.
        Order: fetch OK -> write (or semantic skip) -> remove legacy files.
        Empty fetch touches nothing and returns "empty".
        """
        df = fetcher.fetch_daily(start=start, end=end)
        if df.empty:
            logging.warning("[%s] daily fetch returned no rows; keeping existing files", symbol)
            return "empty"
        written = self.save_if_changed(
            df, symbol=symbol, interval="1d", partition="all", time_col="Date"
        )
        directory = self.raw_data_dir / symbol / "1d"
        for legacy in directory.glob("*.parquet"):
            if legacy.name != "all.parquet":
                legacy.unlink()
                logging.info("Removed legacy daily file: %s", legacy)
        return "replaced" if written is not None else "unchanged"

    def backfill_minute_monthly(
        self,
        fetcher,
        symbol: str,
        start: date,
        end: date,
        overwrite: bool = False,
        max_pages_per_day: int = 4,
        overwrite_partitions: set[str] | None = None,
    ) -> list[str]:
        """Fetch minute data month-by-month, saving each month immediately.

        Crash-safe and resumable: a month whose Parquet already exists is
        skipped unless ``overwrite`` is True or its label is in
        ``overwrite_partitions``. Returns labels actually written.
        """
        forced = overwrite_partitions or set()
        saved: list[str] = []
        for w_start, w_end in _month_windows(start, end):
            partition = f"{w_start.year:04d}-{w_start.month:02d}"
            path = self.raw_data_dir / symbol / "1m" / f"{partition}.parquet"
            if path.exists() and not overwrite and partition not in forced:
                logging.info("Skip existing minute partition: %s", path)
                continue
            df = fetcher.fetch_minute_range(
                start=w_start, end=w_end, max_pages_per_day=max_pages_per_day
            )
            if df.empty:
                continue
            if path.exists() and not overwrite:
                existing = pd.read_parquet(path)
                shrunk = _shrunk_minute_dates(existing, df)
                if shrunk:
                    logging.warning(
                        "[%s] minute partition %s: per-day rows shrank for %s; keeping existing",
                        symbol, partition, shrunk,
                    )
                    continue
            if overwrite:
                # 기존 --overwrite 전체 재수집 경로: 무조건 저장 (기존 동작 불변)
                self.save_raw_parquet(df, symbol=symbol, interval="1m", partition=partition)
                written: Path | None = path
            else:
                # 신규 파티션 및 overwrite_partitions 경로: semantic 비교 후 저장
                written = self.save_if_changed(
                    df, symbol=symbol, interval="1m", partition=partition, time_col="Timestamp"
                )
            if written is not None:
                saved.append(partition)
                logging.info("Saved minute partition %s (%d rows)", partition, len(df))
        return saved

    def force_month_refetch(
        self,
        fetcher,
        symbol: str,
        months: list[str],
        max_pages_per_day: int = 4,
        today: date | None = None,
    ) -> dict[str, str]:
        """Explicit full-month recovery windows, independent of the rolling range.

        Guards against rolling-boundary truncation: never replaces an existing
        partition with strictly fewer unique trading days.
        """
        today = today or date.today()
        results: dict[str, str] = {}
        for label in months:
            year, month = int(label[:4]), int(label[5:7])
            w_start = date(year, month, 1)
            if w_start > today:
                logging.warning("[%s] force month %s is in the future; skipping", symbol, label)
                results[label] = "invalid_future"
                continue
            w_end = min(_month_end(w_start), today)
            df = fetcher.fetch_minute_range(
                start=w_start, end=w_end, max_pages_per_day=max_pages_per_day
            )
            if df.empty:
                logging.warning("[%s] force month %s: no data (rolling limit?)", symbol, label)
                results[label] = "unavailable"
                continue
            path = self.raw_data_dir / symbol / "1m" / f"{label}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                shrunk = _shrunk_minute_dates(existing, df)
                if shrunk:
                    logging.warning(
                        "[%s] force month %s: per-day rows shrank for %s; keeping existing",
                        symbol, label, shrunk,
                    )
                    results[label] = "kept_existing"
                    continue
            written = self.save_if_changed(
                df, symbol=symbol, interval="1m", partition=label, time_col="Timestamp"
            )
            results[label] = "replaced" if written is not None else "unchanged"
        return results
