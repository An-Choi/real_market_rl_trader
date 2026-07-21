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


# 거래일 달력 없이 커버리지를 판단하므로 연휴 허용 오차를 둔다.
# 내부 갭 9일: 최장 연휴(추석)가 캘린더 8일 갭까지 만든다.
_HEAD_TOLERANCE_DAYS = 7
_TAIL_TOLERANCE_DAYS = 7
_MAX_INTERNAL_GAP_DAYS = 9


def _is_partition_complete(days: list[date], window_start: date, window_end: date) -> bool:
    """True if the partition's trading days plausibly cover [window_start, window_end].

    부분 수집으로 멈춘 파티션(예: 월초 이틀만 존재)을 '완료'로 오인해 영구
    스킵하는 것을 막는다. head/tail 결손과 월 중간의 큰 갭을 모두 검사한다.
    """
    if not days:
        return False
    ordered = sorted(days)
    if (ordered[0] - window_start).days > _HEAD_TOLERANCE_DAYS:
        return False
    if (window_end - ordered[-1]).days > _TAIL_TOLERANCE_DAYS:
        return False
    return all(
        (later - earlier).days <= _MAX_INTERNAL_GAP_DAYS
        for earlier, later in zip(ordered, ordered[1:])
    )


def _merge_minute_days(
    existing: pd.DataFrame, new: pd.DataFrame
) -> tuple[pd.DataFrame, list[date]]:
    """Per-day merge of a refetched minute month into the existing partition.

    행이 줄어든 날과 새 응답에 아예 없는 날(rolling 경계 잘림)은 기존 행을
    보존하고, 나머지 날은 새 데이터를 채택한다. 하루의 결손이 그 달 전체
    갱신을 막지 않도록 하기 위한 것(월 단위 all-or-nothing 금지).

    Returns (merged, preserved_dates) — preserved_dates는 기존 행을 지킨 날짜들.
    """
    old_counts = existing["Timestamp"].dt.date.value_counts()
    new_counts = new["Timestamp"].dt.date.value_counts()
    aligned = new_counts.reindex(old_counts.index, fill_value=0)
    preserved = sorted(old_counts.index[aligned < old_counts])
    merged = pd.concat(
        [
            existing[existing["Timestamp"].dt.date.isin(preserved)],
            new[~new["Timestamp"].dt.date.isin(preserved)],
        ],
        ignore_index=True,
    )
    merged = merged[new.columns.tolist()].sort_values("Timestamp").reset_index(drop=True)
    return merged, preserved


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

        Crash-safe and resumable: a month whose Parquet already exists AND
        plausibly covers its window is skipped unless ``overwrite`` is True or
        its label is in ``overwrite_partitions``. Incomplete partitions are
        resumed, and refetches merge per-day (shrunk days keep existing rows).
        Returns labels actually written.
        """
        forced = overwrite_partitions or set()
        saved: list[str] = []
        for w_start, w_end in _month_windows(start, end):
            partition = f"{w_start.year:04d}-{w_start.month:02d}"
            path = self.raw_data_dir / symbol / "1m" / f"{partition}.parquet"
            if path.exists() and not overwrite and partition not in forced:
                existing_days = (
                    pd.read_parquet(path, columns=["Timestamp"])["Timestamp"]
                    .dt.date.unique().tolist()
                )
                if _is_partition_complete(existing_days, w_start, w_end):
                    logging.info("Skip existing minute partition: %s", path)
                    continue
                logging.info("Resuming incomplete minute partition: %s", path)
            df = fetcher.fetch_minute_range(
                start=w_start, end=w_end, max_pages_per_day=max_pages_per_day
            )
            if df.empty:
                continue
            if overwrite:
                # 기존 --overwrite 전체 재수집 경로: 무조건 저장 (기존 동작 불변)
                self.save_raw_parquet(df, symbol=symbol, interval="1m", partition=partition)
                written: Path | None = path
            else:
                if path.exists():
                    existing = pd.read_parquet(path)
                    df, preserved = _merge_minute_days(existing, df)
                    if preserved:
                        logging.warning(
                            "[%s] minute partition %s: kept existing rows for %s (refetch shrank)",
                            symbol, partition, [str(day) for day in preserved],
                        )
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

        Guards against rolling-boundary truncation per day: dates whose refetch
        shrank (or vanished) keep their existing rows, all other dates update.
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
            preserved: list[date] = []
            if path.exists():
                existing = pd.read_parquet(path)
                df, preserved = _merge_minute_days(existing, df)
                if preserved:
                    logging.warning(
                        "[%s] force month %s: kept existing rows for %s (refetch shrank)",
                        symbol, label, [str(day) for day in preserved],
                    )
            written = self.save_if_changed(
                df, symbol=symbol, interval="1m", partition=label, time_col="Timestamp"
            )
            if written is not None:
                results[label] = "replaced"
            else:
                results[label] = "kept_existing" if preserved else "unchanged"
        return results

    def audit_minute_coverage(self, symbol: str, today: date | None = None) -> list[str]:
        """Scan saved 1m partitions for coverage holes.

        부분 수집으로 멈춘 월, 월 중간의 큰 갭, 아예 빠진 월 파일을 경고
        문자열로 반환한다(문제 없으면 빈 리스트). 백필/일일 수집 후 호출해서
        구멍이 조용히 고착되기 전에 드러내는 용도.
        """
        today = today or date.today()
        directory = self.raw_data_dir / symbol / "1m"
        labels = sorted(path.stem for path in directory.glob("*.parquet"))
        warnings: list[str] = []
        for index, label in enumerate(labels):
            days = sorted(
                pd.read_parquet(directory / f"{label}.parquet", columns=["Timestamp"])
                ["Timestamp"].dt.date.unique()
            )
            month_start = date(int(label[:4]), int(label[5:7]), 1)
            # 첫 파티션은 rolling 수집 시작점이라 월 중간 시작이 정상이다.
            window_start = days[0] if index == 0 else month_start
            window_end = min(_month_end(month_start), today)
            if not _is_partition_complete(days, window_start, window_end):
                warnings.append(
                    f"{label}: incomplete partition ({days[0]}..{days[-1]}, {len(days)} days)"
                )
        if labels:
            first = date(int(labels[0][:4]), int(labels[0][5:7]), 1)
            last = date(int(labels[-1][:4]), int(labels[-1][5:7]), 1)
            expected = {
                f"{w_start.year:04d}-{w_start.month:02d}"
                for w_start, _ in _month_windows(first, last)
            }
            for missing in sorted(expected.difference(labels)):
                warnings.append(f"{missing}: partition file missing")
        return warnings
