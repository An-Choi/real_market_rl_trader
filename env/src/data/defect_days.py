"""Minute-data sanitation and trading-day quality classification.

The raw KIS history occasionally contains overlapping snapshots, short holes,
and legitimately delayed sessions. This module owns the quality semantics so
feature generation and collection audits cannot silently disagree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, time
from typing import Any

import pandas as pd


MARKET_OPEN = time(9, 0)
DELAYED_MARKET_OPEN = time(10, 0)
REGULAR_MARKET_END = time(15, 19)

DEFAULT_MIN_COVERAGE = 0.95
DEFAULT_MAX_GAP_MINUTES = 3
QUALITY_POLICY_VERSION = 1


@dataclass(frozen=True)
class TradingDayQuality:
    """Structured result of a one-minute trading-day audit.

    ``max_gap_minutes`` is the longest run of absent expected one-minute bars,
    including a short missing run at the beginning or end of a required full
    session. Closing-auction minutes after 15:19 are not expected regular bars.
    """

    trading_date: date | None
    is_valid: bool
    reasons: tuple[str, ...]
    flags: tuple[str, ...]
    coverage_ratio: float
    expected_rows: int
    observed_rows: int
    missing_rows: int
    duplicate_rows: int
    max_gap_minutes: int
    expected_open: time | None
    observed_open: time | None
    observed_close: time | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic record."""
        record = asdict(self)
        record["trading_date"] = (
            self.trading_date.isoformat() if self.trading_date is not None else None
        )
        for field in ("expected_open", "observed_open", "observed_close"):
            value = record[field]
            record[field] = value.isoformat() if value is not None else None
        record["reasons"] = list(self.reasons)
        record["flags"] = list(self.flags)
        return record


def sanitize_minute_rows(
    minute_df: pd.DataFrame, ts_col: str = "Timestamp"
) -> pd.DataFrame:
    """Return one deterministic, most-complete snapshot per minute.

    Adjacent KIS collection ranges can overlap. Leaving both rows corrupts
    downstream cumulative-value, VWAP, and volume features. Prefer the row
    with the largest TradingValue and then Volume for a duplicated minute.
    """
    if ts_col not in minute_df:
        raise ValueError(f"missing timestamp column: {ts_col}")
    clean = minute_df.copy()
    clean[ts_col] = pd.to_datetime(clean[ts_col])
    if clean[ts_col].isna().any():
        raise ValueError(f"{ts_col} contains invalid timestamps")
    if clean.empty:
        return clean.reset_index(drop=True)
    preference = [
        column for column in ("TradingValue", "Volume") if column in clean.columns
    ]
    clean["__source_order"] = range(len(clean))
    clean = clean.sort_values(
        [ts_col, *preference, "__source_order"],
        kind="stable",
        na_position="first",
    )
    clean = clean.drop_duplicates(subset=[ts_col], keep="last")
    return clean.drop(columns="__source_order").sort_values(ts_col).reset_index(drop=True)


def _invalid_quality(reason: str) -> TradingDayQuality:
    return TradingDayQuality(
        trading_date=None,
        is_valid=False,
        reasons=(reason,),
        flags=(),
        coverage_ratio=0.0,
        expected_rows=0,
        observed_rows=0,
        missing_rows=0,
        duplicate_rows=0,
        max_gap_minutes=0,
        expected_open=None,
        observed_open=None,
        observed_close=None,
    )


def _longest_missing_run(expected: pd.DatetimeIndex, observed: pd.DatetimeIndex) -> int:
    observed_ns = set(observed.asi8.tolist())
    longest = current = 0
    for timestamp_ns in expected.asi8:
        if int(timestamp_ns) in observed_ns:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def classify_trading_day(
    day_df: pd.DataFrame,
    ts_col: str = "Timestamp",
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES,
    require_complete_session: bool = False,
) -> TradingDayQuality:
    """Classify one trading day without filling missing bars.

    A continuous session beginning exactly at 10:00 is treated as a delayed
    market session (for example the CSAT or first trading day of the year), not
    as a corrupt 09:00 session. A partial tail is allowed by default because
    live inference audits today's data before the close. Offline collection
    audits pass ``require_complete_session=True``.
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if max_gap_minutes < 0:
        raise ValueError("max_gap_minutes must be non-negative")
    if ts_col not in day_df:
        return _invalid_quality("missing_timestamp_column")

    parsed = pd.to_datetime(day_df[ts_col], errors="coerce")
    invalid_timestamp_count = int(parsed.isna().sum())
    valid_ts = parsed.dropna().sort_values()
    if valid_ts.empty:
        return _invalid_quality(
            "invalid_timestamp" if invalid_timestamp_count else "empty_day"
        )

    dates = tuple(dict.fromkeys(valid_ts.dt.date))
    trading_date = dates[0]
    duplicate_rows = int(valid_ts.duplicated().sum())
    unique_ts = pd.DatetimeIndex(valid_ts.drop_duplicates())
    observed_open = unique_ts[0].time()
    observed_close = unique_ts[-1].time()

    reasons: list[str] = []
    flags: list[str] = []
    if invalid_timestamp_count:
        reasons.append("invalid_timestamp")
    if len(dates) != 1:
        reasons.append("multiple_trading_dates")
    if duplicate_rows:
        reasons.append("duplicate_timestamp")

    # Exact 10:00 starts are the delayed-session shape. Other starts are
    # assessed against the normal 09:00 grid and may miss only a small head.
    delayed = observed_open == DELAYED_MARKET_OPEN
    expected_open = DELAYED_MARKET_OPEN if delayed else MARKET_OPEN
    if delayed:
        flags.append("delayed_open_session")

    first = unique_ts[0]
    expected_start = first.normalize() + pd.Timedelta(
        hours=expected_open.hour, minutes=expected_open.minute
    )
    regular_end = first.normalize() + pd.Timedelta(
        hours=REGULAR_MARKET_END.hour, minutes=REGULAR_MARKET_END.minute
    )

    observed_regular = unique_ts[(unique_ts >= expected_start) & (unique_ts <= regular_end)]
    if observed_regular.empty:
        reasons.append("no_regular_session_rows")
        expected = pd.DatetimeIndex([])
    else:
        expected_end = regular_end if require_complete_session else observed_regular[-1]
        expected = pd.date_range(expected_start, expected_end, freq="1min")

    observed_on_grid = observed_regular.intersection(expected)
    expected_rows = len(expected)
    observed_rows = len(observed_on_grid)
    missing_rows = expected_rows - observed_rows
    coverage = observed_rows / expected_rows if expected_rows else 0.0
    longest_gap = _longest_missing_run(expected, observed_on_grid) if expected_rows else 0

    if coverage < min_coverage:
        reasons.append("coverage_below_threshold")
    if longest_gap > max_gap_minutes:
        reasons.append("gap_exceeds_limit")
    elif missing_rows:
        flags.append("small_gaps_tolerated")
    if require_complete_session and expected_rows and observed_regular[-1] < regular_end:
        reasons.append("incomplete_session_tail")

    head_delay = max(
        0,
        int(
            (
                unique_ts[0]
                - (first.normalize() + pd.Timedelta(hours=MARKET_OPEN.hour))
            ).total_seconds()
            // 60
        ),
    )
    if not delayed and head_delay > max_gap_minutes:
        reasons.append("unexpected_late_open")

    reasons = list(dict.fromkeys(reasons))
    flags = list(dict.fromkeys(flags))
    return TradingDayQuality(
        trading_date=trading_date,
        is_valid=not reasons,
        reasons=tuple(reasons),
        flags=tuple(flags),
        coverage_ratio=float(coverage),
        expected_rows=expected_rows,
        observed_rows=observed_rows,
        missing_rows=missing_rows,
        duplicate_rows=duplicate_rows,
        max_gap_minutes=longest_gap,
        expected_open=expected_open,
        observed_open=observed_open,
        observed_close=observed_close,
    )


def is_valid_trading_day(
    day_df: pd.DataFrame,
    ts_col: str = "Timestamp",
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES,
    require_complete_session: bool = False,
) -> bool:
    """Return whether a day passes the shared quality policy."""
    return classify_trading_day(
        day_df,
        ts_col,
        min_coverage=min_coverage,
        max_gap_minutes=max_gap_minutes,
        require_complete_session=require_complete_session,
    ).is_valid


def drop_defect_days(
    minute_df: pd.DataFrame,
    ts_col: str = "Timestamp",
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES,
) -> pd.DataFrame:
    """Deduplicate rows and retain days accepted by the shared policy."""
    minute_df = sanitize_minute_rows(minute_df, ts_col)
    ts = pd.to_datetime(minute_df[ts_col])
    keep_dates = {
        day
        for day, group in minute_df.groupby(ts.dt.date)
        if is_valid_trading_day(
            group,
            ts_col,
            min_coverage=min_coverage,
            max_gap_minutes=max_gap_minutes,
        )
    }
    return minute_df[ts.dt.date.isin(keep_dates)].reset_index(drop=True)
