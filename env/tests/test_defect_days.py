from __future__ import annotations

import numpy as np
import pandas as pd

from data.defect_days import (
    classify_trading_day,
    drop_defect_days,
    is_valid_trading_day,
)


def _day(date_str: str, start: str = "09:00", n: int = 20, gap_after: str | None = None) -> pd.DataFrame:
    """1분봉 하루치. gap_after='HH:MM'이면 그 분 뒤 30분 구멍을 낸다."""
    ts = pd.date_range(f"{date_str} {start}", periods=n, freq="1min", tz="Asia/Seoul")
    df = pd.DataFrame({
        "Timestamp": ts,
        "Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0,
        "Volume": 1000, "TradingValue": np.arange(1, n + 1) * 1000,
    })
    if gap_after is not None:
        cut = pd.Timestamp(f"{date_str} {gap_after}", tz="Asia/Seoul")
        keep = df["Timestamp"] <= cut
        after = df["Timestamp"] > cut + pd.Timedelta(minutes=30)
        df = df[keep | after].reset_index(drop=True)
    return df


def test_normal_0900_day_is_valid() -> None:
    assert is_valid_trading_day(_day("2025-06-02")) is True


def test_continuous_delayed_1000_session_is_valid() -> None:
    quality = classify_trading_day(
        _day("2025-11-13", start="10:00", n=320),
        require_complete_session=True,
    )
    assert quality.is_valid
    assert quality.expected_open.strftime("%H:%M") == "10:00"
    assert "delayed_open_session" in quality.flags


def test_small_gap_is_tolerated_without_filling_rows() -> None:
    day = _day("2026-03-03", n=120)
    timestamps = pd.to_datetime(day["Timestamp"])
    original_rows = len(day)
    day = day[~timestamps.dt.strftime("%H:%M").isin(["09:30", "09:31"])]

    quality = classify_trading_day(day)

    assert quality.is_valid
    assert quality.coverage_ratio >= 0.95
    assert quality.max_gap_minutes == 2
    assert len(day) == original_rows - 2


def test_intraday_gap_day_is_invalid() -> None:
    # 09:00 시작이지만 장중 30분 구멍 (구멍 이후 봉이 존재하도록 n을 충분히 크게)
    day = _day("2026-03-04", start="09:00", n=90, gap_after="09:30")
    quality = classify_trading_day(day)
    assert quality.is_valid is False
    assert "gap_exceeds_limit" in quality.reasons
    assert "coverage_below_threshold" in quality.reasons


def test_drop_removes_only_defect_days() -> None:
    good = _day("2025-06-02", start="09:00")
    bad = _day("2026-03-04", start="09:00", n=90, gap_after="09:30")
    combined = pd.concat([good, bad], ignore_index=True)
    out = drop_defect_days(combined)
    kept_days = set(pd.to_datetime(out["Timestamp"]).dt.date)
    assert kept_days == {pd.Timestamp("2025-06-02").date()}
