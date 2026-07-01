from __future__ import annotations

import numpy as np
import pandas as pd

from data.resample import add_minute_trading_value


def _day(date_str: str, cum_tv: list[float]) -> pd.DataFrame:
    n = len(cum_tv)
    ts = pd.date_range(f"{date_str} 09:00", periods=n, freq="1min", tz="Asia/Seoul")
    return pd.DataFrame({
        "Timestamp": ts,
        "Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0,
        "Volume": 1000, "TradingValue": np.array(cum_tv, dtype="int64"),
    })


def test_first_bar_of_day_equals_cumulative() -> None:
    df = _day("2025-06-02", [1000, 3000, 6000])
    out = add_minute_trading_value(df)
    assert out["MinuteTradingValue"].iloc[0] == 1000  # 첫 봉 = 누적값


def test_minute_trading_value_is_diff_within_day() -> None:
    df = _day("2025-06-02", [1000, 3000, 6000])
    out = add_minute_trading_value(df)
    assert list(out["MinuteTradingValue"]) == [1000, 2000, 3000]


def test_minute_trading_value_day_boundary_no_negative() -> None:
    # Day1 누적 커짐, Day2 09:00 = 작은 누적. whole-frame diff면 음수.
    d1 = _day("2025-06-02", [1000, 500000])
    d2 = _day("2025-06-03", [800, 1600])
    out = add_minute_trading_value(pd.concat([d1, d2], ignore_index=True))
    day2_first = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-03").date()].iloc[0]
    assert day2_first["MinuteTradingValue"] == 800  # 전날 누적 안 뺌
    assert (out["MinuteTradingValue"] >= 0).all()


def test_input_not_mutated() -> None:
    df = _day("2025-06-02", [1000, 3000])
    _ = add_minute_trading_value(df)
    assert "MinuteTradingValue" not in df.columns


from data.resample import resample_5min


def _minute_day(date_str: str, closes: list[float], start: str = "09:00") -> pd.DataFrame:
    n = len(closes)
    ts = pd.date_range(f"{date_str} {start}", periods=n, freq="1min", tz="Asia/Seoul")
    return pd.DataFrame({
        "Timestamp": ts,
        "Open": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
        "Close": closes, "Volume": list(range(1, n + 1)),
        "TradingValue": np.cumsum([c * 10 for c in closes]).astype("int64"),
    })


def test_5min_bar_uses_only_completed_minutes() -> None:
    # 09:00~09:04 5개 → 라벨 09:05, Close = 09:04 분봉 Close
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]  # 6분
    out = resample_5min(_minute_day("2025-06-02", closes))
    bar0905 = out[pd.to_datetime(out["Timestamp"]).dt.strftime("%H:%M") == "09:05"].iloc[0]
    assert bar0905["Close"] == 14.0  # 09:04 분봉 Close


def test_5min_aggregation_correct() -> None:
    closes = [10.0, 20.0, 5.0, 8.0, 12.0]  # 09:00~09:04 → 라벨 09:05
    out = resample_5min(_minute_day("2025-06-02", closes))
    bar = out[pd.to_datetime(out["Timestamp"]).dt.strftime("%H:%M") == "09:05"].iloc[0]
    assert bar["Open"] == 10.0 and bar["Close"] == 12.0
    assert bar["High"] == 21.0 and bar["Low"] == 4.0  # High=max(c+1), Low=min(c-1)


def test_incomplete_trailing_bar_dropped() -> None:
    # 7분 → 09:05 bucket(5개) 완전 + 09:10 bucket(2개) 불완전 → drop
    closes = [float(i) for i in range(7)]
    out = resample_5min(_minute_day("2025-06-02", closes))
    labels = set(pd.to_datetime(out["Timestamp"]).dt.strftime("%H:%M"))
    assert "09:05" in labels and "09:10" not in labels


def test_resample_does_not_cross_day_boundary() -> None:
    d1 = _minute_day("2025-06-02", [float(i) for i in range(10)])
    d2 = _minute_day("2025-06-03", [float(i) for i in range(10)])
    out = resample_5min(pd.concat([d1, d2], ignore_index=True))
    dates = pd.to_datetime(out["Timestamp"]).dt.date
    assert set(dates) == {pd.Timestamp("2025-06-02").date(), pd.Timestamp("2025-06-03").date()}
