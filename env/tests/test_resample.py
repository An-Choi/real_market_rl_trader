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
