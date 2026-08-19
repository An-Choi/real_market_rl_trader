import pandas as pd

from data.defect_days import is_valid_trading_day, sanitize_minute_rows


def test_sanitize_minute_rows_keeps_most_complete_snapshot() -> None:
    frame = pd.DataFrame({
        "Timestamp": pd.to_datetime([
            "2026-02-27 09:00", "2026-02-27 09:00", "2026-02-27 09:01"
        ]),
        "TradingValue": [100, 120, 150],
        "Volume": [10, 12, 3],
        "Close": [1.0, 1.0, 1.0],
    })
    clean = sanitize_minute_rows(frame)
    assert clean["Timestamp"].is_unique
    assert clean.loc[0, "TradingValue"] == 120
    assert clean.loc[0, "Volume"] == 12


def test_raw_duplicate_day_is_invalid_until_sanitized() -> None:
    frame = pd.DataFrame({
        "Timestamp": pd.to_datetime([
            "2026-02-27 09:00", "2026-02-27 09:00", "2026-02-27 09:01"
        ])
    })
    assert not is_valid_trading_day(frame)
    assert is_valid_trading_day(sanitize_minute_rows(frame))
