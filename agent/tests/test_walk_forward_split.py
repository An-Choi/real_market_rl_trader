from __future__ import annotations

import pandas as pd
import pytest

from models.walk_forward import describe_walk_forward_splits, split_by_trading_day


def _daily_rows() -> pd.DataFrame:
    frames = []
    for day in range(1, 11):
        frames.append(pd.DataFrame({
            "Timestamp": pd.date_range(f"2025-06-{day:02d} 09:00", periods=2, freq="1min"),
            "Close": [100 + day, 101 + day],
        }))
    return pd.concat(frames, ignore_index=True)


def test_split_by_trading_day_preserves_day_boundaries() -> None:
    data = _daily_rows()

    train = split_by_trading_day(data, split="train")
    validation = split_by_trading_day(data, split="validation")
    test = split_by_trading_day(data, split="test")

    assert pd.to_datetime(train["Timestamp"]).dt.date.nunique() == 7
    assert pd.to_datetime(validation["Timestamp"]).dt.date.nunique() == 1
    assert pd.to_datetime(test["Timestamp"]).dt.date.nunique() == 2
    assert train["Timestamp"].max() < validation["Timestamp"].min()
    assert validation["Timestamp"].max() < test["Timestamp"].min()


def test_describe_walk_forward_splits_reports_boundaries() -> None:
    descriptions = describe_walk_forward_splits(_daily_rows())

    assert [item.name for item in descriptions] == ["train", "validation", "test"]
    assert [item.days for item in descriptions] == [7, 1, 2]
    assert descriptions[0].start == "2025-06-01"
    assert descriptions[-1].end == "2025-06-10"


def test_walk_forward_requires_enough_days() -> None:
    data = pd.DataFrame({
        "Timestamp": pd.date_range("2025-06-01 09:00", periods=2, freq="1D"),
        "Close": [100, 101],
    })

    with pytest.raises(ValueError, match="at least 3 trading days"):
        split_by_trading_day(data, split="train")
