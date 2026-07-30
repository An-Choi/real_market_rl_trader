from __future__ import annotations

import pandas as pd
import pytest

from models.walk_forward import (
    describe_walk_forward_splits,
    generate_expanding_walk_forward_folds,
    split_by_trading_day,
)


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


def test_expanding_walk_forward_uses_fresh_non_overlapping_tests() -> None:
    data = _daily_rows()

    folds = generate_expanding_walk_forward_folds(
        data,
        n_folds=3,
        validation_days=1,
        test_days=2,
    )

    assert [fold.train["Timestamp"].dt.date.nunique() for fold in folds] == [3, 5, 7]
    assert [fold.validation["Timestamp"].dt.date.nunique() for fold in folds] == [1, 1, 1]
    assert [fold.test["Timestamp"].dt.date.nunique() for fold in folds] == [2, 2, 2]
    test_dates = [set(fold.test["Timestamp"].dt.date) for fold in folds]
    assert test_dates[0].isdisjoint(test_dates[1])
    assert test_dates[1].isdisjoint(test_dates[2])
    assert folds[0].test["Timestamp"].max() < folds[1].test["Timestamp"].min()


def test_expanding_walk_forward_description_is_json_ready() -> None:
    fold = generate_expanding_walk_forward_folds(
        _daily_rows(),
        n_folds=1,
        validation_days=2,
        test_days=2,
    )[0]

    description = fold.describe()

    assert description["fold"] == 1
    assert description["train"]["days"] == 6
    assert description["validation"]["start"] == "2025-06-07"
    assert description["test"]["end"] == "2025-06-10"


def test_expanding_walk_forward_rejects_impossible_windows() -> None:
    with pytest.raises(ValueError, match="not enough trading days"):
        generate_expanding_walk_forward_folds(
            _daily_rows(),
            n_folds=4,
            validation_days=2,
            test_days=2,
        )
