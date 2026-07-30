"""Time-ordered walk-forward split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


SplitName = Literal["all", "train", "validation", "test"]


@dataclass(frozen=True)
class WalkForwardSplit:
    """Date-bounded split metadata."""

    name: str
    start: str
    end: str
    rows: int
    days: int


@dataclass(frozen=True)
class ExpandingWalkForwardFold:
    """One expanding-train, fixed-validation, forward-test fold."""

    index: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def describe(self, timestamp_col: str = "Timestamp") -> dict[str, object]:
        """Return JSON-serializable date boundaries and row/day counts."""
        payload: dict[str, object] = {"fold": self.index}
        for name in ("train", "validation", "test"):
            frame = getattr(self, name)
            timestamps = pd.to_datetime(frame[timestamp_col])
            payload[name] = {
                "start": str(timestamps.min().date()),
                "end": str(timestamps.max().date()),
                "rows": int(len(frame)),
                "days": int(timestamps.dt.date.nunique()),
            }
        return payload


def _trading_days(data: pd.DataFrame, timestamp_col: str) -> pd.Series:
    if timestamp_col not in data:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    ts = pd.to_datetime(data[timestamp_col])
    return pd.Series(ts.dt.date, index=data.index)


def split_by_trading_day(
    data: pd.DataFrame,
    *,
    split: SplitName = "all",
    timestamp_col: str = "Timestamp",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> pd.DataFrame:
    """Return a time-ordered train/validation/test slice without splitting days."""
    if split == "all":
        return data.sort_values(timestamp_col).reset_index(drop=True)
    if data.empty:
        return data.copy()
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")

    sorted_data = data.sort_values(timestamp_col).reset_index(drop=True)
    days = _trading_days(sorted_data, timestamp_col)
    unique_days = pd.Index(days.drop_duplicates())
    n_days = len(unique_days)
    if n_days < 3:
        raise ValueError("walk-forward split requires at least 3 trading days")

    train_end = max(1, int(n_days * train_ratio))
    validation_end = max(train_end + 1, int(n_days * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, n_days - 1)

    ranges = {
        "train": unique_days[:train_end],
        "validation": unique_days[train_end:validation_end],
        "test": unique_days[validation_end:],
    }
    selected_days = ranges[split]
    mask = days.isin(selected_days)
    return sorted_data.loc[mask].reset_index(drop=True)


def describe_walk_forward_splits(
    data: pd.DataFrame,
    *,
    timestamp_col: str = "Timestamp",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> list[WalkForwardSplit]:
    """Describe train/validation/test split boundaries."""
    descriptions: list[WalkForwardSplit] = []
    for name in ("train", "validation", "test"):
        part = split_by_trading_day(
            data,
            split=name,
            timestamp_col=timestamp_col,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        ts = pd.to_datetime(part[timestamp_col])
        descriptions.append(
            WalkForwardSplit(
                name=name,
                start=str(ts.min().date()),
                end=str(ts.max().date()),
                rows=len(part),
                days=int(ts.dt.date.nunique()),
            )
        )
    return descriptions


def generate_expanding_walk_forward_folds(
    data: pd.DataFrame,
    *,
    n_folds: int = 3,
    validation_days: int | None = None,
    test_days: int | None = None,
    timestamp_col: str = "Timestamp",
) -> list[ExpandingWalkForwardFold]:
    """Build non-overlapping forward tests with an expanding training window.

    Defaults reserve roughly 10% of all trading days for each validation and
    test block. Each next fold absorbs the previous validation and test period
    into training, then advances to a fresh validation and test block.
    """
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if data.empty:
        raise ValueError("walk-forward data must not be empty")

    sorted_data = data.sort_values(timestamp_col).reset_index(drop=True)
    days = _trading_days(sorted_data, timestamp_col)
    unique_days = pd.Index(days.drop_duplicates())
    n_days = len(unique_days)
    default_window = max(1, n_days // 10)
    validation_days = validation_days or default_window
    test_days = test_days or default_window
    if validation_days <= 0 or test_days <= 0:
        raise ValueError("validation_days and test_days must be positive")

    initial_train_days = n_days - validation_days - n_folds * test_days
    if initial_train_days < 3:
        raise ValueError(
            "not enough trading days for requested folds: "
            f"{n_days} days, {n_folds} folds, "
            f"{validation_days} validation days, {test_days} test days"
        )

    folds: list[ExpandingWalkForwardFold] = []
    for fold_index in range(n_folds):
        train_end = initial_train_days + fold_index * test_days
        validation_end = train_end + validation_days
        test_end = validation_end + test_days
        train_dates = unique_days[:train_end]
        validation_dates = unique_days[train_end:validation_end]
        test_dates = unique_days[validation_end:test_end]
        folds.append(
            ExpandingWalkForwardFold(
                index=fold_index + 1,
                train=sorted_data.loc[days.isin(train_dates)].reset_index(drop=True),
                validation=sorted_data.loc[days.isin(validation_dates)].reset_index(drop=True),
                test=sorted_data.loc[days.isin(test_dates)].reset_index(drop=True),
            )
        )
    return folds
