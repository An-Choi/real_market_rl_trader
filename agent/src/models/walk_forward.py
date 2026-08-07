"""Time-ordered walk-forward split helpers."""

from __future__ import annotations

import datetime
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
class SplitBoundaries:
    """전 종목 공통 달력 경계 + purge. metadata와 1:1 왕복."""

    train_end_date: datetime.date
    validation_end_date: datetime.date
    purge_days: int = 0

    def __post_init__(self) -> None:
        if not self.train_end_date < self.validation_end_date:
            raise ValueError(
                f"train_end_date {self.train_end_date} must be before "
                f"validation_end_date {self.validation_end_date}"
            )
        if self.purge_days < 0:
            raise ValueError(f"purge_days must be >= 0: {self.purge_days}")

    def to_metadata(self) -> dict:
        return {
            "train_end_date": self.train_end_date.isoformat(),
            "validation_end_date": self.validation_end_date.isoformat(),
            "purge_days": int(self.purge_days),
        }

    @classmethod
    def from_metadata(cls, payload: dict) -> "SplitBoundaries":
        if not isinstance(payload, dict):
            raise ValueError(f"split_boundaries must be a dict: {payload!r}")
        missing = [
            key for key in ("train_end_date", "validation_end_date", "purge_days")
            if key not in payload
        ]
        if missing:
            raise ValueError(f"split_boundaries missing keys: {missing}")
        try:
            train_end = datetime.date.fromisoformat(payload["train_end_date"])
            validation_end = datetime.date.fromisoformat(payload["validation_end_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"split_boundaries dates must be ISO YYYY-MM-DD: {payload!r}") from exc
        purge = payload["purge_days"]
        if isinstance(purge, bool) or not isinstance(purge, int):
            raise ValueError(f"purge_days must be an int: {purge!r}")
        return cls(train_end, validation_end, purge)


def _validate_ratios(train_ratio: float, validation_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")


def _trading_days(data: pd.DataFrame, timestamp_col: str) -> pd.Series:
    if timestamp_col not in data:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    ts = pd.to_datetime(data[timestamp_col])
    return pd.Series(ts.dt.date, index=data.index)


def _boundary_split_days(
    unique_days: "pd.Index", boundaries: SplitBoundaries
) -> dict[str, "pd.Index"]:
    """한 종목의 거래일들을 경계 날짜로 3분할하고 val/test 앞쪽을 purge."""
    train = unique_days[unique_days <= boundaries.train_end_date]
    validation = unique_days[
        (unique_days > boundaries.train_end_date)
        & (unique_days <= boundaries.validation_end_date)
    ]
    test = unique_days[unique_days > boundaries.validation_end_date]
    purge = boundaries.purge_days
    return {
        "train": train,
        "validation": validation[purge:],
        "test": test[purge:],
    }


def compute_split_boundaries(
    data_by_symbol: "dict[str, pd.DataFrame]",
    *,
    timestamp_col: str = "Timestamp",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    purge_days: int = 0,
) -> SplitBoundaries:
    """5종목 거래일 union 기준 70/85% 지점의 공유 달력 경계를 계산한다."""
    if not data_by_symbol:
        raise ValueError("data_by_symbol must not be empty")
    _validate_ratios(train_ratio, validation_ratio)
    all_days: set = set()
    days_by_symbol: dict[str, pd.Index] = {}
    for symbol, data in data_by_symbol.items():
        days = _trading_days(data, timestamp_col)
        unique = pd.Index(sorted(days.unique()))
        if unique.empty:
            raise ValueError(f"symbol {symbol} has no trading days")
        days_by_symbol[symbol] = unique
        all_days.update(unique)

    union_days = sorted(all_days)
    n_days = len(union_days)
    if n_days < 3:
        raise ValueError("walk-forward split requires at least 3 trading days")
    train_end = union_days[max(0, int(n_days * train_ratio) - 1)]
    validation_end = union_days[max(0, int(n_days * (train_ratio + validation_ratio)) - 1)]
    boundaries = SplitBoundaries(train_end, validation_end, purge_days)

    for symbol, unique in days_by_symbol.items():
        parts = _boundary_split_days(unique, boundaries)
        for name, part in parts.items():
            if len(part) < 1:
                raise ValueError(
                    f"symbol {symbol} has no trading days in split {name!r} "
                    f"after purge={purge_days} (boundaries: {boundaries.to_metadata()})"
                )
    return boundaries


def split_by_trading_day(
    data: pd.DataFrame,
    *,
    split: SplitName = "all",
    timestamp_col: str = "Timestamp",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    boundaries: SplitBoundaries | None = None,
) -> pd.DataFrame:
    """Return a time-ordered train/validation/test slice without splitting days."""
    if split == "all":
        return data.sort_values(timestamp_col).reset_index(drop=True)
    if data.empty:
        return data.copy()
    if boundaries is not None:
        sorted_data = data.sort_values(timestamp_col).reset_index(drop=True)
        days = _trading_days(sorted_data, timestamp_col)
        unique_days = pd.Index(sorted(days.unique()))
        selected = _boundary_split_days(unique_days, boundaries)[split]
        mask = days.isin(selected)
        return sorted_data.loc[mask].reset_index(drop=True)
    _validate_ratios(train_ratio, validation_ratio)

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
