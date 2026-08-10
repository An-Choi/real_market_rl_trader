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


@dataclass(frozen=True)
class ExpandingWalkForwardFold:
    """One leakage-safe expanding train/validation/test fold."""

    index: int
    train_start: datetime.date
    train_end: datetime.date
    validation_start: datetime.date
    validation_end: datetime.date
    test_start: datetime.date
    test_end: datetime.date
    purge_days: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "index": self.index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "purge_days": self.purge_days,
        }


def build_expanding_walk_forward_folds(
    data_by_symbol: "dict[str, pd.DataFrame]",
    *,
    n_folds: int = 3,
    validation_days: int = 20,
    test_days: int = 20,
    purge_days: int = 5,
    min_train_days: int | None = None,
    timestamp_col: str = "Timestamp",
) -> list[ExpandingWalkForwardFold]:
    """Build non-overlapping forward test windows with expanding training data.

    The most recent fold always ends on the latest shared calendar date. Earlier
    folds move backward by ``test_days``. Validation and test each remain after
    the training boundary, with a purge gap on both transitions.
    """
    if not data_by_symbol:
        raise ValueError("data_by_symbol must not be empty")
    for name, value in {
        "n_folds": n_folds,
        "validation_days": validation_days,
        "test_days": test_days,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if purge_days < 0:
        raise ValueError("purge_days must be >= 0")

    union_days = sorted({
        day
        for data in data_by_symbol.values()
        for day in _trading_days(data, timestamp_col).unique()
    })
    required_tail = n_folds * test_days + validation_days + 2 * purge_days
    train_days = len(union_days) - required_tail
    if min_train_days is not None and train_days < int(min_train_days):
        raise ValueError(
            f"inferred training window has {train_days} days, below "
            f"min_train_days={min_train_days}"
        )
    if train_days < 1:
        raise ValueError(
            f"not enough trading days ({len(union_days)}) for {n_folds} folds: "
            f"need more than {required_tail}"
        )

    folds: list[ExpandingWalkForwardFold] = []
    for index in range(n_folds):
        train_end_idx = train_days - 1 + index * test_days
        validation_start_idx = train_end_idx + purge_days + 1
        validation_end_idx = validation_start_idx + validation_days - 1
        test_start_idx = validation_end_idx + purge_days + 1
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= len(union_days):
            raise ValueError(
                "walk-forward window exceeds available data; reduce min_train_days, "
                "folds, validation_days, test_days, or purge_days"
            )
        fold = ExpandingWalkForwardFold(
            index=index + 1,
            train_start=union_days[0],
            train_end=union_days[train_end_idx],
            validation_start=union_days[validation_start_idx],
            validation_end=union_days[validation_end_idx],
            test_start=union_days[test_start_idx],
            test_end=union_days[test_end_idx],
            purge_days=purge_days,
        )
        for symbol, data in data_by_symbol.items():
            for segment in ("train", "validation", "test"):
                if slice_walk_forward_fold(data, fold, segment=segment).empty:
                    raise ValueError(
                        f"symbol {symbol} has no rows in fold {fold.index} {segment} window"
                    )
        folds.append(fold)
    return folds


def slice_walk_forward_fold(
    data: pd.DataFrame,
    fold: ExpandingWalkForwardFold,
    *,
    segment: Literal["train", "validation", "test"],
    timestamp_col: str = "Timestamp",
) -> pd.DataFrame:
    """Return one explicit fold segment without ratio-based re-splitting."""
    bounds = {
        "train": (fold.train_start, fold.train_end),
        "validation": (fold.validation_start, fold.validation_end),
        "test": (fold.test_start, fold.test_end),
    }
    start, end = bounds[segment]
    dates = _trading_days(data, timestamp_col)
    return data.loc[(dates >= start) & (dates <= end)].copy().reset_index(drop=True)


def split_into_day_windows(
    data: pd.DataFrame,
    *,
    window_days: int,
    min_window_days: int = 1,
    timestamp_col: str = "Timestamp",
) -> dict[str, pd.DataFrame]:
    """Split a chronological frame into non-overlapping trading-day windows.

    A short tail is merged into the preceding window so model selection is not
    dominated by a tiny final segment. Labels include dates for readable logs.
    """
    if window_days < 1:
        raise ValueError("window_days must be positive")
    if min_window_days < 1 or min_window_days > window_days:
        raise ValueError("min_window_days must be between 1 and window_days")
    days = pd.Index(sorted(_trading_days(data, timestamp_col).unique()))
    if days.empty:
        raise ValueError("cannot window data with no trading days")
    chunks = [
        days[start:start + window_days]
        for start in range(0, len(days), window_days)
    ]
    if len(chunks) > 1 and len(chunks[-1]) < min_window_days:
        chunks[-2] = chunks[-2].append(chunks[-1])
        chunks.pop()

    row_days = _trading_days(data, timestamp_col)
    windows: dict[str, pd.DataFrame] = {}
    for chunk in chunks:
        start, end = chunk[0], chunk[-1]
        label = f"{start.isoformat()}..{end.isoformat()}"
        windows[label] = data.loc[row_days.isin(chunk)].copy().reset_index(drop=True)
    return windows


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
