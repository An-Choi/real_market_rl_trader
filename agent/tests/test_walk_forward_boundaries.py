"""공유 날짜 경계·purge 분할 테스트."""
from __future__ import annotations

import datetime

import pandas as pd
import pytest

from models.walk_forward import (
    SplitBoundaries,
    compute_split_boundaries,
    split_by_trading_day,
)


def _frame(start: str, days: int, bars_per_day: int = 3) -> pd.DataFrame:
    """연속 거래일 fixture (주말 무시 — 단순 달력일)."""
    rows = []
    day = pd.Timestamp(start)
    for _ in range(days):
        for bar in range(bars_per_day):
            rows.append({"Timestamp": day + pd.Timedelta(minutes=5 * bar), "Close": 100.0})
        day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_boundaries_are_identical_across_symbols_with_different_ranges():
    data = {
        "AAA": _frame("2026-01-01", 20),          # 01-01 ~ 01-20
        "BBB": _frame("2026-01-05", 16),          # 01-05 ~ 01-20 (늦은 시작)
    }
    b = compute_split_boundaries(data)
    # union = 01-01..01-20 (20일): train_end = day[int(20*.70)-1] = day[13] = 01-14
    assert b.train_end_date == datetime.date(2026, 1, 14)
    # validation_end = day[int(20*.85)-1] = day[16] = 01-17
    assert b.validation_end_date == datetime.date(2026, 1, 17)


def test_invalid_ratios_rejected():
    data = {"AAA": _frame("2026-01-01", 20)}
    with pytest.raises(ValueError):
        compute_split_boundaries(data, train_ratio=1.2)
    with pytest.raises(ValueError):
        compute_split_boundaries(data, train_ratio=0.0)
    with pytest.raises(ValueError):
        compute_split_boundaries(data, validation_ratio=-0.1)
    with pytest.raises(ValueError):
        compute_split_boundaries(data, train_ratio=0.9, validation_ratio=0.1)  # 합 >= 1


def test_split_with_boundaries_filters_by_date():
    df = _frame("2026-01-01", 20)
    b = SplitBoundaries(datetime.date(2026, 1, 14), datetime.date(2026, 1, 17))
    train = split_by_trading_day(df, split="train", boundaries=b)
    val = split_by_trading_day(df, split="validation", boundaries=b)
    test = split_by_trading_day(df, split="test", boundaries=b)
    assert pd.to_datetime(train["Timestamp"]).dt.date.max() == datetime.date(2026, 1, 14)
    assert pd.to_datetime(val["Timestamp"]).dt.date.min() == datetime.date(2026, 1, 15)
    assert pd.to_datetime(val["Timestamp"]).dt.date.max() == datetime.date(2026, 1, 17)
    assert pd.to_datetime(test["Timestamp"]).dt.date.min() == datetime.date(2026, 1, 18)


def test_purge_drops_leading_days_of_validation_and_test_only():
    df = _frame("2026-01-01", 20)
    b = SplitBoundaries(
        datetime.date(2026, 1, 10), datetime.date(2026, 1, 15), purge_days=2
    )
    train = split_by_trading_day(df, split="train", boundaries=b)
    val = split_by_trading_day(df, split="validation", boundaries=b)
    test = split_by_trading_day(df, split="test", boundaries=b)
    assert pd.to_datetime(train["Timestamp"]).dt.date.nunique() == 10  # train 무변경
    # validation 01-11..01-15 (5일) 중 앞 2일 purge → 01-13..01-15
    assert pd.to_datetime(val["Timestamp"]).dt.date.min() == datetime.date(2026, 1, 13)
    # test 01-16..01-20 (5일) 중 앞 2일 purge → 01-18..01-20
    assert pd.to_datetime(test["Timestamp"]).dt.date.min() == datetime.date(2026, 1, 18)


def test_empty_split_after_purge_raises_with_symbol_name():
    data = {
        "AAA": _frame("2026-01-01", 30),
        "TINY": _frame("2026-01-14", 4),  # validation/test 구간 거래일이 purge로 소멸
    }
    with pytest.raises(ValueError, match="TINY"):
        compute_split_boundaries(data, purge_days=3)


def test_metadata_roundtrip_and_validation():
    b = SplitBoundaries(datetime.date(2026, 3, 12), datetime.date(2026, 5, 12), purge_days=5)
    payload = b.to_metadata()
    assert payload == {
        "train_end_date": "2026-03-12",
        "validation_end_date": "2026-05-12",
        "purge_days": 5,
    }
    assert SplitBoundaries.from_metadata(payload) == b

    with pytest.raises(ValueError):
        SplitBoundaries.from_metadata({"train_end_date": "2026-03-12"})  # 누락
    with pytest.raises(ValueError):
        SplitBoundaries.from_metadata(
            {"train_end_date": "03/12/2026", "validation_end_date": "2026-05-12", "purge_days": 0}
        )  # 비ISO
    with pytest.raises(ValueError):
        SplitBoundaries.from_metadata(
            {"train_end_date": "2026-05-12", "validation_end_date": "2026-03-12", "purge_days": 0}
        )  # 역전
    with pytest.raises(ValueError):
        SplitBoundaries.from_metadata(
            {"train_end_date": "2026-03-12", "validation_end_date": "2026-05-12", "purge_days": -1}
        )  # 음수 purge


def test_ratio_path_unchanged_without_boundaries():
    df = _frame("2026-01-01", 10)
    legacy = split_by_trading_day(df, split="train")
    assert pd.to_datetime(legacy["Timestamp"]).dt.date.nunique() == 7  # int(10*0.7)
