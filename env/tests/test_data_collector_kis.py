from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pipeline.data_collector import DataCollector


def _minute_df_days(year: int, month: int, days: list[int]) -> pd.DataFrame:
    """Helper to create a minute-level DataFrame with specified days."""
    ts = [pd.Timestamp(year=year, month=month, day=d, hour=9, tz="Asia/Seoul") for d in days]
    n = len(ts)
    return pd.DataFrame({"Timestamp": ts, "Open": [1.0] * n, "High": [1.0] * n,
                         "Low": [1.0] * n, "Close": [1.0] * n,
                         "Volume": [1] * n, "TradingValue": [1] * n})


@pytest.fixture
def sample_daily() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        "Open": [73.5, 74.0, 75.0],
        "High": [74.5, 74.8, 75.5],
        "Low": [73.0, 73.5, 74.5],
        "Close": [74.0, 74.2, 75.2],
        "Volume": [1_000_000, 1_100_000, 1_200_000],
        "TradingValue": [74_000_000_000, 81_620_000_000, 90_240_000_000],
        "Change": [0.5, 0.2, 1.0],
    })


def test_save_raw_parquet_daily_writes_partition(
    tmp_path: Path, sample_daily: pd.DataFrame
) -> None:
    collector = DataCollector(raw_data_dir=tmp_path)
    output_path = collector.save_raw_parquet(
        sample_daily, symbol="005930", interval="1d", partition="2024-2024"
    )
    assert output_path == tmp_path / "005930" / "1d" / "2024-2024.parquet"
    assert output_path.exists()

    reloaded = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(reloaded, sample_daily)


def test_save_raw_parquet_minute_uses_monthly_partition(tmp_path: Path) -> None:
    minute_df = pd.DataFrame({
        "Timestamp": pd.to_datetime([
            "2024-01-05 09:00:00+09:00",
            "2024-01-05 09:01:00+09:00",
        ]),
        "Open": [75000.0, 75050.0],
        "High": [75100.0, 75150.0],
        "Low": [74950.0, 75000.0],
        "Close": [75050.0, 75100.0],
        "Volume": [1000, 1100],
        "TradingValue": [75_050_000, 82_555_000],
    })
    collector = DataCollector(raw_data_dir=tmp_path)
    output_path = collector.save_raw_parquet(
        minute_df, symbol="005930", interval="1m", partition="2024-01"
    )
    assert output_path == tmp_path / "005930" / "1m" / "2024-01.parquet"
    assert output_path.exists()


def test_save_raw_parquet_cleans_temp_on_write_failure(
    tmp_path: Path, sample_daily: pd.DataFrame, monkeypatch
) -> None:
    def failing_to_parquet(self, path, *args, **kwargs) -> None:
        Path(path).write_text("partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_to_parquet)
    collector = DataCollector(raw_data_dir=tmp_path)

    with pytest.raises(RuntimeError, match="write failed"):
        collector.save_raw_parquet(
            sample_daily, symbol="005930", interval="1d", partition="2024-2024"
        )

    directory = tmp_path / "005930" / "1d"
    assert not (directory / "2024-2024.parquet").exists()
    assert list(directory.glob(".*.tmp.parquet")) == []


def test_fetch_kis_daily_delegates_to_fetcher(
    tmp_path: Path, sample_daily: pd.DataFrame
) -> None:
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_daily.return_value = sample_daily

    collector = DataCollector(raw_data_dir=tmp_path)
    result = collector.fetch_kis_daily(
        fetcher=mock_fetcher,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
    )

    mock_fetcher.fetch_daily.assert_called_once_with(
        start=date(2024, 1, 1), end=date(2024, 1, 31)
    )
    pd.testing.assert_frame_equal(result, sample_daily)


def test_fetch_kis_minute_delegates_to_fetcher(tmp_path: Path) -> None:
    expected = pd.DataFrame({
        "Timestamp": pd.to_datetime(["2024-01-05 09:00:00+09:00"]),
        "Open": [75000.0], "High": [75100.0], "Low": [74900.0], "Close": [75050.0],
        "Volume": [1000], "TradingValue": [75_050_000],
    })
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_minute_range.return_value = expected

    collector = DataCollector(raw_data_dir=tmp_path)
    result = collector.fetch_kis_minute(
        fetcher=mock_fetcher,
        start=date(2024, 1, 5),
        end=date(2024, 1, 5),
    )

    mock_fetcher.fetch_minute_range.assert_called_once_with(
        start=date(2024, 1, 5), end=date(2024, 1, 5), max_pages_per_day=4
    )
    pd.testing.assert_frame_equal(result, expected)


def test_month_windows_split_clamp_and_contiguous() -> None:
    from datetime import date, timedelta
    from pipeline.data_collector import _month_windows

    w = _month_windows(date(2025, 5, 22), date(2026, 1, 3))

    assert w[0] == (date(2025, 5, 22), date(2025, 5, 31))   # start-clamped
    assert w[1] == (date(2025, 6, 1), date(2025, 6, 30))
    assert w[-1] == (date(2026, 1, 1), date(2026, 1, 3))     # end-clamped
    for (_, prev_end), (next_start, _) in zip(w, w[1:]):
        assert next_start == prev_end + timedelta(days=1)     # contiguous
    for s, e in w:
        assert (s.year, s.month) == (e.year, e.month)         # each within one month


def _minute_df_for(start) -> pd.DataFrame:
    ts = pd.Timestamp(year=start.year, month=start.month, day=start.day,
                      hour=9, tz="Asia/Seoul")
    return pd.DataFrame({"Timestamp": [ts], "Open": [1.0], "High": [1.0],
                         "Low": [1.0], "Close": [1.0], "Volume": [1], "TradingValue": [1]})


def test_backfill_minute_monthly_saves_one_file_per_month(tmp_path: Path) -> None:
    from datetime import date
    from unittest.mock import Mock
    from pipeline.data_collector import DataCollector

    fetcher = Mock()
    fetcher.fetch_minute_range.side_effect = lambda start, end, max_pages_per_day: _minute_df_for(start)
    collector = DataCollector(raw_data_dir=tmp_path)

    saved = collector.backfill_minute_monthly(
        fetcher=fetcher, symbol="005930", start=date(2025, 5, 22), end=date(2025, 7, 4)
    )

    assert saved == ["2025-05", "2025-06", "2025-07"]
    assert fetcher.fetch_minute_range.call_count == 3
    for part in saved:
        assert (tmp_path / "005930" / "1m" / f"{part}.parquet").exists()


def test_backfill_minute_monthly_skips_existing_unless_overwrite(tmp_path: Path) -> None:
    from datetime import date
    from unittest.mock import Mock
    from pipeline.data_collector import DataCollector

    # Pre-create the 2025-06 partition with a sentinel.
    june_dir = tmp_path / "005930" / "1m"
    june_dir.mkdir(parents=True)
    sentinel = pd.DataFrame({"Timestamp": [pd.Timestamp("2025-06-02 09:00", tz="Asia/Seoul")],
                             "Close": [999.0]})
    sentinel.to_parquet(june_dir / "2025-06.parquet", index=False)

    fetcher = Mock()
    fetcher.fetch_minute_range.side_effect = lambda start, end, max_pages_per_day: _minute_df_for(start)
    collector = DataCollector(raw_data_dir=tmp_path)

    saved = collector.backfill_minute_monthly(
        fetcher=fetcher, symbol="005930", start=date(2025, 5, 22), end=date(2025, 7, 4)
    )

    assert saved == ["2025-05", "2025-07"]                       # June skipped
    called_starts = [c.kwargs["start"] for c in fetcher.fetch_minute_range.call_args_list]
    assert date(2025, 6, 1) not in called_starts                 # not fetched
    assert pd.read_parquet(june_dir / "2025-06.parquet")["Close"].iloc[0] == 999.0  # untouched

    # overwrite=True re-fetches every month.
    fetcher.fetch_minute_range.reset_mock()
    saved2 = collector.backfill_minute_monthly(
        fetcher=fetcher, symbol="005930", start=date(2025, 5, 22), end=date(2025, 7, 4),
        overwrite=True,
    )
    assert saved2 == ["2025-05", "2025-06", "2025-07"]
    assert fetcher.fetch_minute_range.call_count == 3


def test_save_if_changed_skips_identical_content(tmp_path: Path) -> None:
    df = _minute_df_for(date(2025, 7, 1))
    collector = DataCollector(raw_data_dir=tmp_path)
    first = collector.save_if_changed(df, symbol="005930", interval="1m",
                                      partition="2025-07", time_col="Timestamp")
    assert first is not None and first.exists()

    # 행 순서·컬럼 순서를 섞어도 내용이 같으면 스킵
    shuffled = df.iloc[::-1][list(reversed(df.columns.tolist()))]
    second = collector.save_if_changed(shuffled, symbol="005930", interval="1m",
                                       partition="2025-07", time_col="Timestamp")
    assert second is None


def test_save_if_changed_writes_on_content_or_dtype_change(tmp_path: Path) -> None:
    df = _minute_df_for(date(2025, 7, 1))
    collector = DataCollector(raw_data_dir=tmp_path)
    collector.save_if_changed(df, symbol="005930", interval="1m",
                              partition="2025-07", time_col="Timestamp")

    changed = df.copy()
    changed.loc[0, "Close"] = 2.0
    assert collector.save_if_changed(changed, symbol="005930", interval="1m",
                                     partition="2025-07", time_col="Timestamp") is not None

    dtype_drift = changed.copy()
    dtype_drift["Volume"] = dtype_drift["Volume"].astype("float64")
    assert collector.save_if_changed(dtype_drift, symbol="005930", interval="1m",
                                     partition="2025-07", time_col="Timestamp") is not None


def test_backfill_minute_monthly_overwrite_partitions_selective(tmp_path: Path) -> None:
    from unittest.mock import Mock

    # 2025-06, 2025-07 파티션이 이미 존재
    d = tmp_path / "005930" / "1m"
    d.mkdir(parents=True)
    for part, day in (("2025-06", date(2025, 6, 2)), ("2025-07", date(2025, 7, 1))):
        _minute_df_for(day).to_parquet(d / f"{part}.parquet", index=False)

    fetcher = Mock()
    # 7월 재수집 결과는 기존 일자를 보존하면서 새 거래일이 추가된 상황
    fetcher.fetch_minute_range.side_effect = lambda start, end, max_pages_per_day: (
        pd.concat([
            _minute_df_for(date(2025, 7, 1)),
            _minute_df_for(date(2025, 7, 4)),
        ], ignore_index=True) if start.month == 7 else _minute_df_for(start)
    )
    collector = DataCollector(raw_data_dir=tmp_path)

    saved = collector.backfill_minute_monthly(
        fetcher=fetcher, symbol="005930", start=date(2025, 6, 1), end=date(2025, 7, 4),
        overwrite_partitions={"2025-07"},
    )

    assert saved == ["2025-07"]                                   # 6월은 스킵
    starts = [c.kwargs["start"] for c in fetcher.fetch_minute_range.call_args_list]
    assert starts == [date(2025, 7, 1)]                            # 7월만 fetch


def test_backfill_minute_monthly_forced_partition_keeps_existing_on_shrink(
    tmp_path: Path,
) -> None:
    from unittest.mock import Mock

    d = tmp_path / "005930" / "1m"
    d.mkdir(parents=True)
    existing = _minute_df_days(2025, 7, [1, 2, 3])
    existing.to_parquet(d / "2025-07.parquet", index=False)

    fetcher = Mock()
    fetcher.fetch_minute_range.return_value = _minute_df_days(2025, 7, [2, 3])
    collector = DataCollector(raw_data_dir=tmp_path)

    saved = collector.backfill_minute_monthly(
        fetcher=fetcher, symbol="005930", start=date(2025, 7, 1), end=date(2025, 7, 4),
        overwrite_partitions={"2025-07"},
    )

    assert saved == []
    kept = pd.read_parquet(d / "2025-07.parquet")
    pd.testing.assert_frame_equal(kept, existing)


def test_backfill_minute_monthly_unchanged_refetch_not_reported(tmp_path: Path) -> None:
    from unittest.mock import Mock

    d = tmp_path / "005930" / "1m"
    d.mkdir(parents=True)
    _minute_df_for(date(2025, 7, 1)).to_parquet(d / "2025-07.parquet", index=False)

    fetcher = Mock()  # 재수집해도 동일 내용 (휴장일 시나리오)
    fetcher.fetch_minute_range.side_effect = lambda start, end, max_pages_per_day: _minute_df_for(date(2025, 7, 1))
    collector = DataCollector(raw_data_dir=tmp_path)

    before_mtime = (d / "2025-07.parquet").stat().st_mtime_ns
    saved = collector.backfill_minute_monthly(
        fetcher=fetcher, symbol="005930", start=date(2025, 7, 1), end=date(2025, 7, 4),
        overwrite_partitions={"2025-07"},
    )
    assert saved == []                                             # 저장 없음
    assert (d / "2025-07.parquet").stat().st_mtime_ns == before_mtime  # 파일 안 건드림


def _daily_df(dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": [1.0] * n, "High": [1.0] * n, "Low": [1.0] * n, "Close": [1.0] * n,
        "Volume": [1] * n, "TradingValue": [1] * n, "Change": [0.0] * n,
    })


def test_refresh_daily_all_writes_single_file_and_removes_legacy(tmp_path: Path) -> None:
    from unittest.mock import Mock

    d = tmp_path / "005930" / "1d"
    d.mkdir(parents=True)
    _daily_df(["2024-01-03"]).to_parquet(d / "2020-2026.parquet", index=False)  # 레거시

    fetcher = Mock()
    fetcher.fetch_daily.return_value = _daily_df(["2024-01-03", "2024-01-04"])
    collector = DataCollector(raw_data_dir=tmp_path)

    status = collector.refresh_daily_all(fetcher, symbol="005930",
                                         start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert status == "replaced"
    assert sorted(p.name for p in d.glob("*.parquet")) == ["all.parquet"]  # 레거시 제거됨


def test_refresh_daily_all_unchanged_still_removes_legacy(tmp_path: Path) -> None:
    from unittest.mock import Mock

    df = _daily_df(["2024-01-03", "2024-01-04"])
    d = tmp_path / "005930" / "1d"
    d.mkdir(parents=True)
    df.to_parquet(d / "all.parquet", index=False)
    _daily_df(["2024-01-03"]).to_parquet(d / "2020-2026.parquet", index=False)  # 레거시 공존

    fetcher = Mock()
    fetcher.fetch_daily.return_value = df.copy()
    collector = DataCollector(raw_data_dir=tmp_path)

    status = collector.refresh_daily_all(fetcher, symbol="005930",
                                         start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert status == "unchanged"
    assert sorted(p.name for p in d.glob("*.parquet")) == ["all.parquet"]


def test_refresh_daily_all_empty_fetch_touches_nothing(tmp_path: Path) -> None:
    from unittest.mock import Mock

    d = tmp_path / "005930" / "1d"
    d.mkdir(parents=True)
    _daily_df(["2024-01-03"]).to_parquet(d / "2020-2026.parquet", index=False)

    fetcher = Mock()
    fetcher.fetch_daily.return_value = pd.DataFrame()
    collector = DataCollector(raw_data_dir=tmp_path)

    status = collector.refresh_daily_all(fetcher, symbol="005930",
                                         start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert status == "empty"
    assert (d / "2020-2026.parquet").exists()   # fetch 실패 시 레거시도 보존


def test_force_month_refetch_uses_full_month_window(tmp_path: Path) -> None:
    from unittest.mock import Mock

    fetcher = Mock()
    fetcher.fetch_minute_range.return_value = _minute_df_days(2025, 7, [1, 2, 3])
    collector = DataCollector(raw_data_dir=tmp_path)

    result = collector.force_month_refetch(fetcher, symbol="005930", months=["2025-07"],
                                           today=date(2026, 7, 6))

    assert result == {"2025-07": "replaced"}
    fetcher.fetch_minute_range.assert_called_once_with(
        start=date(2025, 7, 1), end=date(2025, 7, 31), max_pages_per_day=4
    )
    assert (tmp_path / "005930" / "1m" / "2025-07.parquet").exists()


def test_force_month_refetch_clamps_current_month_to_today(tmp_path: Path) -> None:
    from unittest.mock import Mock

    fetcher = Mock()
    fetcher.fetch_minute_range.return_value = _minute_df_days(2026, 7, [1, 2, 3])
    collector = DataCollector(raw_data_dir=tmp_path)

    collector.force_month_refetch(fetcher, symbol="005930", months=["2026-07"],
                                  today=date(2026, 7, 6))

    fetcher.fetch_minute_range.assert_called_once_with(
        start=date(2026, 7, 1), end=date(2026, 7, 6), max_pages_per_day=4
    )


def test_force_month_refetch_coverage_guard_keeps_existing(tmp_path: Path) -> None:
    from unittest.mock import Mock

    d = tmp_path / "005930" / "1m"
    d.mkdir(parents=True)
    _minute_df_days(2025, 7, [1, 2, 3, 4, 7]).to_parquet(d / "2025-07.parquet", index=False)

    fetcher = Mock()  # rolling 경계에 잘려 3거래일만 응답
    fetcher.fetch_minute_range.return_value = _minute_df_days(2025, 7, [3, 4, 7])
    collector = DataCollector(raw_data_dir=tmp_path)

    result = collector.force_month_refetch(fetcher, symbol="005930", months=["2025-07"],
                                           today=date(2026, 7, 6))

    assert result == {"2025-07": "kept_existing"}
    kept = pd.read_parquet(d / "2025-07.parquet")
    assert kept["Timestamp"].dt.date.nunique() == 5                # 기존 데이터 보존


def test_force_month_refetch_row_count_guard_keeps_existing(tmp_path: Path) -> None:
    from unittest.mock import Mock

    d = tmp_path / "005930" / "1m"
    d.mkdir(parents=True)
    # 기존: 2거래일, 일자당 2행 = 총 4행
    existing = pd.concat([
        _minute_df_days(2025, 7, [1, 2]),
        _minute_df_days(2025, 7, [1, 2]).assign(
            Timestamp=lambda x: x["Timestamp"] + pd.Timedelta(minutes=1)),
    ]).sort_values("Timestamp").reset_index(drop=True)
    existing.to_parquet(d / "2025-07.parquet", index=False)

    fetcher = Mock()  # 같은 2거래일이지만 일자당 1행 = 총 2행 (partial 응답)
    fetcher.fetch_minute_range.return_value = _minute_df_days(2025, 7, [1, 2])
    collector = DataCollector(raw_data_dir=tmp_path)

    result = collector.force_month_refetch(fetcher, symbol="005930", months=["2025-07"],
                                           today=date(2026, 7, 6))

    assert result == {"2025-07": "kept_existing"}                  # 거래일 수 같아도 row 감소면 거부
    assert len(pd.read_parquet(d / "2025-07.parquet")) == 4


def test_force_month_refetch_per_day_guard_rejects_redistribution(tmp_path: Path) -> None:
    from unittest.mock import Mock

    d = tmp_path / "005930" / "1m"
    d.mkdir(parents=True)
    # 기존: 7/1에 2행, 7/2에 2행 (총 4행)
    existing = pd.concat([
        _minute_df_days(2025, 7, [1, 2]),
        _minute_df_days(2025, 7, [1, 2]).assign(
            Timestamp=lambda x: x["Timestamp"] + pd.Timedelta(minutes=1)),
    ]).sort_values("Timestamp").reset_index(drop=True)
    existing.to_parquet(d / "2025-07.parquet", index=False)

    # 새 응답: 7/1은 1행으로 줄고 7/2는 3행으로 늘어 총 4행 동일 (재분배)
    new = pd.concat([
        _minute_df_days(2025, 7, [1, 2]),
        _minute_df_days(2025, 7, [2]).assign(
            Timestamp=lambda x: x["Timestamp"] + pd.Timedelta(minutes=1)),
        _minute_df_days(2025, 7, [2]).assign(
            Timestamp=lambda x: x["Timestamp"] + pd.Timedelta(minutes=2)),
    ]).sort_values("Timestamp").reset_index(drop=True)

    fetcher = Mock()
    fetcher.fetch_minute_range.return_value = new
    collector = DataCollector(raw_data_dir=tmp_path)

    result = collector.force_month_refetch(fetcher, symbol="005930", months=["2025-07"],
                                           today=date(2026, 7, 6))

    assert result == {"2025-07": "kept_existing"}                  # 총 row 같아도 7/1 감소면 거부


def test_force_month_refetch_future_month_is_invalid(tmp_path: Path) -> None:
    from unittest.mock import Mock

    fetcher = Mock()
    collector = DataCollector(raw_data_dir=tmp_path)

    result = collector.force_month_refetch(fetcher, symbol="005930", months=["2026-08"],
                                           today=date(2026, 7, 6))

    assert result == {"2026-08": "invalid_future"}
    fetcher.fetch_minute_range.assert_not_called()                 # API 호출 없이 거부


def test_force_month_refetch_empty_is_unavailable(tmp_path: Path) -> None:
    from unittest.mock import Mock

    fetcher = Mock()
    fetcher.fetch_minute_range.return_value = pd.DataFrame()
    collector = DataCollector(raw_data_dir=tmp_path)

    result = collector.force_month_refetch(fetcher, symbol="005930", months=["2024-01"],
                                           today=date(2026, 7, 6))

    assert result == {"2024-01": "unavailable"}
    assert not (tmp_path / "005930" / "1m" / "2024-01.parquet").exists()
