import pandas as pd
import pytest

from errors import ProviderError
from market_data import HistoricalParquetProvider

TZ = "Asia/Seoul"


def _provider(raw_data_dir, **kw):
    return HistoricalParquetProvider(data_dir=raw_data_dir, **kw)


def test_completed_minute_cutoff(raw_data_dir, minute_data):
    # 1분봉 라벨=시작 시각 → 완료 조건: 라벨 + 1분 <= as_of
    day = pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]
    as_of = pd.Timestamp(f"{day} 10:05:00", tz=TZ)
    bars = _provider(raw_data_dir).get_recent_bars("005930", as_of)
    last = pd.to_datetime(bars["Timestamp"]).max()
    assert last == pd.Timestamp(f"{day} 10:04:00", tz=TZ)  # 10:05 bar는 미완료라 제외


def test_cutoff_mid_minute_excludes_forming_bar(raw_data_dir, minute_data):
    day = pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]
    as_of = pd.Timestamp(f"{day} 10:05:30", tz=TZ)
    bars = _provider(raw_data_dir).get_recent_bars("005930", as_of)
    assert pd.to_datetime(bars["Timestamp"]).max() == pd.Timestamp(f"{day} 10:04:00", tz=TZ)


def test_warmup_window_limits_days(raw_data_dir, minute_data):
    day = pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]
    as_of = pd.Timestamp(f"{day} 15:40:00", tz=TZ)
    bars = _provider(raw_data_dir, warmup_days=3).get_recent_bars("005930", as_of)
    assert pd.to_datetime(bars["Timestamp"]).dt.date.nunique() == 3


def test_missing_symbol_dir_raises_provider_error(raw_data_dir):
    with pytest.raises(ProviderError):
        _provider(raw_data_dir).get_recent_bars(
            "999999", pd.Timestamp("2026-06-10 10:00", tz=TZ))


def test_corrupt_parquet_raises_provider_error(raw_data_dir):
    bad = raw_data_dir / "005930" / "1m" / "2026-05.parquet"
    bad.write_bytes(b"not a parquet")
    with pytest.raises(ProviderError):
        _provider(raw_data_dir).get_recent_bars(
            "005930", pd.Timestamp("2026-06-10 10:00", tz=TZ))


def test_check_ready(raw_data_dir):
    _provider(raw_data_dir).check_ready("005930")  # no raise
    with pytest.raises(ProviderError):
        _provider(raw_data_dir).check_ready("999999")


def test_check_ready_rejects_corrupt_latest_file(raw_data_dir):
    files = sorted((raw_data_dir / "005930" / "1m").glob("*.parquet"))
    files[-1].write_bytes(b"not a parquet")
    with pytest.raises(ProviderError):
        _provider(raw_data_dir).check_ready("005930")


def test_check_ready_rejects_missing_column(raw_data_dir):
    files = sorted((raw_data_dir / "005930" / "1m").glob("*.parquet"))
    df = pd.read_parquet(files[-1]).drop(columns=["TradingValue"])
    df.to_parquet(files[-1])
    with pytest.raises(ProviderError):
        _provider(raw_data_dir).check_ready("005930")


def test_get_recent_bars_missing_column_raises_provider_error(raw_data_dir):
    # 월 파일이 여러 개일 수 있다 — 전부 손상시켜야 concat이 컬럼을 되살리지 못한다
    for target in sorted((raw_data_dir / "005930" / "1m").glob("*.parquet")):
        df = pd.read_parquet(target).drop(columns=["TradingValue"])
        df.to_parquet(target)
    with pytest.raises(ProviderError):
        _provider(raw_data_dir).get_recent_bars(
            "005930", pd.Timestamp("2026-06-10 10:00", tz=TZ))
