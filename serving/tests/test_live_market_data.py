import json
from datetime import datetime, timedelta

import pandas as pd
import pytest
import responses

from errors import ProviderError
from live_market_data import LiveKISProvider
from pipeline.kis_auth import KISAuth, KISConfig

TZ = "Asia/Seoul"


class StubFetcher:
    def __init__(self, frame=None, exc=None):
        self._frame = frame
        self._exc = exc
        self.calls = []

    def fetch_minute_for_date(self, target_date):
        self.calls.append(target_date)
        if self._exc is not None:
            raise self._exc
        return self._frame


def _cached_auth(tmp_path) -> KISAuth:
    """HTTP 없이 동작하는 auth — 유효한 토큰 캐시를 미리 심는다."""
    cache = tmp_path / "tok.json"
    cache.write_text(json.dumps({
        "access_token": "cached-token",
        "expires_at": (datetime.now() + timedelta(hours=6)).isoformat(),
    }))
    return KISAuth(KISConfig(env="demo", app_key="k", app_secret="s",
                             token_cache_path=cache))


def _today_frame(day, minutes):
    ts = [pd.Timestamp(f"{day} {m}", tz=TZ) for m in minutes]
    n = len(ts)
    return pd.DataFrame({
        "Timestamp": ts,
        "Open": [100.0] * n, "High": [101.0] * n, "Low": [99.0] * n,
        "Close": [100.5] * n, "Volume": [10] * n, "TradingValue": [1000] * n,
    })


def _provider(raw_data_dir, tmp_path, stub):
    return LiveKISProvider(
        data_dir=raw_data_dir, warmup_days=5,
        auth=_cached_auth(tmp_path), rate_limit_sleep=0.0,
        fetcher_factory=lambda symbol: stub,
    )


def test_merges_history_with_today_and_applies_cutoff(raw_data_dir, minute_data, tmp_path):
    last_hist_day = pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]
    today = last_hist_day + pd.Timedelta(days=3)          # 히스토리에 없는 미래 거래일
    stub = StubFetcher(frame=_today_frame(today, ["09:00", "09:01", "09:02", "09:03"]))
    as_of = pd.Timestamp(f"{today} 09:03:30", tz=TZ)
    bars = _provider(raw_data_dir, tmp_path, stub).get_recent_bars("005930", as_of)
    ts = pd.to_datetime(bars["Timestamp"])
    # cutoff: 라벨+1분 <= as_of → 09:02까지 (09:03은 09:04에 완료)
    assert ts.max() == pd.Timestamp(f"{today} 09:02:00", tz=TZ)
    # 히스토리도 포함됨
    assert ts.dt.date.min() < today
    assert stub.calls == [today]


def test_dedupe_keeps_kis_row_over_parquet(raw_data_dir, minute_data, tmp_path):
    # 당일이 백필 파일에도 있는 겹침: KIS(최신 조회)가 이긴다
    last_day = pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]
    override = _today_frame(last_day, ["09:00"])
    override.loc[0, "Close"] = 99999.0
    stub = StubFetcher(frame=override)
    as_of = pd.Timestamp(f"{last_day} 09:05:00", tz=TZ)
    bars = _provider(raw_data_dir, tmp_path, stub).get_recent_bars("005930", as_of)
    row = bars[pd.to_datetime(bars["Timestamp"]) == pd.Timestamp(f"{last_day} 09:00", tz=TZ)]
    assert len(row) == 1 and float(row["Close"].iloc[0]) == 99999.0


def test_empty_today_returns_history_only(raw_data_dir, minute_data, tmp_path):
    from pipeline.kis_historical import MINUTE_OUTPUT_COLUMNS
    stub = StubFetcher(frame=pd.DataFrame(columns=MINUTE_OUTPUT_COLUMNS))
    last_day = pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]
    as_of = pd.Timestamp(f"{last_day} 15:40:00", tz=TZ)
    bars = _provider(raw_data_dir, tmp_path, stub).get_recent_bars("005930", as_of)
    assert not bars.empty


@pytest.mark.parametrize("exc", [RuntimeError("KIS API exhausted"),
                                 __import__("requests").ConnectionError("boom")])
def test_fetcher_failure_maps_to_provider_error(raw_data_dir, tmp_path, exc):
    stub = StubFetcher(exc=exc)
    with pytest.raises(ProviderError):
        _provider(raw_data_dir, tmp_path, stub).get_recent_bars(
            "005930", pd.Timestamp("2026-06-10 10:00", tz=TZ))


def test_fetcher_cached_per_symbol(raw_data_dir, minute_data, tmp_path):
    made = []
    def factory(symbol):
        made.append(symbol)
        return StubFetcher(frame=_today_frame("2026-06-10", ["09:00"]))
    provider = LiveKISProvider(
        data_dir=raw_data_dir, warmup_days=5,
        auth=_cached_auth(tmp_path), rate_limit_sleep=0.0,
        fetcher_factory=factory,
    )
    as_of = pd.Timestamp("2026-06-10 10:00", tz=TZ)
    provider.get_recent_bars("005930", as_of)
    provider.get_recent_bars("005930", as_of)
    assert made == ["005930"]                       # 심볼당 fetcher 1개 캐시


def test_check_ready_uses_cached_token(raw_data_dir, tmp_path):
    # 유효 캐시가 있으면 HTTP 없이 통과 (responses.activate로 모든 HTTP 차단)
    provider = _provider(raw_data_dir, tmp_path, StubFetcher(frame=None))
    with responses.RequestsMock():
        provider.check_ready("005930")              # no raise, no HTTP


def test_check_ready_token_failure_maps_to_provider_error(raw_data_dir, tmp_path):
    auth = KISAuth(KISConfig(env="demo", app_key="k", app_secret="s",
                             token_cache_path=tmp_path / "missing.json"))
    provider = LiveKISProvider(
        data_dir=raw_data_dir, warmup_days=5, auth=auth, rate_limit_sleep=0.0,
        fetcher_factory=lambda s: StubFetcher(frame=None),
    )
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST,
                 "https://openapivts.koreainvestment.com:29443/oauth2/tokenP",
                 status=500)
        with pytest.raises(ProviderError):
            provider.check_ready("005930")
