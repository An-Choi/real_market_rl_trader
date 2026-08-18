import json
from datetime import datetime, timedelta

import pandas as pd
import pytest
import responses

from errors import ProviderError
from live_market_data import LiveKISProvider
from market_data import HistoricalParquetProvider
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


def test_live_window_matches_next_day_recompute_bit_exact(
        raw_data_dir, minute_data, tmp_path, tiny_artifact_dir):
    """당일 서빙(live)과 익일 backfill 후 재계산(historical)의 observation이
    bit-exact 일치해야 한다 — 아니면 shadow diff가 영원히 값 불일치로 exit 1.

    live: warmup_days=N → 어제까지 마지막 N−1개 고유 날짜(parquet) + 당일(KIS stub)
          = N개 고유 날짜가 되어야 한다 (재trim 없으면 N+1개가 되어 EWM feature가
          어긋난다).
    recompute: 당일을 포함한 전체 parquet에서 HistoricalParquetProvider가
          동일 warmup_days로 마지막 N개 고유 날짜를 고른다.
    """
    from data.feature_engineer import FeatureEngineer
    from friction.friction_model import FrictionModel
    from models.artifact import load_metadata
    from observation_builder import build_decision_inputs

    all_dates = sorted(pd.to_datetime(minute_data["Timestamp"]).dt.date.unique())
    today = all_dates[-1]
    hist_only = minute_data[pd.to_datetime(minute_data["Timestamp"]).dt.date < today]
    today_frame = minute_data[pd.to_datetime(minute_data["Timestamp"]).dt.date == today]

    # live가 기동 시점에 보는 backfill 상태: 당일 이전까지만 parquet로 존재
    without_today_dir = tmp_path / "without_today"
    ts = pd.to_datetime(hist_only["Timestamp"])
    for period, grp in hist_only.groupby(ts.dt.to_period("M")):
        out = without_today_dir / "005930" / "1m"
        out.mkdir(parents=True, exist_ok=True)
        grp.reset_index(drop=True).to_parquet(out / f"{period}.parquet")

    # cross-day warm-up(20거래일)을 지나 feature 행이 나오도록 25일
    # (fixture 26일 중 당일 제외 25일이 상한)
    warmup_days = 25
    as_of = pd.Timestamp(f"{today} 10:35:00", tz=TZ)

    live_provider = LiveKISProvider(
        data_dir=without_today_dir, warmup_days=warmup_days,
        auth=_cached_auth(tmp_path), rate_limit_sleep=0.0,
        fetcher_factory=lambda symbol: StubFetcher(frame=today_frame.reset_index(drop=True)),
    )
    live_bars = live_provider.get_recent_bars("005930", as_of)

    # 익일 backfill 후 상태: raw_data_dir는 이미 당일을 포함한 전체 parquet
    reco_provider = HistoricalParquetProvider(raw_data_dir, warmup_days=warmup_days)
    reco_bars = reco_provider.get_recent_bars("005930", as_of)

    live_dates = sorted(pd.to_datetime(live_bars["Timestamp"]).dt.date.unique())
    reco_dates = sorted(pd.to_datetime(reco_bars["Timestamp"]).dt.date.unique())
    assert live_dates == reco_dates
    assert len(live_dates) == warmup_days

    meta = load_metadata(tiny_artifact_dir)
    friction = FrictionModel(**meta.friction_params)
    fe = FeatureEngineer()

    live_result = build_decision_inputs(
        bars_1m=live_bars, as_of=as_of,
        units_held=0, shares_held=0.0, bars_since_entry=0,
        available_cash=10_000.0, env_params=meta.env_params,
        friction_model=friction, max_bar_age=pd.Timedelta(minutes=10),
        feature_engineer=fe,
    )
    reco_result = build_decision_inputs(
        bars_1m=reco_bars, as_of=as_of,
        units_held=0, shares_held=0.0, bars_since_entry=0,
        available_cash=10_000.0, env_params=meta.env_params,
        friction_model=friction, max_bar_age=pd.Timedelta(minutes=10),
        feature_engineer=fe,
    )
    import numpy as np
    np.testing.assert_array_equal(live_result.observation, reco_result.observation)
