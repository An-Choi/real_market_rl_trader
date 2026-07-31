"""Live KIS provider — 히스토리(로컬 백필 parquet) + 당일(KIS 조회) 병합.

spec §1: 당일 분봉도 backfill과 같은 TR·같은 정규화(KISHistoricalFetcher)로
가져온다 — 데이터 소스 parity. stateless: 매 요청 당일 전체 재조회(1~4콜).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from errors import ProviderError
from market_data import HistoricalParquetProvider, _validate_frame
from pipeline.kis_historical import KISHistoricalFetcher

_MINUTE = pd.Timedelta(minutes=1)
_KST = "Asia/Seoul"


class LiveKISProvider:
    def __init__(
        self,
        data_dir: Path,
        warmup_days: int,
        auth,
        rate_limit_sleep: float,
        fetcher_factory=None,
    ) -> None:
        self._historical = HistoricalParquetProvider(data_dir, warmup_days=warmup_days)
        self._auth = auth
        self._fetcher_factory = fetcher_factory or (
            lambda symbol: KISHistoricalFetcher(
                auth=self._auth, symbol=symbol, rate_limit_sleep=rate_limit_sleep
            )
        )
        # KISHistoricalFetcher는 symbol이 생성자에 고정 — 심볼당 1개 캐시, auth는 공유
        self._fetchers: dict = {}

    def _fetcher(self, symbol: str):
        if symbol not in self._fetchers:
            self._fetchers[symbol] = self._fetcher_factory(symbol)
        return self._fetchers[symbol]

    def check_ready(self, symbol: str) -> None:
        self._historical.check_ready(symbol)
        try:
            self._auth.get_token()  # 캐시 우선 — 강제 재발급 아님 (spec §1)
        except (RuntimeError, requests.RequestException, KeyError) as exc:
            raise ProviderError(f"KIS token check failed: {exc}") from exc

    def get_recent_bars(self, symbol: str, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        if as_of.tzinfo is None:
            raise ProviderError(f"as_of must be timezone-aware: {as_of!r}")
        hist = self._historical.get_recent_bars(symbol, as_of)
        try:
            today = self._fetcher(symbol).fetch_minute_for_date(
                as_of.tz_convert(_KST).date()
            )
        except (RuntimeError, requests.RequestException) as exc:
            raise ProviderError(f"KIS minute fetch failed: {exc}") from exc

        if today.empty:
            return hist
        _validate_frame(today)
        merged = pd.concat([hist, today], ignore_index=True)
        merged = (
            merged.drop_duplicates(subset=["Timestamp"], keep="last")
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
        ts = pd.to_datetime(merged["Timestamp"])
        # 완료 분봉 causal cutoff — Historical과 동일 규칙 재적용 (당일분 포함)
        merged = merged[ts + _MINUTE <= as_of]
        ts = pd.to_datetime(merged["Timestamp"])
        # 익일 재계산(HistoricalParquetProvider)과 동일한 "마지막 warmup_days개
        # 고유 날짜" 재trim — 안 하면 hist(어제까지 warmup_days개) + 당일 = warmup_days+1개
        # 고유 날짜가 되어, EWM 등 선행 히스토리 전체에 의존하는 feature가
        # 익일 재계산과 미세하게 어긋난다 (bit-exact parity 불변식 위반).
        recent_dates = sorted(ts.dt.date.unique())[-self._historical.warmup_days:]
        merged = merged[ts.dt.date.isin(recent_dates)]
        return merged.reset_index(drop=True)


def build_provider(config):
    """config.provider에 따른 provider 팩토리 — main.py 기동 경로 전용.

    live에서 KISConfig.from_env 실패(환경변수 누락)는 변환하지 않는다:
    503이 아니라 서버가 뜨지 않아야 하는 기동 실패다 (spec §1).
    """
    from config import ServingConfig  # 순환 없음 — 타입 문서화용
    if config.provider == "historical":
        from market_data import HistoricalParquetProvider

        return HistoricalParquetProvider(config.data_dir, warmup_days=config.warmup_days)
    from pipeline.kis_auth import KISAuth, KISConfig

    kis_config = KISConfig.from_env(config.kis_token_cache)
    return LiveKISProvider(
        data_dir=config.data_dir,
        warmup_days=config.warmup_days,
        auth=KISAuth(kis_config),
        rate_limit_sleep=config.kis_rate_limit_sleep,
    )
