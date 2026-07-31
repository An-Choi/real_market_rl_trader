"""시장 데이터 provider — spec §1. Task 4(실시간 KIS)는 같은 인터페이스를 구현한다."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from errors import ProviderError

_MINUTE = pd.Timedelta(minutes=1)
_REQUIRED_COLUMNS = ("Timestamp", "Open", "High", "Low", "Close", "Volume", "TradingValue")


class MarketDataProvider(Protocol):
    def get_recent_bars(self, symbol: str, as_of: pd.Timestamp) -> pd.DataFrame: ...
    def check_ready(self, symbol: str) -> None: ...


def _validate_frame(bars: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ProviderError(f"raw 1m parquet missing columns: {missing}")
    if pd.to_datetime(bars["Timestamp"]).dt.tz is None:
        raise ProviderError("raw 1m Timestamps must be timezone-aware")


class HistoricalParquetProvider:
    """data/raw/<symbol>/1m/YYYY-MM.parquet 기반 (backfill 산출물)."""

    def __init__(self, data_dir: Path, warmup_days: int = 30) -> None:
        self.data_dir = Path(data_dir)
        self.warmup_days = warmup_days

    def check_ready(self, symbol: str) -> None:
        """파일 존재 + 최신 파일 실제 읽기 + 스키마·tz 검증 (얕은 glob만으로는
        parquet 손상·컬럼 누락·naive tz가 /ready를 통과해버린다)."""
        files = self._files(symbol)
        if not files:
            raise ProviderError(
                f"no 1m parquet files for symbol {symbol}",
                {"data_dir": str(self.data_dir)},
            )
        try:
            sample = pd.read_parquet(files[-1])
        except Exception as exc:
            raise ProviderError(f"failed to read {files[-1].name}: {exc}") from exc
        _validate_frame(sample)

    def _files(self, symbol: str) -> list[Path]:
        return sorted((self.data_dir / symbol / "1m").glob("*.parquet"))

    def get_recent_bars(self, symbol: str, as_of: pd.Timestamp) -> pd.DataFrame:
        files = self._files(symbol)
        if not files:
            raise ProviderError(
                f"no 1m parquet files for symbol {symbol}",
                {"data_dir": str(self.data_dir)},
            )
        try:
            frames = [pd.read_parquet(f) for f in files]
        except Exception as exc:
            raise ProviderError(f"failed to read 1m parquet: {exc}") from exc
        bars = pd.concat(frames, ignore_index=True)
        _validate_frame(bars)  # 컬럼 누락·naive tz → 구조화된 ProviderError (500 방지)
        ts = pd.to_datetime(bars["Timestamp"])
        as_of = pd.Timestamp(as_of)
        if as_of.tzinfo is None:
            raise ProviderError(f"as_of must be timezone-aware: {as_of!r}")
        # causal cutoff: 완료된 1분봉만 (라벨 = bar 시작 시각)
        bars = bars[ts + _MINUTE <= as_of]
        ts = pd.to_datetime(bars["Timestamp"])
        recent_dates = sorted(ts.dt.date.unique())[-self.warmup_days:]
        bars = bars[ts.dt.date.isin(recent_dates)]
        return bars.sort_values("Timestamp").reset_index(drop=True)
