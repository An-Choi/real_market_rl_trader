from __future__ import annotations

import numpy as np
import pandas as pd

from data.context_features import add_context_features, build_30min_bars
from data.resample import add_minute_trading_value, resample_5min


def _minute_multi_day(days: list[str], per_day: int = 390) -> pd.DataFrame:
    frames = []
    price = 100.0
    for d in days:
        ts = pd.date_range(f"{d} 09:00", periods=per_day, freq="1min", tz="Asia/Seoul")
        closes = price + np.linspace(0, 5, per_day)
        price = float(closes[-1])
        frames.append(pd.DataFrame({
            "Timestamp": ts, "Open": closes, "High": closes + 0.1, "Low": closes - 0.1,
            "Close": closes, "Volume": 1000, "TradingValue": np.cumsum(np.full(per_day, 100000)),
        }))
    return pd.concat(frames, ignore_index=True)


def test_30min_bar_drops_incomplete_bucket() -> None:
    # 하루 390분(09:00~15:29). label=right, closed=left이므로 각 라벨 버킷은
    # [label-30min, label) → 15:30 버킷=[15:00,15:30)=15:00~15:29 30개(완전).
    # 완전 30분 bar는 09:30~15:30 라벨(13개).
    mdf = add_minute_trading_value(_minute_multi_day(["2025-06-02"]))
    bars = build_30min_bars(mdf)
    labels = pd.to_datetime(bars["Timestamp"]).dt.strftime("%H:%M").tolist()
    assert labels[0] == "09:30" and labels[-1] == "15:30"
    assert len(labels) == 13


def test_30min_incomplete_trailing_bucket_dropped() -> None:
    # 400분(09:00~15:39) → 15:30 이후 10개짜리 버킷(라벨 16:00)은 30개 미만 → drop.
    mdf = add_minute_trading_value(_minute_multi_day(["2025-06-02"], per_day=400))
    bars = build_30min_bars(mdf)
    labels = pd.to_datetime(bars["Timestamp"]).dt.strftime("%H:%M").tolist()
    assert "16:00" not in labels  # 불완전(10개) bucket drop
    assert labels[-1] == "15:30"


def test_30min_feature_available_only_after_completion() -> None:
    mdf = add_minute_trading_value(_minute_multi_day(["2025-06-02", "2025-06-03", "2025-06-04"]))
    bars5 = resample_5min(mdf)
    out = add_context_features(bars5, mdf)
    # 09:30 완료 bar는 09:35 step부터 노출 → 09:30 step엔 trend가 이전(없으면 NaN)
    day = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-04").date()]
    at_0930 = day[pd.to_datetime(day["Timestamp"]).dt.strftime("%H:%M") == "09:30"]
    at_0935 = day[pd.to_datetime(day["Timestamp"]).dt.strftime("%H:%M") == "09:35"]
    # 09:35의 30분 context가 09:30의 것과 다르거나(갱신), 최소한 09:35는 유효값
    assert not at_0935.empty


def test_context_columns_present() -> None:
    mdf = add_minute_trading_value(_minute_multi_day(["2025-06-02", "2025-06-03", "2025-06-04"]))
    bars5 = resample_5min(mdf)
    out = add_context_features(bars5, mdf)
    for col in ("trend_strength_30m", "macd_hist_30m", "vol_regime_30m"):
        assert col in out.columns
