"""30분 context feature: continuous 30분 bar 기반 trend/MACD/vol regime.

30분 bar는 거래일을 가로질러 continuous(overnight 이어붙임)하되 과거만 사용.
주말/공휴일 시계열 단절은 의도적으로 무시(빈 bucket 제거로 인접 배치).
5분 그리드 정렬은 거래일 내 shift+ffill로 완료 라벨 이후에만 노출(leakage-safe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MARKET_OPEN = pd.Timestamp("09:00").time()
_MARKET_CLOSE = pd.Timestamp("15:30").time()


def build_30min_bars(minute_df: pd.DataFrame, ts_col: str = "Timestamp") -> pd.DataFrame:
    """continuous 30분 bar [Timestamp, Close]. 1분봉 30개 미만 bucket drop, 장중 라벨만."""
    idx = minute_df.set_index(pd.to_datetime(minute_df[ts_col]))
    counts = idx["Close"].resample("30min", label="right", closed="left").count()
    close = idx["Close"].resample("30min", label="right", closed="left").last()
    bars = pd.DataFrame({"Timestamp": close.index, "Close": close.values})
    bars = bars[counts.values >= 30]
    t = pd.to_datetime(bars["Timestamp"]).dt.time
    bars = bars[(t > _MARKET_OPEN) & (t <= _MARKET_CLOSE)]
    return bars.reset_index(drop=True)


def _compute_30min_features(bars: pd.DataFrame, eps: float) -> pd.DataFrame:
    out = bars.copy()
    close = out["Close"]
    log_ret = np.log(close / close.shift(1))
    out["trend_strength_30m"] = log_ret.rolling(6, min_periods=6).sum()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist_30m"] = (macd - signal) / (close + eps)
    rv_short = log_ret.rolling(6, min_periods=6).std(ddof=1)
    rv_long = log_ret.rolling(20, min_periods=20).std(ddof=1)
    out["vol_regime_30m"] = rv_short / (rv_long + eps)
    return out


_CTX_COLS = ["trend_strength_30m", "macd_hist_30m", "vol_regime_30m"]


def add_context_features(
    bars_5min: pd.DataFrame,
    minute_df: pd.DataFrame,
    ts_col: str = "Timestamp",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """5분 그리드에 30분 context 3개를 leakage-safe하게 정렬."""
    bars30 = build_30min_bars(minute_df, ts_col)
    feats30 = _compute_30min_features(bars30, eps)[["Timestamp"] + _CTX_COLS]

    out = bars_5min.copy()
    out_ts = pd.to_datetime(out[ts_col])
    feat_ts = pd.to_datetime(feats30["Timestamp"])

    # 30분 bar는 라벨(완료시각) '초과' 시점부터 노출 → merge_asof(strictly less).
    left = out.assign(_ts=out_ts).sort_values("_ts")
    right = feats30.assign(_ts=feat_ts).sort_values("_ts")
    merged = pd.merge_asof(
        left, right[["_ts"] + _CTX_COLS], on="_ts",
        direction="backward", allow_exact_matches=False,
    )
    # continuous 30분 bar이므로 전날 완료 bar가 자연스럽게 lookback을 채운다(과거 bar = leakage 아님).
    # 데이터셋 맨 앞 warm-up 구간(21 close bar 미만)만 NaN으로 남고, transform()의 dropna가 제거한다.
    return merged.drop(columns=["_ts"]).reset_index(drop=True)
