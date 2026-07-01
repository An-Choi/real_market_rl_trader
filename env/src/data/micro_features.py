"""5분 micro feature: log returns, realized vol, relative volume, VWAP deviation.

모든 연산은 groupby(date) 내부(day-reset) — overnight 누출 없음.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_micro_features(
    bars_5min: pd.DataFrame, ts_col: str = "Timestamp", eps: float = 1e-8
) -> pd.DataFrame:
    """5분 micro feature 6개 추가(day-reset). warm-up은 NaN."""
    out = bars_5min.copy()
    date = pd.to_datetime(out[ts_col]).dt.date
    close = out["Close"]
    grp_close = close.groupby(date)

    out["log_ret_1"] = grp_close.transform(lambda s: np.log(s / s.shift(1)))
    out["log_ret_3"] = grp_close.transform(lambda s: np.log(s / s.shift(3)))
    out["log_ret_12"] = grp_close.transform(lambda s: np.log(s / s.shift(12)))
    out["realized_vol_12"] = out["log_ret_1"].groupby(date).transform(
        lambda s: s.rolling(12, min_periods=12).std(ddof=1)
    )
    out["relative_volume"] = out["Volume"].groupby(date).transform(
        lambda s: s / (s.rolling(12, min_periods=12).mean().shift(1) + eps)
    )

    cum_tv = out["MinuteTradingValue"].groupby(date).cumsum()
    cum_vol = out["Volume"].groupby(date).cumsum()
    vwap = cum_tv / (cum_vol + eps)
    out["vwap_dev"] = (out["Close"] - vwap) / (vwap + eps)
    return out
