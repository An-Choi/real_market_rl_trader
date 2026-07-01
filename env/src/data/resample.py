"""1분봉 → 5분/30분 causal resample 및 분봉 거래대금 변환."""

from __future__ import annotations

import pandas as pd


def add_minute_trading_value(minute_df: pd.DataFrame, ts_col: str = "Timestamp") -> pd.DataFrame:
    """누적 TradingValue를 거래일별 diff로 분봉 거래대금(MinuteTradingValue)으로 변환.

    whole-frame diff는 거래일 경계에서 전날 누적을 빼 거대 음수를 만든다.
    반드시 groupby(date).diff() + 각 날 첫 봉을 그 봉 누적값으로 대입.
    """
    out = minute_df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["MinuteTradingValue"] = out.groupby(ts.dt.date)["TradingValue"].diff()
    first_of_day = out["MinuteTradingValue"].isna()
    out.loc[first_of_day, "MinuteTradingValue"] = out.loc[first_of_day, "TradingValue"]
    return out
