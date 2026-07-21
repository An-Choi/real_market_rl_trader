"""1분봉 → 5분/30분 causal resample 및 분봉 거래대금 변환."""

from __future__ import annotations

import pandas as pd

# 종가 동시호가 print는 마지막 정규봉(15:19) 이후 gap을 두고 15:20~15:30 사이에 찍힌다.
_AUCTION_GAP_START = pd.Timestamp("15:19").time()


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


def _split_one_day(day_df: pd.DataFrame, ts_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """거래일 내 종가 동시호가 print를 정규봉과 분리한다.

    15:19 이후 >1분 gap 뒤에 찍힌 봉(들)을 종가 동시호가로 본다(수능일처럼
    장시간이 순연된 날도 gap 기준이라 동일하게 동작). 경매 print가 없으면
    (원본, 빈 프레임).
    """
    day = day_df.sort_values(ts_col).reset_index(drop=True)
    ts = pd.to_datetime(day[ts_col])
    gap = ts.diff()
    prev_ts = ts.shift(1)
    prev_after_auction = prev_ts.apply(
        lambda t: pd.notna(t) and t.time() >= _AUCTION_GAP_START
    )
    is_auction = (gap > pd.Timedelta(minutes=1)) & prev_after_auction
    if not is_auction.any():
        return day, day.iloc[0:0]

    first_auction = is_auction.idxmax()  # 첫 True 위치
    regular = day.iloc[:first_auction]
    if regular.empty:
        return day, day.iloc[0:0]  # 방어: 정규봉이 없으면 분리하지 않음
    return regular, day.iloc[first_auction:]


def split_closing_auction(
    minute_df: pd.DataFrame, ts_col: str = "Timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """거래일별로 (정규봉, 종가 동시호가 print)를 분리해 반환한다.

    경매 가격은 관찰(feature)에 넣으면 lookahead가 되므로 정규봉만 resample로
    보내고, 경매 print는 실행 전용(ExecPrice) 데이터로 쓴다. 입력 미변경.
    """
    ts = pd.to_datetime(minute_df[ts_col])
    parts = [_split_one_day(grp, ts_col) for _, grp in minute_df.groupby(ts.dt.date)]
    regular = pd.concat([r for r, _ in parts], ignore_index=True)
    auction = pd.concat([a for _, a in parts], ignore_index=True)
    return regular, auction


_AGG_5MIN = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
    "MinuteTradingValue": "sum",
}


def _resample_one_day(day_df: pd.DataFrame, rule: str, min_bars: int, ts_col: str) -> pd.DataFrame:
    idx = day_df.set_index(pd.to_datetime(day_df[ts_col]))
    counts = idx["Close"].resample(rule, label="right", closed="left").count()
    agg = idx.resample(rule, label="right", closed="left").agg(_AGG_5MIN)
    agg = agg[counts >= min_bars].dropna(subset=["Close"])
    agg = agg.reset_index().rename(columns={"index": ts_col})
    if ts_col not in agg.columns:  # resample index name 방어
        agg = agg.rename(columns={agg.columns[0]: ts_col})
    return agg


def resample_5min(minute_df: pd.DataFrame, ts_col: str = "Timestamp") -> pd.DataFrame:
    """거래일별 5분 causal resample. label=right, closed=left, 불완전 bucket drop."""
    df = minute_df
    if "MinuteTradingValue" not in df.columns:
        df = add_minute_trading_value(df, ts_col)
    ts = pd.to_datetime(df[ts_col])
    frames = [
        _resample_one_day(grp, "5min", min_bars=5, ts_col=ts_col)
        for _, grp in df.groupby(ts.dt.date)
    ]
    return pd.concat(frames, ignore_index=True)
