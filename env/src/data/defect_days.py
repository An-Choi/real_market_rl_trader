"""거래일 결손 판정: late-start · 장중 gap 기반 whole-day drop."""

from __future__ import annotations

import pandas as pd

MARKET_OPEN = pd.Timestamp("09:00").time()
# 허용 gap: 마지막 정규 봉(15:19) → 종가 동시호가(15:30). 그 외 장중 >1min gap은 결손.
_AUCTION_GAP_START = pd.Timestamp("15:19").time()


def is_valid_trading_day(day_df: pd.DataFrame, ts_col: str = "Timestamp") -> bool:
    """단일 거래일이 유효한지 판정.

    ① 첫 봉이 09:00이 아니면 invalid(late-start, allowlist 없음).
    ② 허용 gap(15:20~15:30 동시호가) 외 장중 >1분 gap이 있으면 invalid.
    """
    ts = pd.to_datetime(day_df[ts_col]).sort_values().reset_index(drop=True)
    if ts.empty:
        return False
    if ts.iloc[0].time() != MARKET_OPEN:
        return False
    diffs = ts.diff().dropna()
    for prev, gap in zip(ts.iloc[:-1], diffs):
        if gap > pd.Timedelta(minutes=1):
            # 15:19 이후에서 벌어진 gap만 동시호가로 허용
            if prev.time() >= _AUCTION_GAP_START:
                continue
            return False
    return True


def drop_defect_days(minute_df: pd.DataFrame, ts_col: str = "Timestamp") -> pd.DataFrame:
    """유효 거래일의 행만 남기고 index reset."""
    ts = pd.to_datetime(minute_df[ts_col])
    keep_dates = {
        day
        for day, grp in minute_df.groupby(ts.dt.date)
        if is_valid_trading_day(grp, ts_col)
    }
    mask = ts.dt.date.isin(keep_dates)
    return minute_df[mask].reset_index(drop=True)
