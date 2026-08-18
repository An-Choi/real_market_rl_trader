"""cross-day feature: gap_open, relative_volume_tod + Adv20 pass-through.

day-reset 원칙의 의도적 완화 — **완료된 과거 거래일만** 참조한다(당일 값은
분모·기준값에 절대 불포함). micro/context와 동일하게 입력은 거래일 오름차순
정렬 전제(파이프라인 계약), 순수 함수.

- gap_open: log(당일 첫 5분 bar Open / 전일 종가). 전일 종가는 동시호가
  체결가 우선, 없으면 전일 마지막 정규 5분 bar Close. 하루 상수.
- relative_volume_tod: 슬롯 거래량 / 직전 lookback_days일 동일 슬롯 중앙값.
  기존 relative_volume(당일 rolling)이 못 잡는 시간대 U-shape을 보정.
- Adv20: 직전 lookback_days일 일평균 거래대금(정규장 기준). feature가 아닌
  ExecPrice류 pass-through — env가 portfolio state(유동성) 계산에 사용.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_day_features(
    bars_5min: pd.DataFrame,
    auction: pd.DataFrame,
    ts_col: str = "Timestamp",
    lookback_days: int = 20,
    eps: float = 1e-8,
) -> pd.DataFrame:
    out = bars_5min.copy()
    ts = pd.to_datetime(out[ts_col])
    date = ts.dt.date

    # gap_open — 전일 종가: 동시호가 우선, 없으면 정규 마지막 Close
    day_close = out.groupby(date)["Close"].last().astype(float)
    if not auction.empty:
        auction_close = (
            auction.groupby(pd.to_datetime(auction[ts_col]).dt.date)["Close"]
            .last()
            .astype(float)
        )
        day_close.update(auction_close)
    day_open = out.groupby(date)["Open"].first().astype(float)
    gap = np.log(day_open / day_close.shift(1))
    out["gap_open"] = date.map(gap)

    # relative_volume_tod — 동일 슬롯의 직전 N일 중앙값 대비(당일 제외: shift(1))
    slot_median = out.groupby(ts.dt.time)["Volume"].transform(
        lambda s: s.shift(1).rolling(lookback_days, min_periods=lookback_days).median()
    )
    out["relative_volume_tod"] = out["Volume"] / (slot_median + eps)

    # Adv20 — 직전 N일 일평균 거래대금(당일 제외)
    daily_tv = out.groupby(date)["MinuteTradingValue"].sum()
    adv = daily_tv.shift(1).rolling(lookback_days, min_periods=lookback_days).mean()
    out["Adv20"] = date.map(adv)
    return out
