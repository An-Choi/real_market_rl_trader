"""MTF feature 파이프라인 오케스트레이션 + registry freeze.

파이프라인 순서(불변): 결손일 drop → 거래일별 MinuteTradingValue diff →
종가 동시호가 분리 → 5분 resample → 5분 micro feature → 30분 context feature →
ExecPrice(실행 전용) 부착 → warm-up dropna.
transform()은 순수 함수(self·입력 미변경).
"""

from __future__ import annotations

import pandas as pd

from data.context_features import add_context_features
from data.defect_days import drop_defect_days
from data.micro_features import add_micro_features
from data.resample import add_minute_trading_value, resample_5min, split_closing_auction


def _add_exec_price(
    bars5: pd.DataFrame,
    regular: pd.DataFrame,
    auction: pd.DataFrame,
    ts_col: str,
) -> pd.DataFrame:
    """시장가 주문 근사 체결가(ExecPrice)를 5분 그리드에 부착한다.

    규칙: 라벨 시각 T 이후 같은 날 첫 정규 1분봉의 Open → 없으면(장 마감 결정)
    그날 종가 동시호가 단일가 → 그것도 없으면 NaN (env가 Close로 fallback).
    ExecPrice는 결정 시점 이후의 미래 가격이므로 절대 관찰(feature)에 넣지 않는다.
    """
    out = bars5.copy()
    label_ts = pd.to_datetime(out[ts_col])

    reg = pd.DataFrame({
        "__ts": pd.to_datetime(regular[ts_col]),
        "__date": pd.to_datetime(regular[ts_col]).dt.date,
        "ExecPrice": regular["Open"].astype(float),
    }).sort_values("__ts")

    keyed = out.assign(__ts=label_ts, __date=label_ts.dt.date).sort_values("__ts")
    merged = pd.merge_asof(
        keyed, reg,
        on="__ts", by="__date",
        direction="forward", allow_exact_matches=True,
    )

    if not auction.empty:
        auction_ts = pd.to_datetime(auction[ts_col])
        auction_price = auction.groupby(auction_ts.dt.date)["Close"].last().astype(float)
        missing = merged["ExecPrice"].isna()
        merged.loc[missing, "ExecPrice"] = merged.loc[missing, "__date"].map(auction_price)

    return merged.drop(columns=["__ts", "__date"]).reset_index(drop=True)


class FeatureEngineer:
    # v3: 체결 계약 변경 — 경매 fold-in 제거(관찰 정화) + ExecPrice(실행 전용) 추가.
    FEATURE_SCHEMA_VERSION: int = 3
    FEATURE_COLUMNS: tuple[str, ...] = (
        "log_ret_1", "log_ret_3", "log_ret_12",
        "realized_vol_12", "relative_volume", "vwap_dev",
        "trend_strength_30m", "macd_hist_30m", "vol_regime_30m",
    )

    def transform(self, minute_df: pd.DataFrame, ts_col: str = "Timestamp") -> pd.DataFrame:
        """1분봉(멀티데이) → 5분 그리드 [Timestamp, Close, ExecPrice, <FEATURE_COLUMNS>].

        모든 feature lookback 만족 행만(NaN drop; ExecPrice의 NaN은 유지 —
        env fallback 담당). 순수 함수: self 미변경, 입력 미변경.
        """
        clean = drop_defect_days(minute_df, ts_col)
        clean = add_minute_trading_value(clean, ts_col)
        regular, auction = split_closing_auction(clean, ts_col)
        bars5 = resample_5min(regular, ts_col)
        bars5 = add_micro_features(bars5, ts_col)
        bars5 = add_context_features(bars5, regular, ts_col)
        bars5 = _add_exec_price(bars5, regular, auction, ts_col)

        cols = [ts_col, "Close", "ExecPrice", *self.FEATURE_COLUMNS]
        result = bars5[cols].dropna(subset=list(self.FEATURE_COLUMNS))
        return result.reset_index(drop=True)

    # 하위 호환: 옛 코드가 feature_columns를 읽던 경로 대비(list 계약).
    @property
    def feature_columns(self) -> list[str]:
        return list(self.FEATURE_COLUMNS)
