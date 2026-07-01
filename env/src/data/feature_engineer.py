"""MTF feature 파이프라인 오케스트레이션 + registry freeze.

파이프라인 순서(불변): 결손일 drop → 거래일별 MinuteTradingValue diff →
종가 동시호가 fold-in → 5분 resample → 5분 micro feature → 30분 context feature → warm-up dropna.
transform()은 순수 함수(self·입력 미변경).
"""

from __future__ import annotations

import pandas as pd

from data.context_features import add_context_features
from data.defect_days import drop_defect_days
from data.micro_features import add_micro_features
from data.resample import add_minute_trading_value, fold_closing_auction, resample_5min


class FeatureEngineer:
    # v2: 종가 동시호가(15:30) print를 terminal bar에 fold-in.
    FEATURE_SCHEMA_VERSION: int = 2
    FEATURE_COLUMNS: tuple[str, ...] = (
        "log_ret_1", "log_ret_3", "log_ret_12",
        "realized_vol_12", "relative_volume", "vwap_dev",
        "trend_strength_30m", "macd_hist_30m", "vol_regime_30m",
    )

    def transform(self, minute_df: pd.DataFrame, ts_col: str = "Timestamp") -> pd.DataFrame:
        """1분봉(멀티데이) → 5분 그리드 [Timestamp, Close, <FEATURE_COLUMNS>].

        모든 feature lookback 만족 행만(NaN drop). 순수 함수: self 미변경, 입력 미변경.
        """
        clean = drop_defect_days(minute_df, ts_col)
        clean = add_minute_trading_value(clean, ts_col)
        clean = fold_closing_auction(clean, ts_col)
        bars5 = resample_5min(clean, ts_col)
        bars5 = add_micro_features(bars5, ts_col)
        bars5 = add_context_features(bars5, clean, ts_col)

        cols = [ts_col, "Close", *self.FEATURE_COLUMNS]
        result = bars5[cols].dropna(subset=list(self.FEATURE_COLUMNS))
        return result.reset_index(drop=True)

    # 하위 호환: 옛 코드가 feature_columns를 읽던 경로 대비(list 계약).
    @property
    def feature_columns(self) -> list[str]:
        return list(self.FEATURE_COLUMNS)
