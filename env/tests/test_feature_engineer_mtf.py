from __future__ import annotations

import numpy as np
import pandas as pd

from data.feature_engineer import FeatureEngineer


def _minute_days(days: list[str], per_day: int = 390) -> pd.DataFrame:
    frames = []
    price = 100.0
    rng = np.random.default_rng(0)
    for d in days:
        ts = pd.date_range(f"{d} 09:00", periods=per_day, freq="1min", tz="Asia/Seoul")
        steps = rng.normal(0, 0.2, per_day)
        closes = price + np.cumsum(steps)
        price = float(closes[-1])
        frames.append(pd.DataFrame({
            "Timestamp": ts, "Open": closes, "High": closes + 0.5, "Low": closes - 0.5,
            "Close": closes, "Volume": rng.integers(500, 5000, per_day),
            "TradingValue": np.cumsum(rng.integers(1_000_000, 9_000_000, per_day)),
        }))
    return pd.concat(frames, ignore_index=True)


DAYS = [f"2025-06-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11)]


def test_feature_columns_frozen_order() -> None:
    assert FeatureEngineer.FEATURE_COLUMNS == (
        "log_ret_1", "log_ret_3", "log_ret_12",
        "realized_vol_12", "relative_volume", "vwap_dev",
        "trend_strength_30m", "macd_hist_30m", "vol_regime_30m",
    )


def test_output_columns_and_no_nan() -> None:
    out = FeatureEngineer().transform(_minute_days(DAYS))
    expected = ["Timestamp", "Close"] + list(FeatureEngineer.FEATURE_COLUMNS)
    assert list(out.columns) == expected
    assert not out[list(FeatureEngineer.FEATURE_COLUMNS)].isna().any().any()


def test_transform_deterministic() -> None:
    mdf = _minute_days(DAYS)
    a = FeatureEngineer().transform(mdf)
    b = FeatureEngineer().transform(mdf)
    pd.testing.assert_frame_equal(a, b)


def test_transform_does_not_mutate_self_or_input() -> None:
    mdf = _minute_days(DAYS)
    cols_before = set(mdf.columns)
    fe = FeatureEngineer()
    _ = fe.transform(mdf)
    assert set(mdf.columns) == cols_before  # 입력 미변경
    assert not hasattr(fe, "feature_columns") or isinstance(fe.FEATURE_COLUMNS, tuple)


def test_future_bar_mutation_does_not_change_past_features() -> None:
    mdf = _minute_days(DAYS)
    out_a = FeatureEngineer().transform(mdf)
    altered = mdf.copy()
    # 마지막 거래일의 뒤쪽 분봉 Close를 변조
    last_day = pd.to_datetime(altered["Timestamp"]).dt.date == pd.Timestamp("2025-06-11").date()
    idx = altered[last_day].index[-30:]
    altered.loc[idx, "Close"] = altered.loc[idx, "Close"] + 50.0
    out_b = FeatureEngineer().transform(altered)
    # 변조 이전 거래일(06-10까지) feature는 불변
    mask_a = pd.to_datetime(out_a["Timestamp"]).dt.date < pd.Timestamp("2025-06-11").date()
    mask_b = pd.to_datetime(out_b["Timestamp"]).dt.date < pd.Timestamp("2025-06-11").date()
    pd.testing.assert_frame_equal(
        out_a[mask_a].reset_index(drop=True), out_b[mask_b].reset_index(drop=True)
    )
