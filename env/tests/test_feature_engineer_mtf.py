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
    expected = ["Timestamp", "Close", "ExecPrice"] + list(FeatureEngineer.FEATURE_COLUMNS)
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


def _minute_days_with_auction(days: list[str], auction_offset: float = 2.0) -> pd.DataFrame:
    """정규봉 09:00~15:19(380분, Open=Close−0.3) + 15:30 종가 동시호가 1봉."""
    frames = []
    price = 100.0
    rng = np.random.default_rng(0)
    for d in days:
        ts = pd.date_range(f"{d} 09:00", periods=380, freq="1min", tz="Asia/Seoul")
        closes = price + np.cumsum(rng.normal(0, 0.2, 380))
        price = float(closes[-1])
        cum_tv = np.cumsum(rng.integers(1_000_000, 9_000_000, 380))
        frames.append(pd.DataFrame({
            "Timestamp": ts, "Open": closes - 0.3, "High": closes + 0.5,
            "Low": closes - 0.5, "Close": closes,
            "Volume": rng.integers(500, 5000, 380), "TradingValue": cum_tv,
        }))
        auction_price = price + auction_offset
        frames.append(pd.DataFrame({
            "Timestamp": [pd.Timestamp(f"{d} 15:30", tz="Asia/Seoul")],
            "Open": [auction_price], "High": [auction_price], "Low": [auction_price],
            "Close": [auction_price], "Volume": [9000],
            "TradingValue": [int(cum_tv[-1]) + 900_000],
        }))
    return pd.concat(frames, ignore_index=True)


def test_schema_version_bumped_for_exec_price() -> None:
    assert FeatureEngineer.FEATURE_SCHEMA_VERSION == 3


def test_exec_price_is_next_minute_open() -> None:
    mdf = _minute_days_with_auction(DAYS)
    out = FeatureEngineer().transform(mdf)
    # 비말단 결정 시점: 라벨 T의 ExecPrice = 같은 날 T시각 1분봉의 Open (관찰 봉 Close와 다름)
    row = out[pd.to_datetime(out["Timestamp"]).dt.strftime("%H:%M") == "11:05"].iloc[0]
    label_ts = pd.Timestamp(row["Timestamp"])
    minute_open = mdf.loc[pd.to_datetime(mdf["Timestamp"]) == label_ts, "Open"].iloc[0]
    assert row["ExecPrice"] == minute_open
    assert row["ExecPrice"] != row["Close"]  # same-bar Close 체결 금지


def test_day_final_exec_price_is_auction_single_price() -> None:
    mdf = _minute_days_with_auction(DAYS)
    out = FeatureEngineer().transform(mdf)
    dates = pd.to_datetime(out["Timestamp"]).dt.date
    auction_ts = pd.to_datetime(mdf["Timestamp"])
    for day, group in out.groupby(dates):
        final = group.iloc[-1]
        expected = mdf.loc[
            (auction_ts.dt.date == day) & (auction_ts.dt.strftime("%H:%M") == "15:30"),
            "Close",
        ].iloc[0]
        assert final["ExecPrice"] == expected            # 15:30 단일가 체결
        assert final["Close"] != expected                # 관찰(Close)에는 경매가 미포함


def test_day_without_auction_final_exec_price_is_nan() -> None:
    # 경매 print도 다음 1분봉도 없는 날 말단은 NaN (env가 Close로 fallback).
    out = FeatureEngineer().transform(_minute_days(DAYS))
    dates = pd.to_datetime(out["Timestamp"]).dt.date
    for _, group in out.groupby(dates):
        assert pd.isna(group.iloc[-1]["ExecPrice"])          # 각 날 말단
        assert group.iloc[:-1]["ExecPrice"].notna().all()    # 비말단은 항상 존재


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
