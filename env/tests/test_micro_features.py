from __future__ import annotations

import numpy as np
import pandas as pd

from data.micro_features import add_micro_features


def _bars(date_str: str, closes: list[float], vols: list[int] | None = None) -> pd.DataFrame:
    n = len(closes)
    ts = pd.date_range(f"{date_str} 09:05", periods=n, freq="5min", tz="Asia/Seoul")
    vols = vols or [1000] * n
    return pd.DataFrame({
        "Timestamp": ts, "Open": closes, "High": closes, "Low": closes,
        "Close": closes, "Volume": vols,
        "MinuteTradingValue": [c * v for c, v in zip(closes, vols)],
    })


def test_log_ret_1_value() -> None:
    df = _bars("2025-06-02", [100.0, 110.0])
    out = add_micro_features(df)
    assert abs(out["log_ret_1"].iloc[1] - np.log(110 / 100)) < 1e-9


def test_log_ret_k_causal_first_rows_nan() -> None:
    df = _bars("2025-06-02", [float(100 + i) for i in range(5)])
    out = add_micro_features(df)
    assert np.isnan(out["log_ret_3"].iloc[2])   # shift(3) → 앞 3행 NaN
    assert not np.isnan(out["log_ret_3"].iloc[3])


def test_relative_volume_zero_denominator_safe() -> None:
    # 거래량 0 → rolling mean 0. EPS로 Inf 차단(warm-up NaN은 정상 drop 대상).
    df = _bars("2025-06-02", [float(100 + i) for i in range(15)], vols=[0] * 15)
    out = add_micro_features(df)
    rv = out["relative_volume"]
    assert not np.isinf(rv).any()  # 0/0 blow-up 없음
    # warm-up(rolling 12 + shift 1)을 지난 행은 유한.
    assert np.isfinite(rv.iloc[13:]).all()


def test_features_are_day_reset() -> None:
    # 두 거래일. Day2 첫 행 log_ret_1은 Day1 마지막을 참조하면 안 됨(NaN).
    d1 = _bars("2025-06-02", [float(100 + i) for i in range(3)])
    d2 = _bars("2025-06-03", [float(200 + i) for i in range(3)])
    out = add_micro_features(pd.concat([d1, d2], ignore_index=True))
    day2 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-03").date()]
    assert np.isnan(day2["log_ret_1"].iloc[0])  # overnight return 안 섞임


def test_vwap_dev_uses_intraday_cumulative() -> None:
    df = _bars("2025-06-02", [100.0, 100.0, 100.0])
    out = add_micro_features(df)
    # 가격 일정 → vwap == Close → vwap_dev ≈ 0
    assert abs(out["vwap_dev"].iloc[-1]) < 1e-9
