"""cross-day feature 테스트: gap_open, relative_volume_tod, Adv20.

의도적 day-reset 완화 — 완료된 과거 거래일만 참조(causal). 당일 데이터가
분모·기준값에 섞이면 leakage이므로 각 테스트가 당일-제외를 직접 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.cross_day_features import add_cross_day_features

_SLOTS = ["09:05", "09:10", "09:15", "09:20"]


def _day(date_str: str, opens: list[float], closes: list[float],
         vols: list[int], tvs: list[float]) -> pd.DataFrame:
    ts = [pd.Timestamp(f"{date_str} {s}", tz="Asia/Seoul") for s in _SLOTS]
    return pd.DataFrame({
        "Timestamp": ts, "Open": opens, "High": closes, "Low": closes,
        "Close": closes, "Volume": vols, "MinuteTradingValue": tvs,
    })


def _flat_day(date_str: str, price: float = 100.0, vol: int = 100,
              tv: float = 1_000.0) -> pd.DataFrame:
    n = len(_SLOTS)
    return _day(date_str, [price] * n, [price] * n, [vol] * n, [tv] * n)


def _empty_auction() -> pd.DataFrame:
    return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close"])


def _dates(n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-06-02", periods=n, freq="D")]


def _many_days(n: int, vol_by_day=None, tv_by_day=None) -> pd.DataFrame:
    frames = []
    for i, d in enumerate(_dates(n)):
        vol = vol_by_day(i) if vol_by_day else 100
        tv = tv_by_day(i) if tv_by_day else 1_000.0
        frames.append(_flat_day(d, vol=vol, tv=tv))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- gap_open

def test_gap_open_uses_prev_day_auction_close() -> None:
    d1 = _day("2025-06-02", [100.0] * 4, [100.0, 100.0, 100.0, 102.0],
              [100] * 4, [1_000.0] * 4)
    d2 = _day("2025-06-03", [110.0, 111.0, 112.0, 113.0], [110.0] * 4,
              [100] * 4, [1_000.0] * 4)
    auction = pd.DataFrame({
        "Timestamp": [pd.Timestamp("2025-06-02 15:30", tz="Asia/Seoul")],
        "Open": [105.0], "High": [105.0], "Low": [105.0], "Close": [105.0],
    })
    out = add_cross_day_features(pd.concat([d1, d2], ignore_index=True), auction)
    day2 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-03").date()]
    # 전일 종가는 동시호가 체결가(105) — 정규봉 마지막 Close(102)가 아님
    expected = np.log(110.0 / 105.0)
    assert np.allclose(day2["gap_open"], expected)


def test_gap_open_falls_back_to_last_regular_close() -> None:
    d1 = _day("2025-06-02", [100.0] * 4, [100.0, 100.0, 100.0, 102.0],
              [100] * 4, [1_000.0] * 4)
    d2 = _day("2025-06-03", [110.0] * 4, [110.0] * 4, [100] * 4, [1_000.0] * 4)
    out = add_cross_day_features(pd.concat([d1, d2], ignore_index=True), _empty_auction())
    day2 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-03").date()]
    assert np.allclose(day2["gap_open"], np.log(110.0 / 102.0))


def test_gap_open_first_day_is_nan_and_constant_within_day() -> None:
    d1 = _flat_day("2025-06-02")
    d2 = _day("2025-06-03", [110.0, 111.0, 112.0, 113.0], [110.0] * 4,
              [100] * 4, [1_000.0] * 4)
    out = add_cross_day_features(pd.concat([d1, d2], ignore_index=True), _empty_auction())
    day1 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-02").date()]
    day2 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-03").date()]
    assert day1["gap_open"].isna().all()  # 전일 없음 → NaN (warm-up drop 대상)
    assert day2["gap_open"].nunique() == 1  # 당일 시가 기준 하루 상수


# ------------------------------------------------- relative_volume_tod

def test_relative_volume_tod_uses_prior_days_same_slot_median() -> None:
    # 20일 warm-up: 모든 슬롯 vol 100 → 21일째 vol 250 → 250/100
    df = _many_days(21, vol_by_day=lambda i: 250 if i == 20 else 100)
    out = add_cross_day_features(df, _empty_auction())
    day21 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-22").date()]
    assert np.allclose(day21["relative_volume_tod"], 2.5, rtol=1e-6)


def test_relative_volume_tod_excludes_today_from_median() -> None:
    # 당일 거래량이 아무리 커도 분모(과거 20일 중앙값)는 불변이어야 함
    df = _many_days(21, vol_by_day=lambda i: 100_000 if i == 20 else 100)
    out = add_cross_day_features(df, _empty_auction())
    day21 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-22").date()]
    assert np.allclose(day21["relative_volume_tod"], 1_000.0, rtol=1e-6)


def test_relative_volume_tod_warmup_is_nan() -> None:
    df = _many_days(20)  # 20일째는 과거가 19일뿐 → NaN
    out = add_cross_day_features(df, _empty_auction())
    day20 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-21").date()]
    assert day20["relative_volume_tod"].isna().all()


# ------------------------------------------------------------------ Adv20

def test_adv20_is_mean_of_prior_20_daily_trading_values() -> None:
    # 일별 거래대금 = (i+1)*1000*슬롯4개. 21일째 Adv20 = 1~20일 평균, 당일 제외.
    df = _many_days(21, tv_by_day=lambda i: (i + 1) * 1_000.0)
    out = add_cross_day_features(df, _empty_auction())
    day21 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-22").date()]
    expected = np.mean([(i + 1) * 1_000.0 * 4 for i in range(20)])
    assert np.allclose(day21["Adv20"], expected)
    assert day21["Adv20"].nunique() == 1  # 하루 상수


def test_adv20_warmup_is_nan() -> None:
    df = _many_days(20)
    out = add_cross_day_features(df, _empty_auction())
    day20 = out[pd.to_datetime(out["Timestamp"]).dt.date == pd.Timestamp("2025-06-21").date()]
    assert day20["Adv20"].isna().all()
