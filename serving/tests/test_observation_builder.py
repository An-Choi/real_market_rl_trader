import numpy as np
import pandas as pd
import pytest

from data.feature_engineer import FeatureEngineer
from errors import InsufficientHistoryError, StaleDataError
from friction.friction_model import FrictionModel
from observation_builder import build_decision_inputs

TZ = "Asia/Seoul"


def make_minute_data(days: int, seed: int = 7, start: str = "2026-06-01") -> pd.DataFrame:
    """결손 없는 합성 1분봉: 평일만, 09:00–15:19 정규봉 + 15:30 동시호가 print.

    TradingValue는 당일 누적(파이프라인이 diff한다). 가격은 seeded random walk —
    결정론적이라 replay parity 테스트에 그대로 쓴다.
    """
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, periods=days, tz=TZ)
    frames = []
    price = 300_000.0
    for day in bdays:
        minutes = pd.date_range(day + pd.Timedelta(hours=9),
                                day + pd.Timedelta(hours=15, minutes=19),
                                freq="1min", tz=TZ)
        minutes = minutes.append(pd.DatetimeIndex(
            [day + pd.Timedelta(hours=15, minutes=30)], tz=TZ))
        n = len(minutes)
        steps = rng.normal(0, 120, size=n)
        closes = np.maximum(price + np.cumsum(steps), 1000.0).round(-2)
        opens = np.concatenate([[price], closes[:-1]])
        volume = rng.integers(1_000, 50_000, size=n)
        minute_value = (closes * volume).astype("int64")
        frames.append(pd.DataFrame({
            "Timestamp": minutes,
            "Open": opens, "High": np.maximum(opens, closes) + 100,
            "Low": np.minimum(opens, closes) - 100, "Close": closes,
            "Volume": volume,
            "TradingValue": np.cumsum(minute_value),
        }))
        price = float(closes[-1])
    return pd.concat(frames, ignore_index=True)
ENV_PARAMS = {"unit_fraction": 0.199, "max_units": 5, "initial_cash": 10_000.0,
              "episode_days": 20, "duration_horizon_bars": 1280,
              "nominal_bars_per_day": 64}
FRICTION = FrictionModel(fee_rate=0.00018, spread_rate=0.001, slippage_rate=0.0,
                         execution_uncertainty_rate=0.0, sell_tax_rate=0.002,
                         dynamic_spread=True, date_based_sell_tax=True)


def _build(minute_data, as_of, **overrides):
    kwargs = dict(
        bars_1m=minute_data, as_of=as_of,
        units_held=0, shares_held=0.0, bars_since_entry=0, available_cash=10_000.0,
        env_params=ENV_PARAMS, friction_model=FRICTION,
        max_bar_age=pd.Timedelta(minutes=10),
        feature_engineer=FeatureEngineer(),
    )
    kwargs.update(overrides)
    return build_decision_inputs(**kwargs)


def _last_day(minute_data):
    return pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]


def test_selects_last_completed_5min_bar(minute_data):
    # 의도적으로 전체(미래 포함) 데이터를 그대로 넘긴다 — builder의 방어적
    # cutoff가 없으면 마지막 날 15:20 bar가 선택되고 음수 age로 stale도 통과한다.
    day = _last_day(minute_data)
    as_of = pd.Timestamp(f"{day} 11:00:00", tz=TZ)
    result = _build(minute_data, as_of)
    assert result.bar_ts == pd.Timestamp(f"{day} 11:00:00", tz=TZ)
    assert result.observation.shape == (16,)
    assert result.observation.dtype == np.float32


def test_stale_data_raises(minute_data):
    day = _last_day(minute_data)
    as_of = pd.Timestamp(f"{day} 23:00:00", tz=TZ)  # 마지막 bar에서 수 시간 경과
    with pytest.raises(StaleDataError):
        _build(minute_data, as_of)


def test_insufficient_history_raises(minute_data):
    one_day = make_minute_data(days=1)  # warmup 미달 → transform 출력 empty
    day = pd.to_datetime(one_day["Timestamp"]).dt.date.iloc[-1]
    with pytest.raises(InsufficientHistoryError):
        _build(one_day, pd.Timestamp(f"{day} 11:00:00", tz=TZ))


def test_as_of_before_data_start_raises_insufficient(minute_data):
    # cutoff 결과가 완전히 빔 — transform의 ValueError가 아니라 503 계약 에러여야 한다
    with pytest.raises(InsufficientHistoryError):
        _build(minute_data, pd.Timestamp("2020-01-01 10:00:00", tz=TZ))


def test_all_defect_days_raise_insufficient(minute_data):
    # 09:00 행 제거 → 모든 날 late-start 결손 판정 → 파이프라인 내부
    # ValueError("No objects to concatenate")가 InsufficientHistoryError로 매핑
    ts = pd.to_datetime(minute_data["Timestamp"])
    defective = minute_data[ts.dt.time != pd.Timestamp("09:00").time()].reset_index(drop=True)
    day = pd.to_datetime(defective["Timestamp"]).dt.date.iloc[-1]
    with pytest.raises(InsufficientHistoryError):
        _build(defective, pd.Timestamp(f"{day} 11:00:00", tz=TZ))


def test_causality_future_rows_do_not_change_past(minute_data):
    day = _last_day(minute_data)
    as_of = pd.Timestamp(f"{day} 11:00:00", tz=TZ)
    base = _build(minute_data, as_of)
    longer = make_minute_data(days=27)  # 같은 seed → 같은 prefix + 미래 하루 추가
    with_future = _build(longer, as_of)
    np.testing.assert_array_equal(base.observation, with_future.observation)
    assert base.bar_ts == with_future.bar_ts


def test_corrupt_numeric_column_does_not_raise_insufficient_history(minute_data):
    # 데이터 손상(비수치 Close)이 warmup 부족과 동일한 InsufficientHistoryError로
    # 매핑되면 실제 데이터 버그가 조용히 fail-closed 503으로 위장된다 — 다른
    # 예외(전파된 TypeError/ValueError 등)여야 한다.
    corrupt = minute_data.copy()
    corrupt["Close"] = "corrupt"
    day = _last_day(corrupt)
    as_of = pd.Timestamp(f"{day} 11:00:00", tz=TZ)
    with pytest.raises(Exception) as excinfo:
        _build(corrupt, as_of)
    assert not isinstance(excinfo.value, InsufficientHistoryError)


def test_portfolio_fields_flow_into_observation(minute_data):
    day = _last_day(minute_data)
    as_of = pd.Timestamp(f"{day} 11:00:00", tz=TZ)
    flat = _build(minute_data, as_of)
    held = _build(minute_data, as_of, units_held=2,
                  shares_held=0.02, bars_since_entry=37)
    assert held.observation[11] == np.float32(2 / 5)
    assert held.observation[13] == np.float32(37 / 1280)
    np.testing.assert_array_equal(flat.observation[:11], held.observation[:11])
    assert bool(held.action_mask[2]) is True and bool(flat.action_mask[2]) is False
