from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.trading_env import TradingEnvironment


def _days(n_days: int, bars_per_day: int = 3, start: str = "2025-06-02") -> pd.DataFrame:
    """평일 n_days개 × bars_per_day봉, 가격 100 고정."""
    dates = pd.bdate_range(start, periods=n_days)
    frames = []
    for d in dates:
        ts = pd.date_range(f"{d.date()} 09:00", periods=bars_per_day, freq="5min", tz="Asia/Seoul")
        frames.append(pd.DataFrame({
            "Timestamp": ts,
            "Close": np.full(bars_per_day, 100.0),
            "ma_5": np.full(bars_per_day, 100.0),
        }))
    return pd.concat(frames, ignore_index=True)


def make_env(data: pd.DataFrame, **kwargs) -> TradingEnvironment:
    return TradingEnvironment(
        market_data=data,
        feature_columns=["ma_5"],
        initial_cash=10_000.0,
        unit_fraction=0.20,
        max_units=5,
        **kwargs,
    )


def test_episode_days_must_be_positive_int() -> None:
    data = _days(2)
    with pytest.raises(ValueError):
        make_env(data, episode_days=0)
    with pytest.raises(ValueError):
        make_env(data, episode_days=-1)


def test_reset_selects_consecutive_days() -> None:
    env = make_env(_days(5), episode_days=3)
    _, info = env.reset(seed=0)
    assert info["episode_days"] == 3
    assert len(env._episode_dates) == 3
    # 연속 거래일: available_dates에서 인접한 3개
    idx = env._available_dates.index(env._episode_dates[0])
    assert list(env._episode_dates) == env._available_dates[idx:idx + 3]


def test_random_start_only_from_full_windows() -> None:
    env = make_env(_days(5), episode_days=3)
    valid_starts = set(env._available_dates[:3])  # 5일 중 3일 윈도우 시작 가능일 = 앞 3일
    for seed in range(20):
        env.reset(seed=seed)
        assert env._episode_dates[0] in valid_starts


def test_same_seed_same_start_date() -> None:
    env = make_env(_days(5), episode_days=3)
    env.reset(seed=42)
    first = env._episode_dates[0]
    env.reset(seed=42)
    assert env._episode_dates[0] == first


def test_start_date_option_and_truncate() -> None:
    env = make_env(_days(5), episode_days=3)
    dates = env._available_dates
    # 뒤에서 2번째 날 시작 → 3일 요청해도 2일로 truncate
    _, info = env.reset(seed=0, options={"start_date": str(dates[3])})
    assert info["episode_days"] == 2
    assert list(env._episode_dates) == dates[3:]


def test_date_alias_and_start_date_precedence() -> None:
    env = make_env(_days(5), episode_days=1)
    dates = env._available_dates
    env.reset(seed=0, options={"date": str(dates[1])})
    assert env._episode_dates[0] == dates[1]
    env.reset(seed=0, options={"date": str(dates[1]), "start_date": str(dates[2])})
    assert env._episode_dates[0] == dates[2]


def test_reset_episode_days_override_validation() -> None:
    env = make_env(_days(5), episode_days=3)
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"episode_days": 0})
    _, info = env.reset(seed=0, options={"episode_days": 2})
    assert info["episode_days"] == 2


def test_window_needs_at_least_two_bars() -> None:
    env = make_env(_days(1, bars_per_day=1))
    with pytest.raises(ValueError):
        env.reset(seed=0)


def test_reset_info_has_initial_market_price() -> None:
    env = make_env(_days(2), episode_days=2)
    _, info = env.reset(seed=0)
    assert info["initial_market_price"] == 100.0
