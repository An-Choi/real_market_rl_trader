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


def _two_day_gap_data() -> pd.DataFrame:
    """day1 종가 100 고정, day2 시가부터 110 — overnight gap fixture."""
    frames = []
    for day, price in (("2025-06-02", 100.0), ("2025-06-03", 110.0)):
        ts = pd.date_range(f"{day} 09:00", periods=3, freq="5min", tz="Asia/Seoul")
        frames.append(pd.DataFrame({
            "Timestamp": ts,
            "Close": np.full(3, price),
            "ma_5": np.full(3, price),
        }))
    return pd.concat(frames, ignore_index=True)


def test_b_bars_yield_b_minus_1_steps_and_truncation() -> None:
    env = make_env(_days(2, bars_per_day=3), episode_days=2)  # 6 bars
    env.reset(seed=0)
    flags = []
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        flags.append((terminated, truncated))
    assert flags[:4] == [(False, False)] * 4
    assert flags[4] == (False, True)  # 마지막 step: truncated, NOT terminated


def test_position_persists_across_day_boundary() -> None:
    env = make_env(_two_day_gap_data(), episode_days=2)
    env.reset(seed=0)
    env.step(1)  # bar0에서 Add
    # bar2(=day1 마지막 bar)에서 실행되는 step: day 경계를 넘어 bar3에서 평가
    env.step(0)
    _, _, _, _, info = env.step(0)
    assert info["units_held"] == 1  # 강제청산 없이 유지
    ts_exec = pd.Timestamp(info["valuation_timestamp"])
    assert ts_exec.date().isoformat() == "2025-06-03"


def test_overnight_gap_reflected_in_portfolio_value() -> None:
    env = make_env(_two_day_gap_data(), episode_days=2)
    env.reset(seed=0)
    _, _, _, _, info0 = env.step(1)  # buy 2000 notional @100 → 20 shares
    value_before_gap = info0["portfolio_value"]
    env.step(0)  # bar1 실행 → bar2 평가 (같은 날, 가격 동일)
    _, _, _, _, info_gap = env.step(0)  # bar2 실행 → bar3(익일, 110) 평가
    assert info_gap["portfolio_value"] - value_before_gap == pytest.approx(20.0 * 10.0)


def test_last_executable_bar_action_is_respected() -> None:
    env = make_env(_days(1, bars_per_day=3), episode_days=1)  # 3 bars → 2 steps
    env.reset(seed=0)
    env.step(0)
    _, _, terminated, truncated, info = env.step(1)  # 마지막 실행 bar에서 Add
    assert truncated is True and terminated is False
    assert info["units_held"] == 1  # 강제청산으로 무효화되지 않는다
    assert "forced_clear" not in info


def test_valuation_timestamp_always_after_execution() -> None:
    env = make_env(_days(2, bars_per_day=3), episode_days=2)
    env.reset(seed=0)
    done = False
    while not done:
        exec_ts = env.get_current_market_row()["Timestamp"]
        _, _, terminated, truncated, info = env.step(0)
        assert pd.Timestamp(info["valuation_timestamp"]) > pd.Timestamp(exec_ts)
        done = terminated or truncated


def test_estimate_liquidation_cost() -> None:
    env = make_env(_days(1, bars_per_day=3), episode_days=1)
    env.reset(seed=0)
    assert env.estimate_liquidation_cost() == 0.0  # flat
    _, _, _, _, info = env.step(1)  # Add
    expected = env.friction_model.calculate_total_friction(
        trade_value=info["held_market_value"], side="sell", liquidity_score=None,
    )
    assert env.estimate_liquidation_cost() == pytest.approx(expected)
