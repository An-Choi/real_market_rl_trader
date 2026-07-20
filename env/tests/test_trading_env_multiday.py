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


def test_feature_schema_version_stored_and_validated() -> None:
    data = _days(2)
    env = make_env(data, feature_schema_version=3)
    assert env.feature_schema_version == 3

    env_default = make_env(data)
    assert env_default.feature_schema_version is None

    with pytest.raises(ValueError):
        make_env(data, feature_schema_version=0)
    with pytest.raises(ValueError):
        make_env(data, feature_schema_version=-1)
    with pytest.raises(ValueError):
        make_env(data, feature_schema_version=True)


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


def test_random_reset_clamps_when_data_shorter_than_episode_days() -> None:
    env = make_env(_days(2), episode_days=5)
    _, info = env.reset(seed=0)
    assert info["episode_days"] == 2
    assert list(env._episode_dates) == env._available_dates


def test_window_needs_at_least_two_bars() -> None:
    env = make_env(_days(1, bars_per_day=1))
    with pytest.raises(ValueError):
        env.reset(seed=0)


def test_reset_info_has_initial_market_price() -> None:
    env = make_env(_days(2), episode_days=2)
    _, info = env.reset(seed=0)
    assert info["initial_market_price"] == 100.0


def test_action_masks_remove_duplicate_no_op_actions() -> None:
    env = make_env(_days(2, bars_per_day=8), episode_days=2)
    env.reset(seed=0)
    assert env.action_masks().tolist() == [True, True, False]

    for _ in range(env.max_units):
        env.step(1)
    assert env.action_masks().tolist() == [True, False, True]

    env.step(2)
    assert env.action_masks().tolist() == [True, True, False]


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


def test_duration_horizon_bars_validation_and_default() -> None:
    data = _days(2)
    with pytest.raises(ValueError):
        make_env(data, episode_days=2, duration_horizon_bars=0)
    env = make_env(data, episode_days=2)
    assert env.duration_horizon_bars == 2 * 64  # 기본값 = episode_days * 64


def test_holding_duration_uses_fixed_horizon_and_clips() -> None:
    # fixture feature가 1개라 obs는 5차원 — portfolio state는 음수 인덱스로 참조
    # (obs[-2] = holding_duration_norm, obs[-1] = tod_frac)
    env = make_env(_days(2, bars_per_day=3), episode_days=2, duration_horizon_bars=2)
    obs, _ = env.reset(seed=0)
    assert obs[-2] == 0.0  # flat
    obs, *_ = env.step(1)   # entry_step=0 → 경과 1 bar
    assert obs[-2] == pytest.approx(1 / 2)
    obs, *_ = env.step(0)   # 경과 2 bars
    assert obs[-2] == pytest.approx(1.0)
    obs, *_ = env.step(0)   # 경과 3 bars → clip at 1.0 (날짜 경계를 넘어도 연속)
    assert obs[-2] == pytest.approx(1.0)


def test_holding_duration_independent_of_runtime_episode_length() -> None:
    # 같은 horizon이면 episode 길이가 달라도 같은 경과 bar에서 같은 값
    env_short = make_env(_days(2, bars_per_day=3), episode_days=1, duration_horizon_bars=10)
    env_long = make_env(_days(2, bars_per_day=3), episode_days=2, duration_horizon_bars=10)
    obs_s = env_short.reset(seed=0, options={"start_date": "2025-06-02"})[0]
    obs_l = env_long.reset(seed=0, options={"start_date": "2025-06-02"})[0]
    obs_s, *_ = env_short.step(1)
    obs_l, *_ = env_long.step(1)
    assert obs_s[-2] == obs_l[-2] == pytest.approx(1 / 10)


def test_tod_frac_resets_each_day() -> None:
    env = make_env(_days(2, bars_per_day=3), episode_days=2, nominal_bars_per_day=3)
    env.reset(seed=0)
    tods = []
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(0)
        tods.append(float(obs[-1]))
        done = terminated or truncated
    # valuation bar 기준: day1 bar1→0.5, bar2→1.0, day2 bar0→0.0, bar1→0.5, bar2→1.0
    assert tods == pytest.approx([0.5, 1.0, 0.0, 0.5, 1.0])


def test_tod_frac_independent_of_actual_day_length() -> None:
    """tod_frac은 nominal_bars_per_day(고정 상수)만 분모로 써야 한다 — 그날 실제
    bar 수(미래 정보, 결손 시 달라짐)에 의존하면 causal 계약 위반이다.

    같은 day1 앞 2개 bar가 동일한 두 데이터셋(하나는 day1이 3-bar, 하나는
    day1의 마지막 bar가 결손돼 2-bar)에서 첫 step의 tod_frac(obs[-1])이
    같아야 한다 — 그날 뒤쪽 bar 존재 여부가 앞선 step의 관측을 바꾸면 안 된다.
    """
    full_data = _days(2, bars_per_day=3)
    # day1의 마지막 bar(index 2)를 제거해 day1을 2-bar로 결손시킨다.
    truncated_data = full_data.drop(index=2).reset_index(drop=True)

    env_full = make_env(full_data, episode_days=2, nominal_bars_per_day=3)
    env_truncated = make_env(truncated_data, episode_days=2, nominal_bars_per_day=3)

    obs_full, _ = env_full.reset(seed=0, options={"start_date": "2025-06-02"})
    obs_truncated, _ = env_truncated.reset(seed=0, options={"start_date": "2025-06-02"})
    assert obs_full[-1] == obs_truncated[-1]

    obs_full, *_ = env_full.step(0)
    obs_truncated, *_ = env_truncated.step(0)
    assert float(obs_full[-1]) == pytest.approx(float(obs_truncated[-1]))


def test_tod_frac_clips_at_one_for_days_longer_than_nominal() -> None:
    """nominal_bars_per_day보다 실제로 긴 날에는 tod_frac이 1.0에서 clip된다."""
    env = make_env(_days(1, bars_per_day=5), episode_days=1, nominal_bars_per_day=3)
    env.reset(seed=0)
    tods = []
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(0)
        tods.append(float(obs[-1]))
        done = terminated or truncated
    # nominal=3 → 분모 max(3-1,1)=2. step 1..4 valuation bar는 1,2,3,4 → frac 0.5,1.0,1.5→clip,2.0→clip
    assert tods == pytest.approx([0.5, 1.0, 1.0, 1.0])
