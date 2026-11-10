import numpy as np
import pandas as pd
import pytest

from env.research_env import ResearchTradingEnvironment, TargetExposureEnvironment, market_window
from friction.friction_model import FrictionModel


def frame(n=30, execution=100.0):
    return pd.DataFrame({"Timestamp": pd.date_range("2026-01-05 10:00", periods=n, freq="5min", tz="Asia/Seoul"),
                         "Close": 100.0, "ExecPrice": execution, "signal": np.arange(n, dtype=float)})


def make(cls=TargetExposureEnvironment, data=None, **kwargs):
    return cls(market_data=frame() if data is None else data,
               feature_columns=["signal"], initial_cash=10_000,
               friction_model=FrictionModel(fee_rate=0.001, spread_rate=0,
                   slippage_rate=0, sell_tax_rate=0), **kwargs)


def test_target_accounting_and_liquidation():
    env = make()
    env.reset(seed=0)
    for action in (3, 1, 5, 0):
        before = env.cash + env.shares_held * 100
        _, _, _, _, info = env.step(action)
        assert env.cash >= 0
        assert env.shares_held == int(env.shares_held)
        assert info["portfolio_value"] == pytest.approx(before - info["friction_cost"])
        assert env.cost_basis == pytest.approx(env.shares_held * 100)
    assert env.shares_held == env.cost_basis == 0


def test_planned_buy_quantity_does_not_increase_with_future_discount():
    env = make(data=frame(execution=50))
    env.reset(seed=0)
    env.step(3)  # .6 * 10,000 / observed 100 => 60 planned shares
    assert env.shares_held == 60  # not 120 using the future fill price


def test_gap_up_cash_limited_fill_and_no_negative_cash():
    env = make(data=frame(execution=120))
    env.reset(seed=0)
    env.step(5)
    assert env.shares_held == 83
    assert env.cash >= 0


def test_repeated_target_within_tolerance_does_not_churn():
    env = make()
    env.reset(seed=0)
    env.step(3)
    _, _, _, _, info = env.step(3)
    assert info["trade_value"] == info["friction_cost"] == 0


def test_incremental_lifo_cost_and_observed_size():
    data = frame()
    data.loc[0, "ExecPrice"] = 90
    env = make(ResearchTradingEnvironment, data=data)
    env.reset(seed=0)
    env.step(1)
    assert env.shares_held == 20
    assert env.cost_basis == 1800
    env.step(1)
    assert env.cost_basis == 3800
    env.step(2)
    assert env.shares_held == 20
    assert env.cost_basis == 1800
    env.step(3)
    assert env.units_held == env.shares_held == env.cost_basis == 0


def test_windows_are_causal_reset_on_day_and_match_shared_builder():
    data = frame()
    data.loc[10:, "Timestamp"] += pd.Timedelta(days=1)
    env = make(data=data, window_length=12)
    env.reset(seed=0, options={"start_date": str(data.Timestamp.iloc[0].date()), "episode_days": 2})
    for i in range(15):
        obs = env._get_observation()
        expected = market_window(data, ["signal"], i, 12)
        np.testing.assert_array_equal(obs[:24], expected)
        if i == 10:
            assert sum(obs[12:24]) == 1
        env.step(0)
    changed = data.copy()
    changed.loc[15:, "signal"] = 100_000
    np.testing.assert_array_equal(market_window(data, ["signal"], 14, 12),
                                  market_window(changed, ["signal"], 14, 12))


def test_gaps_reset_context():
    data = frame()
    data.loc[3:, "Timestamp"] += pd.Timedelta(minutes=5)
    assert market_window(data, ["signal"], 3, 12)[12:].sum() == 1


@pytest.mark.parametrize("cls", [ResearchTradingEnvironment, TargetExposureEnvironment])
def test_missing_future_execution_prevents_fill_without_changing_observation(cls):
    data = frame()
    data.loc[0, "ExecPrice"] = np.nan
    env = make(cls, data=data)
    env.reset(seed=0)
    _, _, _, _, info = env.step(1 if cls is ResearchTradingEnvironment else 5)
    assert env.cash == 10_000
    assert env.shares_held == env.units_held == env.cost_basis == 0
    assert info["friction_cost"] == 0
    assert info["fill_unavailable"]


@pytest.mark.parametrize("cls", [ResearchTradingEnvironment, TargetExposureEnvironment])
def test_gym_contract(cls):
    from gymnasium.utils.env_checker import check_env
    check_env(make(cls), skip_render_check=True)


def test_terminal_cost_does_not_use_next_execution_price():
    env = make()
    env.reset(seed=0)
    env.step(3)
    original = env.estimate_liquidation_cost()
    env.market_data.loc[1, "ExecPrice"] = 10_000
    assert env.estimate_liquidation_cost() == original


def test_random_actions_preserve_cash_and_equity_identity():
    rng = np.random.default_rng(12)
    data = frame(n=200)
    data["Close"] = 100 * np.exp(np.cumsum(rng.normal(0, .01, 200)))
    data["ExecPrice"] = data.Close * np.exp(rng.normal(0, .01, 200))
    for cls in (ResearchTradingEnvironment, TargetExposureEnvironment):
        env = make(cls, data=data)
        env.reset(seed=0, options={"start_date": str(data.Timestamp.iloc[0].date()),
                                   "episode_days": 10})
        for _ in range(199):
            valid = np.flatnonzero(env.action_masks())
            _, reward, _, _, info = env.step(int(rng.choice(valid)))
            assert np.isfinite(reward)
            assert env.cash >= -1e-8
            assert env.cost_basis >= 0
            assert env.portfolio_value == pytest.approx(env.cash + env.shares_held * info["valuation_price"])
