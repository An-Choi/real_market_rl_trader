import numpy as np
import pandas as pd
import pytest

from env.controlled_exposure_env import ControlledExposureEnvironment
from friction.friction_model import FrictionModel
from friction.research_friction import ScaledFrictionModel


def make(n=15, **kwargs):
    data = pd.DataFrame({"Timestamp": pd.date_range("2026-01-05 10:00", periods=n, freq="5min", tz="Asia/Seoul"),
                         "Close": np.linspace(100, 110, n), "ExecPrice": np.linspace(100, 110, n),
                         "signal": np.arange(n, dtype=float)})
    return ControlledExposureEnvironment(market_data=data, feature_columns=["signal"],
        initial_cash=10000, friction_model=FrictionModel(fee_rate=.001, spread_rate=0,
            slippage_rate=0, sell_tax_rate=0), **kwargs)


def test_macro_transition_rebalances_once_and_discounts_rewards():
    env = make(decision_interval=6, bar_gamma=.9)
    env.reset(seed=0)
    _, reward, done, truncated, info = env.step(3)
    records = info["bar_records"]
    assert len(records) == info["elapsed_bars"] == 6
    assert sum(r["executed_trades"] for r in records) == 1
    assert reward == pytest.approx(sum(.9**i * r["bar_reward"] for i, r in enumerate(records)))
    assert not done and not truncated
    assert env.shares_held == 60


def test_terminal_settlement_is_in_training_reward_and_absorbing():
    env = make(n=4, decision_interval=6, bar_gamma=1., reward_return_mode="log_return")
    env.reset(seed=0)
    observation, reward, done, truncated, info = env.step(3)
    assert done and not truncated
    assert info["elapsed_bars"] == 3
    assert env.shares_held == env.cost_basis == 0
    assert env.portfolio_value == env.cash
    assert reward == pytest.approx(np.log(env.cash / env.initial_cash))
    assert env.estimate_liquidation_cost() == 0
    assert info["bar_records"][-1]["terminal_settlement_cost"] > 0
    assert observation[-4] == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_risk_cost_is_observed_drawdown_excess_and_penalty_is_separate():
    env = make(n=8, drawdown_threshold=.01, risk_multiplier=.3, reward_scale=100)
    env.market_data.loc[:, "Close"] = np.linspace(100, 70, 8)
    env.market_data.loc[:, "ExecPrice"] = env.market_data.Close
    env.reset(seed=0)
    _, reward, _, _, info = env.step(5)
    record = info["bar_records"][0]
    assert record["risk_cost"] == pytest.approx(max(0, record["drawdown"] - .01))
    assert record["risk_penalty"] == pytest.approx(100 * .3 * record["risk_cost"])
    assert reward == pytest.approx(100 * record["base_return"] - record["risk_penalty"])


def test_band_avoids_small_adjustments_and_zero_always_clears():
    env = make(rebalance_tolerance=.05)
    env.reset(seed=0)
    env.step(3)
    _, _, _, _, info = env.step(3)
    assert info["bar_records"][0]["executed_trades"] == 0
    env.step(0)
    assert env.shares_held == 0


def test_gap_ends_macro_transition_and_does_not_repeat_order():
    env = make(decision_interval=6)
    env.market_data.loc[3:, "Timestamp"] += pd.Timedelta(minutes=5)
    env.reset(seed=0)
    _, _, _, _, info = env.step(3)
    assert info["elapsed_bars"] == 3
    assert env.current_step == 3
    assert sum(r["executed_trades"] for r in info["bar_records"]) == 1


def test_multiplier_state_and_gym_contract():
    from gymnasium.utils.env_checker import check_env
    env = make(decision_interval=6)
    check_env(env, skip_render_check=True)
    env.reset(seed=0)
    env.set_risk_multiplier(2)
    assert env._get_observation()[-3] == pytest.approx(.2)


def test_exit_cost_reserve_recognizes_roundtrip_without_double_charging():
    env = make(n=4, bar_gamma=1., reward_return_mode="log_return")
    env.market_data["Close"] = env.market_data["ExecPrice"] = 100.
    env.reset(seed=0)
    _, first_reward, _, _, first = env.step(3)
    # 60 shares: buy fee 6 and closing fee reserve 6, not just the buy fee.
    assert first["liquidation_value"] == pytest.approx(9988)
    assert first_reward == pytest.approx(np.log(9988 / 10000))
    _, hold_reward, _, _, _ = env.step(3)
    _, closing_reward, done, _, _ = env.step(3)
    assert done and env.cash == pytest.approx(9988)
    assert hold_reward == closing_reward == pytest.approx(0)


@pytest.mark.parametrize("multiplier", [0., .5, 1., 2.])
def test_cost_multiplier_scales_dynamic_spread_and_date_tax(multiplier):
    from datetime import date
    kwargs = dict(trade_value=100000, side="sell", price=10000,
                  trade_date=date(2026, 1, 5), liquidity_score=.5)
    baseline = FrictionModel(dynamic_spread=True, date_based_sell_tax=True)
    stress = ScaledFrictionModel(dynamic_spread=True, date_based_sell_tax=True, multiplier=multiplier)
    assert stress.calculate_total_friction(**kwargs) == pytest.approx(multiplier * baseline.calculate_total_friction(**kwargs))
