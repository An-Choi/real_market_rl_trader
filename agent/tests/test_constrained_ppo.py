from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.constrained_ppo import (DurationAwareRolloutBuffer, ResearchPPO,
                                    RiskBudgetCallback, projected_dual_update)
from experiments.run_ablation import (synthetic_data, make_env, evaluate,
    save_research_bundle, load_research_bundle, evaluation_config, data_fingerprints)
from models.normalization import FeatureNormalizer
from data.feature_engineer import FeatureEngineer
from utils.config_loader import load_config


def test_duration_gae_matches_hand_computed_returns_and_terminal_boundary():
    buffer = DurationAwareRolloutBuffer(3, gym.spaces.Box(-1, 1, (1,)),
        gym.spaces.Discrete(2), n_envs=1, gamma=.9, gae_lambda=1.)
    buffer.rewards[:, 0] = [1, 2, 3]
    buffer.values[:] = 0
    buffer.episode_starts[:] = 0
    buffer.durations[:, 0] = [2, 1, 3]
    buffer.compute_returns_and_advantage(torch.tensor([10.]), np.array([True]))
    np.testing.assert_allclose(buffer.returns[:, 0], [1 + .9**2 * (2 + .9*3), 2 + .9*3, 3])
    buffer.compute_returns_and_advantage(torch.tensor([10.]), np.array([False]))
    assert buffer.returns[-1, 0] == pytest.approx(3 + .9**3 * 10)
    buffer.episode_starts[2, 0] = 1
    buffer.compute_returns_and_advantage(torch.tensor([10.]), np.array([False]))
    assert buffer.returns[1, 0] == pytest.approx(2)


def test_dual_ascent_direction_budget_and_projection():
    assert projected_dual_update(.1, .02, .01, 10) == pytest.approx(.2)
    assert projected_dual_update(.1, 0, .01, 10) == 0
    assert projected_dual_update(9, 1, 0, 10) == 10
    assert projected_dual_update(.1, .01, .01, 10) == .1
    with pytest.raises(ValueError):
        projected_dual_update(.1, float("nan"), .01, 10)


def test_duration_gae_applies_lambda_per_underlying_bar():
    buffer = DurationAwareRolloutBuffer(2, gym.spaces.Box(-1, 1, (1,)),
        gym.spaces.Discrete(2), n_envs=1, gamma=.9, gae_lambda=.8)
    buffer.rewards[:, 0] = [1, 2]
    buffer.values[:] = 0
    buffer.episode_starts[:] = 0
    buffer.durations[:, 0] = [3, 1]
    buffer.compute_returns_and_advantage(torch.tensor([0.]), np.array([True]))
    assert buffer.advantages[0, 0] == pytest.approx(1 + (.9*.8)**3 * 2)


def test_data_freeze_includes_liquidity_and_future_execution_inputs():
    data = synthetic_data(days=2)[0]
    original = data_fingerprints(data)
    data["SYNTHETIC"].loc[1, "Adv20"] *= 2
    assert data_fingerprints(data) != original
    original = data_fingerprints(data)
    data["SYNTHETIC"].loc[2, "ExecPrice"] *= 2
    assert data_fingerprints(data) != original


def test_daily_evaluation_compounds_cash_and_matches_training_settlement():
    data = synthetic_data(days=3)[0]["SYNTHETIC"]
    data["Close"] = data["ExecPrice"] = 100.
    config = load_config(ROOT / "env/configs/config.yaml")
    config["environment"].update(initial_cash=10000., episode_days=1)
    config["friction"].update(fee_rate=.001, spread_rate=0., slippage_rate=0.,
        execution_uncertainty_rate=0., sell_tax_rate=0., dynamic_spread=False, date_based_sell_tax=False)
    stats = FeatureNormalizer.fit(data, list(FeatureEngineer.FEATURE_COLUMNS))
    env = make_env(data, config, "R", stats)
    metrics, trace = evaluate(None, env, baseline="intraday_buy_and_hold")
    # Each day buys 99 shares at 100, entry fee 9.9 and exit fee 9.9.
    assert env.initial_cash == 10000.
    assert env.cash == pytest.approx(10000 - 3*19.8)
    assert metrics["net_return"] == pytest.approx(-3*19.8/10000)
    assert metrics["bars"] == len(data) - 3
    assert metrics["evaluation_episodes"] == 3
    assert metrics["trade_count"] == 6
    assert metrics["friction_cost"] == pytest.approx(3*19.8)
    assert (trace.groupby("valuation_date").shares.last() == 0).all()
    cash, _ = evaluate(None, env, baseline="cash")
    assert cash["net_return"] == 0
    assert cash["trade_count"] == 0


@pytest.mark.parametrize("condition", ["Q", "R", "E", "F", "G", "H", "I", "J", "K"])
def test_controlled_observations_and_masks_do_not_use_future_execution(condition):
    data = synthetic_data(days=2)[0]["SYNTHETIC"]
    config = load_config(ROOT / "env/configs/config.yaml")
    config["environment"]["episode_days"] = 1
    stats = FeatureNormalizer.fit(data, list(FeatureEngineer.FEATURE_COLUMNS))
    altered = data.copy()
    altered["ExecPrice"] *= 10
    first, second = make_env(data, config, condition, stats), make_env(altered, config, condition, stats)
    before, _ = first.reset(seed=1)
    after, _ = second.reset(seed=1)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(first.action_masks(), second.action_masks())


@pytest.mark.parametrize("condition", ["Q", "R", "E", "F", "G", "H", "I", "J", "K"])
def test_controlled_learning_bar_budget_and_exact_bundle_roundtrip(tmp_path, condition):
    torch.set_num_threads(1)
    data = synthetic_data(days=3)[0]["SYNTHETIC"]
    config = load_config(ROOT / "env/configs/config.yaml")
    stats = FeatureNormalizer.fit(data, list(FeatureEngineer.FEATURE_COLUMNS))
    env = make_env(data, config, condition, stats)
    from experiments.run_ablation import EXTENSIONS
    mode = EXTENSIONS[condition]["risk_mode"]
    callback = RiskBudgetCallback(mode=mode,initial_multiplier=.1 if mode!="off" else 0.)
    model = ResearchPPO("MlpPolicy", env, rollout_buffer_class=DurationAwareRolloutBuffer,
        n_steps=8, batch_size=8, n_epochs=1, seed=1, verbose=0,
        gamma=config["agent"]["ppo"]["gamma"],
        policy_kwargs={"net_arch": [16, 16]})
    model.research_bar_budget = 100
    model.learn(100, callback=callback)
    assert 100 <= model.research_bar_transitions < 100 + 8 * env.decision_interval
    assert sum(r["bars"] for r in callback.history) == model.research_bar_transitions
    assert all(r["multiplier_before"] >= 0 for r in callback.history)
    frozen = evaluation_config(config, model)
    evaluation_env = make_env(data, frozen, condition, stats)
    expected, trace = evaluate(model, evaluation_env)
    assert expected["bars"] == len(data) - 1 == len(trace)
    assert expected["decisions"] <= expected["bars"]
    save_research_bundle(model, stats, evaluation_env, frozen, tmp_path, {"condition": condition})
    restored, restored_env, _ = load_research_bundle(tmp_path, data)
    actual, _ = evaluate(restored, restored_env)
    assert actual == expected


def test_training_rejects_mismatched_bar_discount_before_any_update():
    data = synthetic_data(days=2)[0]["SYNTHETIC"]
    config = load_config(ROOT / "env/configs/config.yaml")
    stats = FeatureNormalizer.fit(data, list(FeatureEngineer.FEATURE_COLUMNS))
    model = ResearchPPO("MlpPolicy", make_env(data, config, "F", stats),
        rollout_buffer_class=DurationAwareRolloutBuffer, gamma=.9, n_steps=8, batch_size=8, verbose=0)
    with pytest.raises(ValueError, match="same per-bar gamma"):
        model.learn(8, callback=RiskBudgetCallback())


def test_training_dual_multiplier_increases_on_observed_budget_violation():
    from experiments.run_learning_diagnostics import easy_market
    data = easy_market("falling", 42, days=3)
    config = load_config(ROOT / "env/configs/config.yaml")
    config["environment"]["episode_days"] = 1
    config["research"] = {"drawdown_threshold": 0., "risk_budget": 0.}
    stats = FeatureNormalizer.fit(data, list(FeatureEngineer.FEATURE_COLUMNS))
    callback = RiskBudgetCallback(mode="dual", initial_multiplier=.1, budget=0.)
    model = ResearchPPO("MlpPolicy", make_env(data, config, "H", stats),
        rollout_buffer_class=DurationAwareRolloutBuffer, gamma=config["agent"]["ppo"]["gamma"],
        n_steps=16, batch_size=16, n_epochs=1, seed=42, verbose=0, policy_kwargs={"net_arch": [16,16]})
    model.learn(64, callback=callback)
    assert callback.history[0]["mean_risk_cost"] > 0
    assert model.research_risk_multiplier > .1
    assert all(row["multiplier_after"] >= row["multiplier_before"] for row in callback.history)
