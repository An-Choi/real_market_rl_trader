from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from env.hybrid_market_env import HybridMarketEnv
from experiments.backtest_engine import BacktestEngine


@pytest.fixture
def market_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    periods = 60
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=periods))
    ma_20 = pd.Series(close).rolling(20).mean().to_numpy()
    volume = rng.integers(100_000, 500_000, size=periods).astype(float)
    return pd.DataFrame({"Close": close, "ma_20": ma_20, "Volume": volume})


@pytest.fixture
def env(market_data: pd.DataFrame) -> HybridMarketEnv:
    return HybridMarketEnv(
        market_data=market_data,
        feature_columns=["ma_20"],
        noise_strength=0.001,
        trend_strength=0.002,
        mean_reversion_strength=0.001,
        max_agent_impact=0.05,
    )


# --- Basic shape & flow ---


def test_reset_returns_correct_observation_shape(env: HybridMarketEnv) -> None:
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)  # 1 feature + 3 portfolio state
    assert "portfolio_value" in info


def test_observation_space_unchanged_from_parent(env: HybridMarketEnv) -> None:
    # 부모와 동일한 shape 유지
    assert env.observation_space.shape == (4,)


def test_episode_runs_to_completion(env: HybridMarketEnv) -> None:
    env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
    assert steps > 0


# --- info dict ---


def test_info_contains_hybrid_fields(env: HybridMarketEnv) -> None:
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)
    for key in ("real_price", "simulated_price", "agent_impact", "execution_price"):
        assert key in info


def test_info_separates_execution_and_valuation_steps(env: HybridMarketEnv) -> None:
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)

    assert info["execution_step"] == 0
    assert info["valuation_step"] == 1
    assert info["execution_simulated_price"] == pytest.approx(
        env.trade_history[-1].price
    )
    assert info["simulated_price"] == pytest.approx(info["valuation_simulated_price"])


def test_simulated_price_is_positive(env: HybridMarketEnv) -> None:
    env.reset(seed=0)
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(0)
        assert info["simulated_price"] > 0
        if terminated or truncated:
            break


def test_simulated_price_differs_from_real_at_least_once(
    env: HybridMarketEnv, market_data: pd.DataFrame
) -> None:
    env.reset(seed=0)
    diffs = []
    for _ in range(len(market_data) - 1):
        _, _, terminated, truncated, info = env.step(0)
        diffs.append(abs(info["simulated_price"] - info["real_price"]))
        if terminated or truncated:
            break
    assert any(d > 0 for d in diffs)


def test_friction_cost_is_nonnegative_on_trade(env: HybridMarketEnv) -> None:
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)
    assert info["friction_cost"] >= 0


def test_execution_price_recorded_after_trade(env: HybridMarketEnv) -> None:
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)
    assert not math.isnan(info["execution_price"])
    assert info["execution_price"] > 0


# --- Determinism (Phase 1 핵심 contract) ---


def test_same_seed_produces_same_trajectory(market_data: pd.DataFrame) -> None:
    def run() -> list[float]:
        env_ = HybridMarketEnv(
            market_data=market_data,
            feature_columns=["ma_20"],
        )
        env_.reset(seed=123)
        values = []
        for action in [1, 0, 2, 1, 0, 0, 1, 2, 0, 0]:
            _, _, terminated, truncated, info = env_.step(action)
            values.append(info["portfolio_value"])
            if terminated or truncated:
                break
        return values

    assert run() == run()


# --- Portfolio invariant (가장 중요) ---


def test_portfolio_value_equals_cash_plus_position_times_simulated_price(
    env: HybridMarketEnv,
) -> None:
    env.reset(seed=0)
    for action in [1, 0, 2, 0, 1, 0]:
        _, _, terminated, truncated, info = env.step(action)
        expected = env.cash + env.position * info["simulated_price"]
        assert info["portfolio_value"] == pytest.approx(expected, rel=1e-9, abs=1e-6)
        if terminated or truncated:
            break


# --- Clipping ---


def test_agent_impact_within_clip_bound(market_data: pd.DataFrame) -> None:
    # 비정상적으로 큰 strength로 만들어 clipping이 발동되는지 확인
    env_ = HybridMarketEnv(
        market_data=market_data,
        feature_columns=["ma_20"],
        noise_strength=0.5,
        trend_strength=0.5,
        mean_reversion_strength=0.5,
        max_agent_impact=0.05,
    )
    env_.reset(seed=0)
    for _ in range(20):
        _, _, terminated, truncated, info = env_.step(0)
        assert abs(info["agent_impact"]) <= 0.05 + 1e-9
        if terminated or truncated:
            break


# --- Reset state ---


def test_reset_clears_to_nan(env: HybridMarketEnv) -> None:
    env.reset(seed=0)
    env.step(1)
    env.reset(seed=0)
    assert math.isnan(env.simulated_price)
    assert math.isnan(env.real_price)
    assert math.isnan(env.agent_impact)
    assert math.isnan(env.execution_price)


def test_reset_clears_impact_cache(env: HybridMarketEnv) -> None:
    env.reset(seed=0)
    env.step(0)
    assert len(env._impact_cache) > 0
    env.reset(seed=0)
    assert len(env._impact_cache) == 0


# --- Look-ahead guard ---


def test_agent_receives_only_past_history(market_data: pd.DataFrame) -> None:
    """spy agent로 history 길이가 step+1과 일치하는지 검증."""
    seen: list[tuple[int, int]] = []

    class SpyAgent:
        def compute_impact(self, history: pd.DataFrame, step: int) -> float:
            seen.append((step, len(history)))
            return 0.0

        def reset(self, seed: int | None = None) -> None:
            pass

    env_ = HybridMarketEnv(market_data=market_data, feature_columns=["ma_20"])
    env_.virtual_agents = [SpyAgent()]
    env_.reset(seed=0)
    for _ in range(5):
        _, _, terminated, truncated, _ = env_.step(0)
        if terminated or truncated:
            break

    for step, history_len in seen:
        assert history_len == step + 1, f"step {step} got history len {history_len}"


# --- Backtest integration ---


def test_backtest_seed_replays_same_hybrid_trajectory(
    market_data: pd.DataFrame,
) -> None:
    class BuyThenHoldAgent:
        def predict(self, observation, market_row=None):
            return 1 if market_row.name == 0 else 0

    env_ = HybridMarketEnv(
        market_data=market_data,
        feature_columns=["ma_20"],
        noise_strength=0.001,
        trend_strength=0.0,
        mean_reversion_strength=0.0,
    )
    engine = BacktestEngine(agent=BuyThenHoldAgent(), environment=env_)

    first = engine.run(max_steps=10, seed=123)
    second = engine.run(max_steps=10, seed=123)

    assert first["portfolio_value"].tolist() == second["portfolio_value"].tolist()
    assert first["agent_impact"].tolist() == second["agent_impact"].tolist()


def test_backtest_results_keep_hybrid_diagnostics(
    market_data: pd.DataFrame,
) -> None:
    class HoldAgent:
        def predict(self, observation, market_row=None):
            return 0

    env_ = HybridMarketEnv(market_data=market_data, feature_columns=["ma_20"])
    engine = BacktestEngine(agent=HoldAgent(), environment=env_)

    results = engine.run(max_steps=1, seed=123)

    expected_columns = {
        "real_price",
        "simulated_price",
        "agent_impact",
        "execution_price",
        "execution_step",
        "valuation_step",
    }
    assert expected_columns.issubset(results.columns)
