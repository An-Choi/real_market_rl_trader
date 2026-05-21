from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.virtual_agents import (
    MeanReversionTrader,
    NoiseTrader,
    TrendFollower,
    VirtualAgent,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    periods = 30
    close = np.array([100.0 + i * 0.5 for i in range(periods)])
    ma_20 = pd.Series(close).rolling(20).mean().to_numpy()
    return pd.DataFrame({"Close": close, "ma_20": ma_20})


# --- Protocol conformance ---


def test_all_agents_conform_to_protocol() -> None:
    agents: list[VirtualAgent] = [
        NoiseTrader(),
        TrendFollower(),
        MeanReversionTrader(),
    ]
    for a in agents:
        assert hasattr(a, "compute_impact")
        assert hasattr(a, "reset")


# --- NoiseTrader ---


def test_noise_trader_impact_bounded(sample_data):
    agent = NoiseTrader(noise_strength=0.001, seed=42)
    impacts = [agent.compute_impact(sample_data.iloc[: i + 1], i) for i in range(10, 30)]
    assert all(abs(v) <= 0.001 + 1e-9 for v in impacts)


def test_noise_trader_returns_float(sample_data):
    agent = NoiseTrader(noise_strength=0.001, seed=0)
    impact = agent.compute_impact(sample_data.iloc[:11], 10)
    assert isinstance(impact, float)


def test_noise_trader_reset_makes_deterministic(sample_data):
    agent = NoiseTrader(noise_strength=0.001, seed=7)
    history = sample_data.iloc[:11]
    seq_a = [agent.compute_impact(history, 10) for _ in range(20)]
    agent.reset(seed=7)
    seq_b = [agent.compute_impact(history, 10) for _ in range(20)]
    assert seq_a == seq_b


# --- TrendFollower ---


def test_trend_follower_positive_trend(sample_data):
    agent = TrendFollower(trend_strength=0.002, window=5)
    impact = agent.compute_impact(sample_data.iloc[:11], 10)
    assert impact == pytest.approx(0.002)


def test_trend_follower_negative_trend(sample_data):
    df = sample_data.copy()
    df["Close"] = [100.0 - i * 0.5 for i in range(len(df))]
    agent = TrendFollower(trend_strength=0.002, window=5)
    impact = agent.compute_impact(df.iloc[:11], 10)
    assert impact == pytest.approx(-0.002)


def test_trend_follower_returns_zero_when_insufficient_data(sample_data):
    agent = TrendFollower(trend_strength=0.002, window=5)
    impact = agent.compute_impact(sample_data.iloc[:3], 2)
    assert impact == 0.0


def test_trend_follower_uses_exact_n_period_return(sample_data):
    # window=5에서는 iloc[step-5]와 iloc[step] 두 점만 비교해야 함.
    df = sample_data.copy()
    df["Close"] = [100.0] * 10 + [100.0] * 5 + [110.0] + [100.0] * (len(df) - 16)
    # step=15에서 past=df[10]=100, curr=df[15]=110 → positive
    agent = TrendFollower(trend_strength=0.002, window=5)
    impact = agent.compute_impact(df.iloc[:16], 15)
    assert impact == pytest.approx(0.002)
    # step=16에서 past=df[11]=100, curr=df[16]=100 → zero
    impact = agent.compute_impact(df.iloc[:17], 16)
    assert impact == 0.0


# --- MeanReversionTrader ---


def test_mean_reversion_above_ma(sample_data):
    df = sample_data.copy()
    df.loc[25, "Close"] = 200.0
    df.loc[25, "ma_20"] = 100.0
    agent = MeanReversionTrader(mean_reversion_strength=0.001)
    impact = agent.compute_impact(df.iloc[:26], 25)
    assert impact == pytest.approx(-0.001)


def test_mean_reversion_below_ma(sample_data):
    df = sample_data.copy()
    df.loc[25, "Close"] = 50.0
    df.loc[25, "ma_20"] = 100.0
    agent = MeanReversionTrader(mean_reversion_strength=0.001)
    impact = agent.compute_impact(df.iloc[:26], 25)
    assert impact == pytest.approx(0.001)


def test_mean_reversion_missing_ma_col(sample_data):
    df = sample_data.drop(columns=["ma_20"])
    agent = MeanReversionTrader()
    impact = agent.compute_impact(df.iloc[:11], 10)
    assert impact == 0.0


def test_mean_reversion_nan_ma(sample_data):
    df = sample_data.copy()
    df.loc[5, "ma_20"] = np.nan
    agent = MeanReversionTrader()
    impact = agent.compute_impact(df.iloc[:6], 5)
    assert impact == 0.0


def test_mean_reversion_custom_price_col(sample_data):
    df = sample_data.copy()
    df["Last"] = df["Close"]
    df.loc[25, "Last"] = 200.0
    df.loc[25, "ma_20"] = 100.0
    agent = MeanReversionTrader(mean_reversion_strength=0.001, price_col="Last")
    impact = agent.compute_impact(df.iloc[:26], 25)
    assert impact == pytest.approx(-0.001)
