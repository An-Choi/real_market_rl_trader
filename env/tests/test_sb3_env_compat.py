from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.hybrid_market_env import HybridMarketEnv


def test_hybrid_market_env_passes_stable_baselines3_check_env() -> None:
    env_checker = pytest.importorskip(
        "stable_baselines3.common.env_checker",
        reason="stable-baselines3 is required for PPO environment compatibility checks",
    )

    periods = 40
    close = 100.0 + np.linspace(0.0, 2.0, periods)
    market_data = pd.DataFrame(
        {
            "Close": close,
            "ma_20": pd.Series(close).rolling(20, min_periods=1).mean(),
        }
    )
    env = HybridMarketEnv(
        market_data=market_data,
        feature_columns=["ma_20"],
        noise_strength=0.0,
        trend_strength=0.0,
        mean_reversion_strength=0.0,
    )

    env_checker.check_env(env, warn=True)
