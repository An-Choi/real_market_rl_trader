from __future__ import annotations

import numpy as np
import pandas as pd

from env.trading_env import TradingEnvironment
from policies.evaluation import BacktestEngine


class HoldAgent:
    def predict(self, observation):
        return 0, None


def _three_day_data() -> pd.DataFrame:
    frames = []
    for day in ("2025-06-02", "2025-06-03", "2025-06-04"):
        timestamps = pd.date_range(
            f"{day} 09:00", periods=4, freq="5min", tz="Asia/Seoul"
        )
        frames.append(pd.DataFrame({
            "Timestamp": timestamps,
            "Close": np.array([100.0, 101.0, 102.0, 103.0]),
            "feature": np.zeros(4),
        }))
    return pd.concat(frames, ignore_index=True)


def test_backtest_engine_evaluates_every_split_date() -> None:
    environment = TradingEnvironment(
        market_data=_three_day_data(),
        feature_columns=["feature"],
    )

    results = BacktestEngine(HoldAgent(), environment).run(max_steps=2, seed=7)

    assert results["episode"].nunique() == 3
    assert results.groupby("episode").size().tolist() == [2, 2, 2]
    assert results["episode_date"].tolist()[::2] == [
        "2025-06-02",
        "2025-06-03",
        "2025-06-04",
    ]


def test_backtest_engine_uses_non_overlapping_multi_day_windows() -> None:
    environment = TradingEnvironment(
        market_data=_three_day_data(),
        feature_columns=["feature"],
        episode_days=2,
    )

    results = BacktestEngine(HoldAgent(), environment).run(seed=7)

    assert results["episode"].nunique() == 2
    assert results.groupby("episode").size().tolist() == [8, 4]
    assert results.groupby("episode")["episode_date"].first().tolist() == [
        "2025-06-02",
        "2025-06-04",
    ]
    assert pd.to_datetime(results["timestamp"]).dt.date.nunique() == 3
