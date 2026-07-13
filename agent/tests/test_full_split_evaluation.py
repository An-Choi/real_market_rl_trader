from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.trading_env import TradingEnvironment
from evaluation.metrics import summarize_backtest
from policies.evaluation import BacktestEngine


class HoldAgent:
    def predict(self, observation):
        return 0, None


def _three_day_data() -> pd.DataFrame:
    frames = []
    for d in pd.bdate_range("2025-06-02", periods=n_days):
        ts = pd.date_range(f"{d.date()} 09:00", periods=bars, freq="5min", tz="Asia/Seoul")
        frames.append(pd.DataFrame({
            "Timestamp": ts,
            "Close": np.full(bars, price),
            "feature": np.zeros(bars),
        }))
    return pd.concat(frames, ignore_index=True)


def test_backtest_engine_evaluates_every_split_date() -> None:
    environment = TradingEnvironment(
        market_data=_three_day_data(),
        feature_columns=["feature"],
    )
    assert engine.terminal_liquidation_cost == pytest.approx(expected_cost)
    metrics = summarize_backtest(
        results,
        initial_value=engine.reset_info["portfolio_value"],
        terminal_liquidation_cost=engine.terminal_liquidation_cost,
    )
    settled = results["portfolio_value"].iloc[-1] - expected_cost
    assert metrics["final_portfolio_value"] == pytest.approx(settled)
    assert metrics["open_at_end"] == 1.0


def test_clear_before_end_has_zero_settlement() -> None:
    data = _data(n_days=1, bars=4)  # 3 steps: Add, Hold, Clear(마지막 실행 bar)
    engine = BacktestEngine(ClearAtStepAgent(clear_step=2), _env(data))
    results = engine.run(seed=0)
    assert results["units_held"].iloc[-1] == 0
    assert engine.terminal_liquidation_cost == 0.0


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
