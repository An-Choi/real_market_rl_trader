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


class AddOnceAgent:
    def __init__(self) -> None:
        self.bought = False

    def predict(self, observation):
        if not self.bought:
            self.bought = True
            return 1, None
        return 0, None


class ClearAtStepAgent:
    """지정 step에서 Clear, 그 전에 한 번 Add."""

    def __init__(self, clear_step: int) -> None:
        self.clear_step = clear_step
        self.step = 0

    def predict(self, observation):
        action = 0
        if self.step == 0:
            action = 1
        elif self.step == self.clear_step:
            action = 2
        self.step += 1
        return action, None


def _data(n_days: int = 3, bars: int = 4, price: float = 100.0) -> pd.DataFrame:
    frames = []
    for d in pd.bdate_range("2025-06-02", periods=n_days):
        ts = pd.date_range(f"{d.date()} 09:00", periods=bars, freq="5min", tz="Asia/Seoul")
        frames.append(pd.DataFrame({
            "Timestamp": ts,
            "Close": np.full(bars, price),
            "feature": np.zeros(bars),
        }))
    return pd.concat(frames, ignore_index=True)


def _env(data: pd.DataFrame) -> TradingEnvironment:
    return TradingEnvironment(market_data=data, feature_columns=["feature"])


def test_engine_runs_whole_split_as_single_episode() -> None:
    engine = BacktestEngine(HoldAgent(), _env(_data(n_days=3, bars=4)))
    results = engine.run(seed=7)
    assert len(results) == 3 * 4 - 1  # B bars → B−1 steps, 윈도우 분할 없음
    assert engine.terminal_liquidation_cost == 0.0
    # 날짜 경계 step 2개: execution date != valuation date
    exec_d = pd.to_datetime(results["timestamp"]).dt.date
    val_d = pd.to_datetime(results["valuation_timestamp"]).dt.date
    assert int((exec_d != val_d).sum()) == 2


def test_settled_equity_matches_manual_friction_formula() -> None:
    env = _env(_data())
    engine = BacktestEngine(AddOnceAgent(), env)
    results = engine.run(seed=0)
    held = results["held_market_value"].iloc[-1]
    assert held > 0
    expected_cost = env.friction_model.calculate_total_friction(
        trade_value=held, side="sell", liquidity_score=None,
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


def test_equal_price_fixture_clear_equals_hold_plus_settlement() -> None:
    # 모든 bar 동일가 → 마지막 실행가(B−2)와 정산가(B−1)가 같아 등가가 성립
    data = _data(n_days=1, bars=4, price=100.0)
    clear_engine = BacktestEngine(ClearAtStepAgent(clear_step=2), _env(data))
    clear_results = clear_engine.run(seed=0)
    clear_final = clear_results["portfolio_value"].iloc[-1]

    hold_env = _env(data)
    hold_engine = BacktestEngine(AddOnceAgent(), hold_env)
    hold_results = hold_engine.run(seed=0)
    hold_settled = (
        hold_results["portfolio_value"].iloc[-1] - hold_engine.terminal_liquidation_cost
    )
    assert hold_settled == pytest.approx(clear_final)
