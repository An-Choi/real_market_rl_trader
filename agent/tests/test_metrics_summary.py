from __future__ import annotations

import pandas as pd
import pytest

from evaluation.metrics import summarize_backtest


def test_summarize_backtest_includes_execution_metrics() -> None:
    results = pd.DataFrame({
        "portfolio_value": [100.0, 110.0, 105.0],
        "reward": [0.0, 0.10, -0.045],
        "action": [1, 0, 2],
        "trade_value": [20.0, 0.0, -22.0],
        "forced_clear": [False, False, True],
    })

    metrics = summarize_backtest(results)

    assert metrics["total_return"] == pytest.approx(0.05)
    assert metrics["final_portfolio_value"] == pytest.approx(105.0)
    assert metrics["trade_count"] == 2.0
    assert metrics["number_of_trades"] == 2.0
    assert metrics["turnover"] == pytest.approx(42.0 / 105.0)
    assert metrics["forced_clear_count"] == 1.0
    assert metrics["hold_action_rate"] == pytest.approx(1 / 3)
    assert metrics["add_action_rate"] == pytest.approx(1 / 3)
    assert metrics["clear_action_rate"] == pytest.approx(1 / 3)


def test_summarize_backtest_empty_has_all_keys() -> None:
    metrics = summarize_backtest(pd.DataFrame())

    assert metrics == {
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "final_portfolio_value": 0.0,
        "win_rate": 0.0,
        "trade_count": 0.0,
        "number_of_trades": 0.0,
        "turnover": 0.0,
        "forced_clear_count": 0.0,
        "evaluated_days": 0.0,
        "hold_action_rate": 0.0,
        "add_action_rate": 0.0,
        "clear_action_rate": 0.0,
    }
