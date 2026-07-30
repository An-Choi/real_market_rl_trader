from __future__ import annotations

import pytest

from experiments.robustness import summarize_robustness_runs


def _run(seed: int, result: float, buy_hold: float, static_80: float) -> dict:
    return {
        "fold": 1,
        "seed": seed,
        "ent_coef": 0.001,
        "test_metrics": {
            "total_return": result,
            "max_drawdown": -0.10,
            "turnover": 2.0,
        },
        "excess_vs_buy_and_hold": result - buy_hold,
        "excess_vs_static_80pct": result - static_80,
    }


def test_robustness_summary_aggregates_independent_training_runs() -> None:
    summary = summarize_robustness_runs([
        _run(1, 0.10, 0.02, 0.03),
        _run(2, -0.02, 0.02, 0.03),
        _run(3, 0.04, 0.02, 0.03),
    ])["0.001"]

    assert summary["runs"] == 3
    assert summary["median_test_return"] == pytest.approx(0.04)
    assert summary["worst_test_return"] == pytest.approx(-0.02)
    assert summary["win_rate_vs_buy_and_hold"] == pytest.approx(2 / 3)
    assert summary["win_rate_vs_static_80pct"] == pytest.approx(2 / 3)
