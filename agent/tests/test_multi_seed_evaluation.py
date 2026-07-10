from __future__ import annotations

import pytest

from policies.evaluation import summarize_seed_runs


def test_summarize_seed_runs_aggregates_metric_mean_and_std() -> None:
    summary = summarize_seed_runs([
        {"agent": "random", "metrics": {"total_return": 0.10, "trade_count": 1.0}},
        {"agent": "random", "metrics": {"total_return": -0.10, "trade_count": 3.0}},
    ])

    assert len(summary["runs"]) == 2
    assert summary["mean_metrics"]["total_return"] == pytest.approx(0.0)
    assert summary["std_metrics"]["total_return"] == pytest.approx(0.10)
    assert summary["mean_metrics"]["trade_count"] == pytest.approx(2.0)
    assert summary["std_metrics"]["trade_count"] == pytest.approx(1.0)
