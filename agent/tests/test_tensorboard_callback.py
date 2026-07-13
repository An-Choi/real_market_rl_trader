from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.tensorboard_callback import (
    REWARD_INFO_KEYS,
    TradingMetricsTensorBoardCallback,
)


class CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}
        self.dumps: list[tuple[int, dict[str, float]]] = []

    def record(self, name: str, value: float, exclude=None) -> None:
        self.records[name] = value

    def dump(self, step: int) -> None:
        self.dumps.append((step, dict(self.records)))


def _info(
    *,
    portfolio_value: float,
    portfolio_return: float,
    benchmark_return: float,
    friction_cost: float,
    units_held: int,
    forced_clear: bool,
    execution_date: str = "2025-06-02",
    valuation_date: str = "2025-06-02",
) -> dict[str, float | int | bool | str]:
    info: dict[str, float | int | bool | str] = {
        "portfolio_value": portfolio_value,
        "portfolio_return": portfolio_return,
        "benchmark_simple_return": benchmark_return,
        "friction_cost": friction_cost,
        "units_held": units_held,
        "forced_clear": forced_clear,
        "execution_date": execution_date,
        "valuation_date": valuation_date,
    }
    info.update({key: 0.01 for key in REWARD_INFO_KEYS})
    return info


def test_callback_records_requested_trading_metrics() -> None:
    logger = CaptureLogger()
    callback = TradingMetricsTensorBoardCallback(initial_cash=10_000.0, max_units=5)
    callback.model = SimpleNamespace(logger=logger)
    callback._on_rollout_start()

    callback.locals = {
        "infos": [_info(
            portfolio_value=10_100.0,
            portfolio_return=0.01,
            benchmark_return=0.005,
            friction_cost=2.0,
            units_held=1,
            forced_clear=False,
        )],
        "actions": [1],
        "rewards": [0.8],
        "dones": [False],
    }
    assert callback._on_step() is True

    callback.locals = {
        "infos": [_info(
            portfolio_value=10_200.0,
            portfolio_return=10_200.0 / 10_100.0 - 1.0,
            benchmark_return=1.01 / 1.005 - 1.0,
            friction_cost=3.0,
            units_held=0,
            forced_clear=True,
        )],
        "actions": [2],
        "rewards": [0.4],
        "dones": [True],
    }
    assert callback._on_step() is True
    callback._on_rollout_end()

    assert logger.records["portfolio/value_mean"] == pytest.approx(10_150.0)
    assert logger.records["portfolio/exposure_mean"] == pytest.approx(0.1)
    assert logger.records["actions/add_rate"] == pytest.approx(0.5)
    assert logger.records["actions/clear_rate"] == pytest.approx(0.5)
    assert logger.records["actions/hold_rate"] == pytest.approx(0.0)
    assert logger.records["cost/friction_sum"] == pytest.approx(5.0)
    assert logger.records["trading/forced_clear_count"] == pytest.approx(1.0)
    assert logger.records["returns/cumulative_return"] == pytest.approx(0.02)
    assert logger.records["returns/benchmark_cumulative_return"] == pytest.approx(0.01)
    assert logger.records["returns/excess_cumulative_return"] == pytest.approx(
        1.02 / 1.01 - 1.0
    )
    assert logger.records["reward/shaped_mean"] == pytest.approx(0.6)
    for key in REWARD_INFO_KEYS:
        assert f"reward/{key}_mean" in logger.records
    daily_records = logger.dumps[-1][1]
    assert daily_records["daily/date"] == pytest.approx(20250602)
    assert daily_records["daily/portfolio_return"] == pytest.approx(0.02)
    assert daily_records["daily/benchmark_return"] == pytest.approx(0.01)
    assert daily_records["daily/friction_sum"] == pytest.approx(5.0)
    assert daily_records["daily/add_rate"] == pytest.approx(0.5)
    assert daily_records["daily/clear_rate"] == pytest.approx(0.5)
    assert daily_records["daily/forced_clear_count"] == pytest.approx(1.0)
    assert "daily/reward_base_return_mean" in daily_records
