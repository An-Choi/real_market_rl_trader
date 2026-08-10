from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.tensorboard_callback import (
    TradingMetricsTensorBoardCallback,
    active_reward_info_keys,
)


class CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, name: str, value: float, exclude=None) -> None:
        self.records[name] = value


def _info(portfolio_return, benchmark_return, units, friction, forced=False):
    return {
        "portfolio_return": portfolio_return,
        "benchmark_simple_return": benchmark_return,
        "units_held": units,
        "friction_cost": friction,
        "forced_clear": forced,
        "base_return": portfolio_return,
        "drawdown_penalty": 0.01,
    }


def test_callback_records_only_compact_trading_dashboard() -> None:
    logger = CaptureLogger()
    callback = TradingMetricsTensorBoardCallback(
        initial_cash=10_000.0,
        max_units=5,
        reward_info_keys=("base_return",),
    )
    callback.model = SimpleNamespace(logger=logger)
    callback._on_rollout_start()
    callback.locals = {
        "infos": [_info(0.01, 0.005, 1, 2.0)],
        "actions": [1],
        "rewards": [0.8],
        "dones": [False],
    }
    assert callback._on_step() is True
    callback.locals = {
        "infos": [_info(0.02, 0.01, 0, 3.0, True)],
        "actions": [2],
        "rewards": [0.4],
        "dones": [True],
    }
    assert callback._on_step() is True
    callback._on_rollout_end()

    assert logger.records["risk/exposure_mean"] == pytest.approx(0.1)
    assert logger.records["policy/add_rate"] == pytest.approx(0.5)
    assert logger.records["policy/clear_rate"] == pytest.approx(0.5)
    assert logger.records["trading/friction_total"] == pytest.approx(5.0)
    assert logger.records["performance/episode_return_mean"] == pytest.approx(1.01 * 1.02 - 1)
    assert logger.records["performance/benchmark_episode_return_mean"] == pytest.approx(
        1.005 * 1.01 - 1
    )
    assert "performance/cumulative_return" not in logger.records
    assert logger.records["reward/shaped_mean"] == pytest.approx(0.6)
    assert "reward/base_return_mean" in logger.records
    assert "reward/drawdown_penalty_mean" not in logger.records
    assert not any(name.startswith("daily/") for name in logger.records)


def test_only_enabled_reward_terms_are_selected() -> None:
    assert active_reward_info_keys({}) == ("base_return",)
    assert active_reward_info_keys({
        "drawdown_penalty_rate": 0.2,
        "benchmark_relative_rate": 0.5,
    }) == ("base_return", "benchmark_relative_reward", "drawdown_penalty")
