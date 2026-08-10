"""FullSplitValidationCallback 다종목 평균 선택 테스트."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from models.validation import FullSplitValidationCallback, ValidationSnapshot


def _snapshot(total_return: float, timestep: int = 0) -> ValidationSnapshot:
    return ValidationSnapshot(
        timestep=timestep, total_return=total_return, final_portfolio_value=10_000.0,
        max_drawdown=-0.1, turnover=0.5, trade_count=3,
        hold_action_rate=0.8, add_action_rate=0.1, clear_action_rate=0.1,
    )


class _Logger:
    def __init__(self):
        self.records = {}

    def record(self, key, value):
        self.records[key] = value


def _callback() -> tuple[FullSplitValidationCallback, _Logger]:
    cb = FullSplitValidationCallback(
        {"AAA": object(), "BBB": object()},
        eval_freq=100, use_action_masks=False, verbose=0,
    )
    cb.model = type("M", (), {
        "get_parameters": lambda self: {"w": 1},
        "set_parameters": lambda self, p, exact_match=True: None,
    })()
    logger = _Logger()
    # BaseCallback.logger는 property일 수 있다 — 구현에서 record 경로를
    # self._record_value(key, value) 내부 메서드로 감싸고, 테스트는 그 메서드를
    # logger stub으로 patch한다 (아래는 그 계약을 전제로 한 예시).
    cb._record_value = logger.record
    cb.num_timesteps = 0
    return cb, logger


def test_best_selection_uses_mean_across_symbols():
    cb, _ = _callback()
    returns = iter([
        {"AAA": 0.10, "BBB": 0.00},   # 1차: mean 0.05
        {"AAA": 0.02, "BBB": 0.20},   # 2차: mean 0.11 → best 갱신
        {"AAA": 0.30, "BBB": -0.20},  # 3차: mean 0.05 → 갱신 안 됨
    ])

    with patch.object(
        FullSplitValidationCallback,
        "_run_all_splits",
        side_effect=lambda self: {s: _snapshot(r) for s, r in next(returns).items()},
        autospec=True,
    ):
        cb._evaluate_and_maybe_update()
        cb._evaluate_and_maybe_update()
        cb._evaluate_and_maybe_update()

    assert cb.best_score == pytest.approx(0.11)
    summary = cb.summary()
    assert summary["metric"] == "mean_total_return"
    assert summary["best"]["mean_total_return"] == pytest.approx(0.11)
    assert set(summary["best"]["per_symbol"]) == {"AAA", "BBB"}


def test_logger_records_per_symbol_and_mean():
    cb, logger = _callback()
    with patch.object(
        FullSplitValidationCallback,
        "_run_all_splits",
        return_value={"AAA": _snapshot(0.10), "BBB": _snapshot(0.30)},
        autospec=False,
    ):
        cb._evaluate_and_maybe_update()
    assert logger.records["validation/return_AAA"] == pytest.approx(0.10)
    assert logger.records["validation/return_BBB"] == pytest.approx(0.30)
    assert logger.records["validation/mean_total_return"] == pytest.approx(0.20)


def test_rejects_empty_env_dict():
    with pytest.raises(ValueError):
        FullSplitValidationCallback({}, eval_freq=100, use_action_masks=False)
