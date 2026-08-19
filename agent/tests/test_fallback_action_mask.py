"""MaskablePPO fallback mask — 관측의 units_held_frac 필드에서 mask 유도.

portfolio state가 5필드(v4)가 되면서 units_held_frac 위치는 끝에서 5번째다.
잘못된 인덱스는 PnL 부호에 따라 clear가 차단되는 실거래급 버그가 된다
(PR #18 리뷰 P1).
"""

from __future__ import annotations

import numpy as np
import pytest

from models.rl_agent import RLAgent


def _obs(units_held_frac: float, pnl: float) -> np.ndarray:
    # 시장 feature 11개(0.0) + [units, pnl, duration, tod, liquidity]
    return np.array([0.0] * 11 + [units_held_frac, pnl, 0.1, 0.5, 0.25],
                    dtype=np.float32)


def test_held_position_with_negative_pnl_allows_clear() -> None:
    # 리뷰 재현 케이스: units 0.4, 손익 −0.02 → clear는 허용되어야 한다
    mask = RLAgent._mask_from_observation(_obs(0.4, -0.02))
    np.testing.assert_array_equal(mask, [True, True, True, True])


def test_flat_position_blocks_clear() -> None:
    mask = RLAgent._mask_from_observation(_obs(0.0, 0.0))
    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_full_position_blocks_add() -> None:
    mask = RLAgent._mask_from_observation(_obs(1.0, 0.03))
    np.testing.assert_array_equal(mask, [True, False, True, True])


def test_observation_shorter_than_portfolio_state_raises() -> None:
    with pytest.raises(ValueError):
        RLAgent._mask_from_observation(np.zeros(4, dtype=np.float32))
