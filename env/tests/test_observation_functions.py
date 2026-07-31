"""observation.py 순수 함수 단위 테스트 — 공식은 observation-contract §2·§3 기준."""
import numpy as np
import pandas as pd
import pytest

from env.observation import (
    assemble_observation,
    build_portfolio_state,
    can_afford_add,
    compute_action_mask,
    extract_feature_vector,
)
from friction.friction_model import FrictionModel


ENV_PARAMS = dict(
    max_units=5, initial_cash=10_000.0, unit_fraction=0.199,
    duration_horizon_bars=1280, nominal_bars_per_day=64,
)


def test_portfolio_state_flat_position_is_zeroed():
    state = build_portfolio_state(
        units_held=0, shares_held=0.0, price=300_000.0,
        bars_since_entry=None, step_in_day=0, **ENV_PARAMS,
    )
    assert state.dtype == np.float32
    np.testing.assert_array_equal(state, np.zeros(4, dtype=np.float32))


def test_portfolio_state_held_position_formulas():
    # 2 Units, shares 0.02주, price 305000 → contract §2 공식 수기 계산과 일치
    state = build_portfolio_state(
        units_held=2, shares_held=0.02, price=305_000.0,
        bars_since_entry=37, step_in_day=10, **ENV_PARAMS,
    )
    held_value = 0.02 * 305_000.0
    cost_basis = 2 * 10_000.0 * 0.199
    assert state[0] == np.float32(2 / 5)
    assert state[1] == np.float32((held_value - cost_basis) / 10_000.0)
    assert state[2] == np.float32(37 / 1280)
    assert state[3] == np.float32(10 / 63)


def test_portfolio_state_clips_duration_and_tod():
    state = build_portfolio_state(
        units_held=1, shares_held=0.001, price=300_000.0,
        bars_since_entry=99_999, step_in_day=99_999, **ENV_PARAMS,
    )
    assert state[2] == np.float32(1.0)
    assert state[3] == np.float32(1.0)


def test_tod_frac_degenerate_grid_no_division_error():
    params = dict(ENV_PARAMS, nominal_bars_per_day=1)
    state = build_portfolio_state(
        units_held=0, shares_held=0.0, price=1.0,
        bars_since_entry=None, step_in_day=0, **params,
    )
    assert np.isfinite(state[3])


def test_can_afford_add_gates_on_notional_plus_friction():
    friction = FrictionModel(fee_rate=0.1, spread_rate=0.0, slippage_rate=0.0,
                             execution_uncertainty_rate=0.0, sell_tax_rate=0.0,
                             dynamic_spread=False, date_based_sell_tax=False)
    kwargs = dict(initial_cash=10_000.0, unit_fraction=0.2, friction_model=friction,
                  price=300_000.0, trade_date=pd.Timestamp("2026-07-01").date())
    # notional 2000 + fee 200 = 2200
    assert can_afford_add(cash=2200.0, **kwargs) is True
    assert can_afford_add(cash=2199.99, **kwargs) is False


def test_action_mask_layout():
    friction = FrictionModel(fee_rate=0.0, spread_rate=0.0, slippage_rate=0.0,
                             execution_uncertainty_rate=0.0, sell_tax_rate=0.0,
                             dynamic_spread=False, date_based_sell_tax=False)
    common = dict(max_units=5, initial_cash=10_000.0, unit_fraction=0.2,
                  friction_model=friction, price=300_000.0,
                  trade_date=pd.Timestamp("2026-07-01").date())
    flat = compute_action_mask(units_held=0, cash=10_000.0, **common)
    np.testing.assert_array_equal(flat, [True, True, False])
    maxed = compute_action_mask(units_held=5, cash=10_000.0, **common)
    np.testing.assert_array_equal(maxed, [True, False, True])
    broke = compute_action_mask(units_held=1, cash=0.0, **common)
    np.testing.assert_array_equal(broke, [True, False, True])


def test_extract_and_assemble_dtypes():
    row = pd.Series({"f1": 1.5, "f2": "bad", "f3": None})
    features = extract_feature_vector(row, ["f1", "f2", "f3"])
    np.testing.assert_array_equal(features, np.array([1.5, 0.0, 0.0], dtype=np.float32))
    obs = assemble_observation(features, np.zeros(4, dtype=np.float32))
    assert obs.dtype == np.float32 and obs.shape == (7,)
