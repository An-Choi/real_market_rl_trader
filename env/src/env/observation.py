"""Observation·mask 공유 순수 함수 — env와 serving이 같은 공식을 호출한다.

train/serve parity의 핵심: portfolio state 5필드와 action mask의 공식이
여기 한 곳에만 존재한다 (observation-contract §2·§3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def adv_value_from_row(row: pd.Series) -> float | None:
    """v4 pass-through 컬럼 Adv20 → 유동성 계산 입력. 부재/NaN은 None(기준점 0)."""
    raw = row.get("Adv20")
    if raw is None or pd.isna(raw):
        return None
    return float(raw)


def extract_feature_vector(row: pd.Series, feature_columns: list[str]) -> np.ndarray:
    return (
        pd.to_numeric(row[feature_columns], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def build_portfolio_state(
    *,
    units_held: int,
    max_units: int,
    shares_held: float,
    price: float,
    initial_cash: float,
    unit_fraction: float,
    bars_since_entry: int | None,
    duration_horizon_bars: int,
    step_in_day: int,
    nominal_bars_per_day: int,
    adv_value: float | None,
) -> np.ndarray:
    units_held_frac = units_held / max(max_units, 1)
    held_value = shares_held * price
    cost_basis = units_held * initial_cash * unit_fraction
    unrealized_pnl_norm = (held_value - cost_basis) / max(initial_cash, 1e-9)
    if units_held > 0 and bars_since_entry is not None:
        holding_duration_norm = min(bars_since_entry / duration_horizon_bars, 1.0)
    else:
        holding_duration_norm = 0.0
    tod_frac = min(step_in_day / max(nominal_bars_per_day - 1, 1), 1.0)
    # liquidity_pressure: unit 주문금액이 ADV에서 차지하는 비중의 log10 스케일.
    # 기준점 0 = 1e-4 (unit이 ADV의 0.01%), ±1 = 1e-8~1. adv 부재 시 기준점 0.
    if adv_value is not None and np.isfinite(adv_value) and adv_value > 0:
        unit_notional = initial_cash * unit_fraction
        liquidity_pressure = float(
            np.clip((np.log10(unit_notional / adv_value) + 4.0) / 4.0, -1.0, 1.0)
        )
    else:
        liquidity_pressure = 0.0
    return np.array(
        [units_held_frac, unrealized_pnl_norm, holding_duration_norm, tod_frac,
         liquidity_pressure],
        dtype=np.float32,
    )


def can_afford_add(
    *,
    cash: float,
    initial_cash: float,
    unit_fraction: float,
    friction_model,
    price: float,
    trade_date,
    liquidity_score: float | None = None,
) -> bool:
    unit_notional = initial_cash * unit_fraction
    buy_friction = friction_model.calculate_total_friction(
        trade_value=unit_notional,
        side="buy",
        liquidity_score=liquidity_score,
        price=price,
        trade_date=trade_date,
    )
    return cash >= unit_notional + buy_friction


def compute_action_mask(
    *,
    units_held: int,
    max_units: int,
    cash: float,
    initial_cash: float,
    unit_fraction: float,
    friction_model,
    price: float,
    trade_date,
    liquidity_score: float | None = None,
) -> np.ndarray:
    return np.array(
        [
            True,
            units_held < max_units
            and can_afford_add(
                cash=cash,
                initial_cash=initial_cash,
                unit_fraction=unit_fraction,
                friction_model=friction_model,
                price=price,
                trade_date=trade_date,
                liquidity_score=liquidity_score,
            ),
            units_held > 0,
        ],
        dtype=bool,
    )


def assemble_observation(features: np.ndarray, portfolio_state: np.ndarray) -> np.ndarray:
    return np.concatenate([features, portfolio_state]).astype(np.float32)
