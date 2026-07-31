"""요청 + 시장 데이터 → raw observation·mask — spec §2 '서버가 하는 일'.

completed-minute causal cutoff(1분봉 라벨 + 1분 <= as_of)를 **여기서 다시
적용한다** — provider도 같은 규칙을 쓰지만, builder 계약을 provider 구현과
독립적으로 안전하게 만들기 위한 defense-in-depth다(미래 row가 섞인 입력이
와도 미래 bar 선택·음수 staleness가 불가능해진다).
cutoff 후 transform하면 마지막 5분 bar가 결정 bar다 — resample이
label=right·min_bars=5라 불완전 bucket은 존재하지 않는다.
정규화는 하지 않는다 — agent.predict() 내부에서만 적용(이중 정규화 금지).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from env.observation import (
    assemble_observation,
    build_portfolio_state,
    compute_action_mask,
    extract_feature_vector,
)
from errors import InsufficientHistoryError, StaleDataError


@dataclass(frozen=True)
class DecisionInputs:
    observation: np.ndarray
    action_mask: np.ndarray
    bar_ts: pd.Timestamp
    price: float


def build_decision_inputs(
    *,
    bars_1m: pd.DataFrame,
    as_of: pd.Timestamp,
    units_held: int,
    shares_held: float,
    bars_since_entry: int,
    available_cash: float,
    env_params: dict,
    friction_model,
    max_bar_age: pd.Timedelta,
    feature_engineer,
) -> DecisionInputs:
    # defense-in-depth: 완료 분봉 cutoff를 transform 입력 단계에서 재적용
    ts = pd.to_datetime(bars_1m["Timestamp"])
    bars_1m = bars_1m[ts + pd.Timedelta(minutes=1) <= as_of]
    if bars_1m.empty:
        raise InsufficientHistoryError(
            "no completed minute bars at or before as_of",
            {"as_of": str(as_of)},
        )
    try:
        featured = feature_engineer.transform(bars_1m)
    except ValueError as exc:
        # 모든 날이 결손 판정으로 제거되면 파이프라인 내부 concat이
        # ValueError("No objects to concatenate")를 낸다 — 데이터 부족과
        # 동일한 fail-closed 503(INSUFFICIENT_HISTORY)으로 매핑한다.
        raise InsufficientHistoryError(
            f"feature pipeline produced no rows: {exc}",
            {"as_of": str(as_of), "input_rows": int(len(bars_1m))},
        ) from exc
    if featured.empty:
        raise InsufficientHistoryError(
            "not enough history to produce any feature row",
            {"as_of": str(as_of), "input_rows": int(len(bars_1m))},
        )
    row = featured.iloc[-1]
    bar_ts = pd.Timestamp(row["Timestamp"])
    if as_of - bar_ts > max_bar_age:
        raise StaleDataError(
            "latest completed bar is older than max_bar_age",
            {"latest_bar_ts": str(bar_ts), "as_of": str(as_of),
             "max_bar_age_minutes": int(max_bar_age.total_seconds() // 60)},
        )
    price = float(row["Close"])

    day_rows = featured[pd.to_datetime(featured["Timestamp"]).dt.date == bar_ts.date()]
    step_in_day = len(day_rows) - 1  # 결정 bar의 당일 내 0-based index

    portfolio_state = build_portfolio_state(
        units_held=units_held,
        max_units=int(env_params["max_units"]),
        shares_held=shares_held,
        price=price,
        initial_cash=float(env_params["initial_cash"]),
        unit_fraction=float(env_params["unit_fraction"]),
        bars_since_entry=bars_since_entry if units_held > 0 else None,
        duration_horizon_bars=int(env_params["duration_horizon_bars"]),
        step_in_day=step_in_day,
        nominal_bars_per_day=int(env_params["nominal_bars_per_day"]),
    )
    features = extract_feature_vector(row, list(feature_engineer.FEATURE_COLUMNS))
    action_mask = compute_action_mask(
        units_held=units_held,
        max_units=int(env_params["max_units"]),
        cash=available_cash,
        initial_cash=float(env_params["initial_cash"]),
        unit_fraction=float(env_params["unit_fraction"]),
        friction_model=friction_model,
        price=price,
        trade_date=bar_ts.date(),
        liquidity_score=None,  # featured 데이터에 liquidity_score 없음 — env와 동일 경로
    )
    return DecisionInputs(
        observation=assemble_observation(features, portfolio_state),
        action_mask=action_mask,
        bar_ts=bar_ts,
        price=price,
    )
