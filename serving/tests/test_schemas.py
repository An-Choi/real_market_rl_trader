import pytest
from pydantic import ValidationError

from schemas import PredictRequest


def _payload(**overrides):
    base = {
        "symbol": "005930",
        "portfolio": {"units_held": 2, "shares_held": 512.0,
                      "bars_since_entry": 37, "available_cash": 5_800.0},
    }
    base.update(overrides)
    return base


def test_valid_request_parses():
    req = PredictRequest.model_validate(_payload())
    assert req.as_of is None
    assert req.portfolio.cost_basis is None  # v4 이하 요청 하위 호환


def test_cost_basis_parses_for_held_position() -> None:
    payload = _payload()
    payload["portfolio"]["cost_basis"] = 4_000.0
    req = PredictRequest.model_validate(payload)
    assert req.portfolio.cost_basis == 4_000.0


@pytest.mark.parametrize("portfolio", [
    {"units_held": 0, "shares_held": 1.0, "bars_since_entry": 0, "available_cash": 1.0},   # flat인데 shares
    {"units_held": 1, "shares_held": 0.0, "bars_since_entry": 0, "available_cash": 1.0},   # 보유인데 shares 0
    {"units_held": 0, "shares_held": 0.0, "bars_since_entry": 3, "available_cash": 1.0},   # flat인데 duration
    {"units_held": -1, "shares_held": 0.0, "bars_since_entry": 0, "available_cash": 1.0},  # 음수
    {"units_held": 1, "shares_held": float("nan"), "bars_since_entry": 0, "available_cash": 1.0},  # NaN
    {"units_held": True, "shares_held": 1.0, "bars_since_entry": 0, "available_cash": 1.0},   # bool → strict int 거부
    {"units_held": 1.0, "shares_held": 1.0, "bars_since_entry": 0, "available_cash": 1.0},    # float → strict int 거부
    {"units_held": 0, "shares_held": 0.0, "cost_basis": 1.0, "bars_since_entry": 0, "available_cash": 1.0},  # flat인데 cost basis
    {"units_held": 1, "shares_held": 1.0, "cost_basis": float("nan"), "bars_since_entry": 0, "available_cash": 1.0},
])
def test_invariant_violations_rejected(portfolio):
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(_payload(portfolio=portfolio))


def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(_payload(asof="2026-07-30T09:00:00+09:00"))


def test_unknown_portfolio_field_rejected():
    portfolio = {"units_held": 2, "shares_held": 512.0,
                 "bars_since_entry": 37, "available_cash": 5_800.0,
                 "extra_field": "unexpected"}
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(_payload(portfolio=portfolio))
