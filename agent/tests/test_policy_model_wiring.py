from __future__ import annotations

import pytest
import pandas as pd

from models import RLAgent, make_rl_agent
from policies import (
    SUPPORTED_BASELINES,
    BuyAndHoldAgent,
    MovingAverageCrossoverAgent,
    RandomAgent,
    StaticAllocationAgent,
    VolatilityScaledAgent,
    make_baseline_agent,
)


def test_make_baseline_agent_by_name() -> None:
    assert SUPPORTED_BASELINES == (
        "cash",
        "static_20pct",
        "static_40pct",
        "static_60pct",
        "static_80pct",
        "buy_and_hold",
        "volatility_scaled",
        "random",
        "ma_crossover",
    )
    assert isinstance(make_baseline_agent("cash"), StaticAllocationAgent)
    assert make_baseline_agent("static_80pct").target_units == 4
    assert isinstance(make_baseline_agent("buy_and_hold"), BuyAndHoldAgent)
    assert isinstance(make_baseline_agent("volatility_scaled"), VolatilityScaledAgent)
    assert isinstance(make_baseline_agent("random", seed=1), RandomAgent)
    assert isinstance(make_baseline_agent("ma_crossover"), MovingAverageCrossoverAgent)


def test_buy_and_hold_scales_to_full_position_then_holds() -> None:
    agent = BuyAndHoldAgent(target_units=3)

    actions = [agent.predict(None)[0] for _ in range(5)]

    assert actions == [3, 3, 3, 3, 3]


def test_volatility_scaled_baseline_reduces_exposure_when_volatility_spikes() -> None:
    agent = VolatilityScaledAgent(window=3, max_units=5)

    actions = [
        agent.predict(None, market_row=pd.Series({"realized_vol_12": volatility}))[0]
        for volatility in (0.01, 0.01, 0.01, 0.05)
    ]

    assert actions[:3] == [5, 5, 5]
    assert actions[-1] == 1


def test_unknown_baseline_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        make_baseline_agent("missing")


def test_moving_average_baseline_uses_close_history() -> None:
    agent = MovingAverageCrossoverAgent(fast_window=2, slow_window=3)

    actions = [
        agent.predict(None, market_row=pd.Series({"Close": price}))[0]
        for price in (1.0, 2.0, 3.0)
    ]

    assert actions == [0, 0, 5]


def test_make_rl_agent_applies_defaults_without_overwriting_kwargs() -> None:
    agent = make_rl_agent(seed=1, device="cpu", model_kwargs={"seed": 7})

    assert isinstance(agent, RLAgent)
    assert agent.model_name == "PPO"
    assert agent.policy == "MlpPolicy"
    assert agent.model_kwargs["seed"] == 7
    assert agent.model_kwargs["device"] == "cpu"
