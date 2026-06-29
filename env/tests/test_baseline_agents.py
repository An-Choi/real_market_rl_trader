from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_SRC = PROJECT_ROOT / "agent" / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from policies.baseline_agents import (  # noqa: E402
    BuyAndHoldAgent,
    MovingAverageCrossoverAgent,
    RandomAgent,
    make_baseline_agent,
)


def test_buy_and_hold_reset_replays_first_buy() -> None:
    agent = BuyAndHoldAgent()

    first_action, _ = agent.predict(None)
    second_action, _ = agent.predict(None)
    agent.reset()
    replay_action, _ = agent.predict(None)

    assert first_action == 1
    assert second_action == 0
    assert replay_action == 1


def test_random_agent_reset_replays_seeded_sequence() -> None:
    agent = RandomAgent(seed=123)

    first = [agent.predict(None)[0] for _ in range(5)]
    agent.reset()
    second = [agent.predict(None)[0] for _ in range(5)]

    assert first == second


def test_make_baseline_agent_creates_known_agents() -> None:
    assert isinstance(make_baseline_agent("buy_and_hold"), BuyAndHoldAgent)
    assert isinstance(
        make_baseline_agent("moving_average_crossover"),
        MovingAverageCrossoverAgent,
    )
    assert isinstance(make_baseline_agent("random", seed=123), RandomAgent)


def test_make_baseline_agent_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown baseline agent"):
        make_baseline_agent("not_a_strategy")


def test_moving_average_crossover_reset_is_noop() -> None:
    agent = MovingAverageCrossoverAgent()
    market_row = pd.Series({"ma_5": 2.0, "ma_20": 1.0})

    agent.reset()
    action, _ = agent.predict(None, market_row=market_row)

    assert action == 1
