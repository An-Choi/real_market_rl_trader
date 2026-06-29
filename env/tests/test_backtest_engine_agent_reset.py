from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_SRC = PROJECT_ROOT / "agent" / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from env.trading_env import TradingEnvironment  # noqa: E402
from experiments.backtest_engine import BacktestEngine  # noqa: E402
from policies.baseline_agents import BuyAndHoldAgent  # noqa: E402


def _make_environment() -> TradingEnvironment:
    market_data = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0, 103.0],
            "feature": [0.0, 0.1, 0.2, 0.3],
        }
    )
    return TradingEnvironment(market_data=market_data, feature_columns=["feature"])


def test_backtest_engine_resets_stateful_agent_each_run() -> None:
    agent = BuyAndHoldAgent()
    engine = BacktestEngine(agent=agent, environment=_make_environment())

    first = engine.run(max_steps=2, seed=0)
    second = engine.run(max_steps=2, seed=0)

    assert first["action"].tolist() == [1, 0]
    assert second["action"].tolist() == [1, 0]
