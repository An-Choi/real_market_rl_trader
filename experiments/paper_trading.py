"""Paper-trading entrypoint placeholder."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paper_trading_engine import PaperTradingEngine


def main() -> None:
    """Reserve the executable paper-trading hook for live-like experiments."""
    raise NotImplementedError("Paper trading is not wired to a market data stream yet.")


__all__ = ["PaperTradingEngine", "main"]


if __name__ == "__main__":
    main()
