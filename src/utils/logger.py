"""Logger setup utility."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str = "real_market_rl_trader",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Create a configured logger."""
    # TODO: Add structured logging for experiments and backtests.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
