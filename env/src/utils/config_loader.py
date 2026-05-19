"""YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    # TODO: Add schema validation and environment variable overrides.
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
