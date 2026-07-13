from __future__ import annotations

from pathlib import Path

from utils.config_loader import load_config


def test_config_has_multiday_episode_settings() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = load_config(config_path)
    assert config["environment"]["episode_days"] == 20
    assert config["environment"]["nominal_bars_per_day"] == 64
