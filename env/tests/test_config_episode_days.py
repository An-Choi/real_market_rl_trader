from __future__ import annotations

from pathlib import Path

from utils.config_loader import load_config


def test_config_has_multiday_episode_settings() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = load_config(config_path)
    assert config["environment"]["episode_days"] == 20
    assert config["environment"]["nominal_bars_per_day"] == 64
    assert config["agent"]["rl_model_name"] == "MaskablePPO"
    assert config["agent"]["ppo"]["gamma"] == 0.999
    assert config["environment"]["benchmark_relative_rate"] == 0.0
    assert config["environment"]["drawdown_penalty_rate"] == 0.0
    assert config["environment"]["downside_penalty_rate"] == 0.0
