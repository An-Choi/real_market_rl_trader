from pathlib import Path

import pytest

from config import ServingConfig, load_serving_config


def test_load_config_yaml(tmp_path):
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: artifacts/ppo-fs3-x\n"
        "data_dir: data/raw\n"
        "symbols: ['005930']\n"
        "audit_log_dir: serving/logs\n",
        encoding="utf-8",
    )
    cfg = load_serving_config(cfg_file)
    assert cfg.symbols == ["005930"]
    assert cfg.max_bar_age_minutes == 10   # default
    assert cfg.warmup_days == 30           # default
    assert isinstance(cfg.artifact_dir, Path)


def test_load_config_rejects_unknown_key(tmp_path):
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: a\ndata_dir: d\nsymbols: ['005930']\n"
        "audit_log_dir: l\nmystery: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mystery"):
        load_serving_config(cfg_file)


def test_load_config_rejects_nonlocal_host(tmp_path):
    # 인증/TLS 없는 스펙 범위 — localhost 바인딩 강제 (spec §5)
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: a\ndata_dir: d\nsymbols: ['005930']\n"
        "audit_log_dir: l\nhost: 0.0.0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="host"):
        load_serving_config(cfg_file)
