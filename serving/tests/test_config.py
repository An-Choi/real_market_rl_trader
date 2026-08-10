from pathlib import Path

import pytest

from config import ServingConfig, load_serving_config


def _write_cfg(tmp_path, extra: str) -> "Path":
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: a\ndata_dir: d\nsymbols: ['005930']\n"
        "audit_log_dir: l\n" + extra,
        encoding="utf-8",
    )
    return cfg_file


def test_provider_defaults_to_historical(tmp_path):
    cfg = load_serving_config(_write_cfg(tmp_path, ""))
    assert cfg.provider == "historical"
    assert cfg.kis_token_cache == Path("data/.kis_token.json")
    assert cfg.kis_rate_limit_sleep == 0.5


def test_provider_live_accepted_and_token_cache_is_path(tmp_path):
    cfg = load_serving_config(_write_cfg(
        tmp_path, "provider: live\nkis_token_cache: data/tok.json\n"))
    assert cfg.provider == "live"
    assert isinstance(cfg.kis_token_cache, Path)


def test_unknown_provider_rejected(tmp_path):
    with pytest.raises(ValueError, match="provider"):
        load_serving_config(_write_cfg(tmp_path, "provider: websocket\n"))


@pytest.mark.parametrize("bad", ["-0.1", ".nan", ".inf", "nan", "true", "'0.5'"])
def test_invalid_rate_limit_sleep_rejected(tmp_path, bad):
    # ".nan"/".inf"는 YAML이 실제 float nan/inf로 파싱한다 — isfinite 경로 검증
    with pytest.raises(ValueError, match="kis_rate_limit_sleep"):
        load_serving_config(_write_cfg(tmp_path, f"kis_rate_limit_sleep: {bad}\n"))


def test_zero_rate_limit_sleep_accepted(tmp_path):
    cfg = load_serving_config(_write_cfg(tmp_path, "kis_rate_limit_sleep: 0\n"))
    assert cfg.kis_rate_limit_sleep == 0


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


def test_load_config_rejects_string_symbols(tmp_path):
    # str도 "non-empty str의 iterable"이라 all(isinstance(s, str)...)를 통과해버린다
    # — symbols: "005930"이 ["0","0","5","9","3","0"]처럼 취급되면 안 된다.
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: a\ndata_dir: d\nsymbols: '005930'\n"
        "audit_log_dir: l\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="symbols"):
        load_serving_config(cfg_file)


@pytest.mark.parametrize("warmup_days", [0, -1])
def test_load_config_rejects_nonpositive_warmup_days(tmp_path, warmup_days):
    # warmup_days<=0이면 provider의 [-warmup_days:] slicing이 전체(또는 예상외) 범위를
    # 선택한다 — 반드시 양의 정수여야 한다.
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: a\ndata_dir: d\nsymbols: ['005930']\n"
        f"audit_log_dir: l\nwarmup_days: {warmup_days}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="warmup_days"):
        load_serving_config(cfg_file)


def test_load_config_rejects_non_integer_warmup_days(tmp_path):
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: a\ndata_dir: d\nsymbols: ['005930']\n"
        "audit_log_dir: l\nwarmup_days: 2.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="warmup_days"):
        load_serving_config(cfg_file)


def test_load_config_rejects_nonpositive_max_bar_age_minutes(tmp_path):
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: a\ndata_dir: d\nsymbols: ['005930']\n"
        "audit_log_dir: l\nmax_bar_age_minutes: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_bar_age_minutes"):
        load_serving_config(cfg_file)


@pytest.mark.parametrize("port", [0, 70000])
def test_load_config_rejects_out_of_range_port(tmp_path, port):
    cfg_file = tmp_path / "serving.yaml"
    cfg_file.write_text(
        "artifact_dir: a\ndata_dir: d\nsymbols: ['005930']\n"
        f"audit_log_dir: l\nport: {port}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="port"):
        load_serving_config(cfg_file)
