from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import responses

from pipeline.kis_auth import KISAuth, KISConfig


@pytest.fixture
def config(tmp_path: Path) -> KISConfig:
    return KISConfig(
        env="demo",
        app_key="APPKEY1234567890",
        app_secret="APPSECRET1234567890",
        token_cache_path=tmp_path / "kis_token.json",
    )


def test_demo_uses_paper_url(config: KISConfig) -> None:
    auth = KISAuth(config)
    assert auth.base_url == "https://openapivts.koreainvestment.com:29443"


def test_real_uses_prod_url(config: KISConfig) -> None:
    config = KISConfig(
        env="real",
        app_key=config.app_key,
        app_secret=config.app_secret,
        token_cache_path=config.token_cache_path,
    )
    auth = KISAuth(config)
    assert auth.base_url == "https://openapi.koreainvestment.com:9443"


@responses.activate
def test_get_token_issues_when_no_cache(config: KISConfig) -> None:
    responses.add(
        responses.POST,
        "https://openapivts.koreainvestment.com:29443/oauth2/tokenP",
        json={
            "access_token": "TOKEN_XYZ",
            "expires_in": 86400,
        },
        status=200,
    )
    auth = KISAuth(config)
    token = auth.get_token()
    assert token == "TOKEN_XYZ"
    assert config.token_cache_path.exists()
    cached = json.loads(config.token_cache_path.read_text())
    assert cached["access_token"] == "TOKEN_XYZ"


@responses.activate
def test_get_token_uses_cache_when_valid(config: KISConfig) -> None:
    config.token_cache_path.write_text(
        json.dumps({
            "access_token": "CACHED_TOKEN",
            "expires_at": (datetime.now() + timedelta(hours=12)).isoformat(),
        })
    )
    auth = KISAuth(config)
    token = auth.get_token()
    assert token == "CACHED_TOKEN"
    assert len(responses.calls) == 0  # HTTP 호출 없음


@responses.activate
def test_get_token_reissues_when_cache_expired(config: KISConfig) -> None:
    config.token_cache_path.write_text(
        json.dumps({
            "access_token": "OLD_TOKEN",
            "expires_at": (datetime.now() - timedelta(hours=1)).isoformat(),
        })
    )
    responses.add(
        responses.POST,
        "https://openapivts.koreainvestment.com:29443/oauth2/tokenP",
        json={
            "access_token": "NEW_TOKEN",
            "expires_in": 86400,
        },
        status=200,
    )
    auth = KISAuth(config)
    token = auth.get_token()
    assert token == "NEW_TOKEN"


def test_from_env_reads_environment_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KIS_ENV", "demo")
    monkeypatch.setenv("KIS_APP_KEY", "ENVKEY")
    monkeypatch.setenv("KIS_APP_SECRET", "ENVSECRET")
    config = KISConfig.from_env(token_cache_path=tmp_path / "tok.json")
    assert config.app_key == "ENVKEY"
    assert config.app_secret == "ENVSECRET"
    assert config.env == "demo"


def test_from_env_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="KIS_APP_KEY"):
        KISConfig.from_env(token_cache_path=tmp_path / "tok.json")


def test_credentials_not_in_repr(config):
    assert config.app_key not in repr(config)
    assert config.app_secret not in repr(config)


def test_cache_is_private_and_atomic(config, monkeypatch):
    import os
    import stat
    auth = KISAuth(config)
    auth._write_cache("old", datetime.now() + timedelta(hours=1))
    assert stat.S_IMODE(config.token_cache_path.stat().st_mode) == 0o600
    original = config.token_cache_path.read_text()
    def fail_replace(*args):
        raise OSError("simulated interruption")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        auth._write_cache("new", datetime.now() + timedelta(hours=1))
    assert config.token_cache_path.read_text() == original
    assert not list(config.token_cache_path.parent.glob(".kis-token-*"))


@pytest.mark.parametrize("payload", [{}, [], {"expires_at": "bad"},
    {"expires_at": "2099-01-01T00:00:00", "access_token": 3}])
def test_malformed_cache_is_ignored(config, payload):
    config.token_cache_path.write_text(json.dumps(payload))
    assert KISAuth(config)._read_cache() is None
