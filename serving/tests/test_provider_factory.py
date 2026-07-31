import dataclasses

import pytest

from config import ServingConfig
from live_market_data import LiveKISProvider, build_provider
from market_data import HistoricalParquetProvider


def _config(tmp_path, **overrides):
    base = dict(
        artifact_dir=tmp_path / "a", data_dir=tmp_path / "d",
        symbols=["005930"], audit_log_dir=tmp_path / "l",
        kis_token_cache=tmp_path / "tok.json",
    )
    base.update(overrides)
    return ServingConfig(**base)


def test_historical_provider_by_default(tmp_path, monkeypatch):
    # historical은 KIS 환경변수 없이도 기동해야 한다 (기존 동작 불변)
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    provider = build_provider(_config(tmp_path))
    assert isinstance(provider, HistoricalParquetProvider)


def test_live_provider_with_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KIS_APP_KEY", "k")
    monkeypatch.setenv("KIS_APP_SECRET", "s")
    monkeypatch.setenv("KIS_ENV", "demo")
    provider = build_provider(_config(tmp_path, provider="live"))
    assert isinstance(provider, LiveKISProvider)


def test_live_provider_missing_env_is_startup_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    # spec §1: 환경변수 누락은 ProviderError(503)가 아니라 기동 실패(RuntimeError 전파)
    with pytest.raises(RuntimeError, match="KIS_APP_KEY"):
        build_provider(_config(tmp_path, provider="live"))
