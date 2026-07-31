import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import ServingConfig
from market_data import HistoricalParquetProvider
from predictor import Predictor

TZ = "Asia/Seoul"


@pytest.fixture()
def client(tmp_path, raw_data_dir, tiny_artifact_dir):
    config = ServingConfig(
        artifact_dir=tiny_artifact_dir, data_dir=raw_data_dir,
        symbols=["005930"], audit_log_dir=tmp_path / "logs",
    )
    predictor = Predictor.load(tiny_artifact_dir)
    provider = HistoricalParquetProvider(raw_data_dir, warmup_days=config.warmup_days)
    return TestClient(create_app(config, predictor, provider))


def _payload(as_of, **portfolio_overrides):
    portfolio = {"units_held": 0, "shares_held": 0.0,
                 "bars_since_entry": 0, "available_cash": 10_000.0}
    portfolio.update(portfolio_overrides)
    return {"symbol": "005930", "portfolio": portfolio, "as_of": as_of}


def _valid_as_of(client_unused, minute_data):
    day = pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]
    return f"{day}T11:00:00+09:00"


def test_predict_happy_path(client, minute_data):
    resp = client.post("/predict", json=_payload(_valid_as_of(client, minute_data)))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"] in ("hold", "add_unit", "clear")
    assert len(body["observation"]) == 13
    assert body["action_mask"][0] is True
    assert body["artifact_id"] == "ppo-fs3-test"
    assert body["feature_schema_version"] == 3


def test_predict_writes_audit_log(client, minute_data, tmp_path):
    client.post("/predict", json=_payload(_valid_as_of(client, minute_data)))
    logs = list((tmp_path / "logs").glob("predict-*.jsonl"))
    assert logs, "audit jsonl not written"
    record = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[-1])
    assert record["symbol"] == "005930" and "observation" in record


def test_stale_data_maps_to_503_and_audited(client, minute_data, tmp_path):
    day = pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]
    resp = client.post("/predict", json=_payload(f"{day}T23:00:00+09:00"))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "STALE_DATA"
    logs = list((tmp_path / "logs").glob("predict-*.jsonl"))
    assert logs, "error must be audited too (spec §3)"
    record = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[-1])
    assert record["error"]["code"] == "STALE_DATA"


def test_validation_error_audited(client, minute_data, tmp_path):
    resp = client.post("/predict", json=_payload(
        _valid_as_of(client, minute_data), units_held=0, shares_held=5.0))
    assert resp.status_code == 422
    logs = list((tmp_path / "logs").glob("predict-*.jsonl"))
    assert logs
    record = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[-1])
    assert record["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_symbol_maps_to_422(client, minute_data):
    payload = _payload(_valid_as_of(client, minute_data))
    payload["symbol"] = "999999"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_units_over_artifact_max_maps_to_422(client, minute_data):
    resp = client.post("/predict", json=_payload(
        _valid_as_of(client, minute_data),
        units_held=6, shares_held=1.0, bars_since_entry=1))
    assert resp.status_code == 422


def test_invariant_violation_maps_to_422(client, minute_data):
    resp = client.post("/predict", json=_payload(
        _valid_as_of(client, minute_data), units_held=0, shares_held=5.0))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_metadata_endpoint(client):
    body = client.get("/metadata").json()
    assert body["artifact_id"] == "ppo-fs3-test"
    assert body["feature_schema_version"] == 3
    assert body["env_params"]["max_units"] == 5
    assert body["action_space"]["labels"] == ["hold", "add_unit", "clear"]


def test_health_and_ready(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_ready_fails_without_data(tmp_path, tiny_artifact_dir):
    config = ServingConfig(
        artifact_dir=tiny_artifact_dir, data_dir=tmp_path / "empty",
        symbols=["005930"], audit_log_dir=tmp_path / "logs",
    )
    predictor = Predictor.load(tiny_artifact_dir)
    provider = HistoricalParquetProvider(tmp_path / "empty")
    client = TestClient(create_app(config, predictor, provider))
    assert client.get("/ready").status_code == 503


def test_unexpected_error_maps_to_500_and_audited(
    monkeypatch, tmp_path, raw_data_dir, tiny_artifact_dir, minute_data
):
    config = ServingConfig(
        artifact_dir=tiny_artifact_dir, data_dir=raw_data_dir,
        symbols=["005930"], audit_log_dir=tmp_path / "logs",
    )
    predictor = Predictor.load(tiny_artifact_dir)
    provider = HistoricalParquetProvider(raw_data_dir, warmup_days=config.warmup_days)

    def _boom(symbol, as_of):
        raise RuntimeError("unexpected provider failure")

    monkeypatch.setattr(provider, "get_recent_bars", _boom)
    app = create_app(config, predictor, provider)
    # Starlette's TestClient re-raises unhandled exceptions by default; the
    # fail-closed contract under test is that the *client-visible* response
    # (not just an in-process exception) is a structured 500 body.
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/predict", json=_payload(_valid_as_of(client, minute_data)))
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "SERVING_ERROR"

    logs = list((tmp_path / "logs").glob("predict-*.jsonl"))
    assert logs, "unexpected error must be audited too (spec §3)"
    record = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[-1])
    assert record["error"]["code"] == "SERVING_ERROR"
