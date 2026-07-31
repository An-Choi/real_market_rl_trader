"""FastAPI predict 서버 — spec §2·§3. 서버는 stateless."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import ServingConfig
from data.feature_engineer import FeatureEngineer
from errors import ModelError, RequestValidationFailure, ServingError
from models.artifact import EXPECTED_ACTION_LABELS
from observation_builder import build_decision_inputs
from schemas import PredictRequest, PredictResponse

KST = "Asia/Seoul"


def _error_response(code: str, status: int, message: str, detail: dict) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "detail": detail}},
    )


def _audit(config: ServingConfig, record: dict) -> None:
    config.audit_log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = config.audit_log_dir / f"predict-{stamp}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def create_app(config: ServingConfig, predictor, provider) -> FastAPI:
    app = FastAPI(title="rl-trader predict server")
    feature_engineer = FeatureEngineer()
    max_bar_age = pd.Timedelta(minutes=config.max_bar_age_minutes)

    async def _request_body_json(request: Request) -> dict | None:
        # best-effort: 감사 목적이므로 파싱 실패 시 조용히 생략한다.
        try:
            raw = await request.body()
            return json.loads(raw)
        except Exception:
            return None

    @app.exception_handler(ServingError)
    async def _serving_error(request: Request, exc: ServingError) -> JSONResponse:
        detail = jsonable_encoder(exc.detail)
        if request.url.path == "/predict":
            # spec §3: 에러도 audit — "모든 predict 요청/응답(+에러)"
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "path": "/predict",
                "error": {"code": exc.code, "message": exc.message, "detail": detail},
            }
            body = await _request_body_json(request)
            if body is not None:
                record["request"] = body
            _audit(config, record)
        return _error_response(exc.code, exc.http_status, exc.message, detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # exc.errors()의 ctx에 ValueError 객체가 섞일 수 있어 jsonable_encoder 필수
        errors = jsonable_encoder(exc.errors())
        if request.url.path == "/predict":
            _audit(config, {
                "ts": datetime.now(timezone.utc).isoformat(),
                "path": "/predict",
                "error": {"code": "VALIDATION_ERROR", "errors": errors},
                "request": jsonable_encoder(exc.body),
            })
        return _error_response("VALIDATION_ERROR", 422, "invalid request",
                               {"errors": errors})

    @app.exception_handler(Exception)
    async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # fail-closed catch-all: ServingError/RequestValidationError로 분류되지 않은
        # 모든 예외도 구조화된 에러 바디 + audit 계약을 지켜야 한다 (spec §3).
        detail = jsonable_encoder({"exception_type": type(exc).__name__})
        if request.url.path == "/predict":
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "path": "/predict",
                "error": {"code": "SERVING_ERROR", "message": "unexpected server error",
                          "detail": detail},
            }
            body = await _request_body_json(request)
            if body is not None:
                record["request"] = body
            _audit(config, record)
        return _error_response(
            "SERVING_ERROR", 500, "unexpected server error", detail)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict:
        for symbol in config.symbols:
            provider.check_ready(symbol)  # 실패 시 ProviderError → 503
        return {"status": "ready", "artifact_id": predictor.meta.artifact_id}

    @app.get("/metadata")
    async def metadata() -> dict:
        meta = predictor.meta
        return {
            "artifact_id": meta.artifact_id,
            "feature_schema_version": meta.feature_schema_version,
            "action_space": meta.action_space,
            "env_params": meta.env_params,
            "train_git_sha": meta.train_git_sha,
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        if request.symbol not in config.symbols:
            raise RequestValidationFailure(
                f"symbol not served: {request.symbol}", {"symbols": config.symbols})
        max_units = int(predictor.env_params["max_units"])
        if request.portfolio.units_held > max_units:
            raise RequestValidationFailure(
                f"units_held {request.portfolio.units_held} > artifact max_units {max_units}")
        if request.as_of is not None:
            as_of = pd.Timestamp(request.as_of)
            if as_of.tzinfo is None:
                raise RequestValidationFailure("as_of must be timezone-aware")
            as_of = as_of.tz_convert(KST)
        else:
            as_of = pd.Timestamp.now(tz=KST)

        bars = provider.get_recent_bars(request.symbol, as_of)
        result = build_decision_inputs(
            bars_1m=bars,
            as_of=as_of,
            units_held=request.portfolio.units_held,
            shares_held=request.portfolio.shares_held,
            bars_since_entry=request.portfolio.bars_since_entry,
            available_cash=request.portfolio.available_cash,
            env_params=predictor.env_params,
            friction_model=predictor.friction_model,
            max_bar_age=max_bar_age,
            feature_engineer=feature_engineer,
        )
        try:
            action = predictor.predict(result.observation, result.action_mask)
        except Exception as exc:  # inference 내부 오류만 MODEL_ERROR로
            raise ModelError(f"inference failed: {exc}") from exc

        if not 0 <= action < len(result.action_mask):
            raise ModelError(
                f"model returned out-of-range action {action}",
                {"action": action, "n_actions": len(result.action_mask)},
            )
        if not bool(result.action_mask[action]):
            raise ModelError(
                f"model returned masked action {action}",
                {"action": action, "action_mask": [bool(x) for x in result.action_mask]},
            )

        response = PredictResponse(
            action=action,
            label=EXPECTED_ACTION_LABELS[action],
            action_mask=[bool(x) for x in result.action_mask],
            bar_ts=result.bar_ts.to_pydatetime(),
            artifact_id=predictor.meta.artifact_id,
            feature_schema_version=predictor.meta.feature_schema_version,
            observation=[float(x) for x in result.observation],
        )
        _audit(config, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "path": "/predict",
            "as_of": str(as_of), "symbol": request.symbol,
            "portfolio": request.portfolio.model_dump(),
            "bar_ts": str(result.bar_ts), "action": action,
            "label": response.label, "action_mask": response.action_mask,
            "observation": response.observation,
            "artifact_id": predictor.meta.artifact_id,
        })
        return response

    return app
