"""서빙 에러 계약 — spec §3. 결정 불가는 항상 HTTP 에러(fail-closed)."""

from __future__ import annotations


class ServingError(Exception):
    code: str = "SERVING_ERROR"
    http_status: int = 500

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class StaleDataError(ServingError):
    code, http_status = "STALE_DATA", 503


class InsufficientHistoryError(ServingError):
    code, http_status = "INSUFFICIENT_HISTORY", 503


class ProviderError(ServingError):
    code, http_status = "PROVIDER_ERROR", 503


class RequestValidationFailure(ServingError):
    code, http_status = "VALIDATION_ERROR", 422


class ModelError(ServingError):
    code, http_status = "MODEL_ERROR", 500
