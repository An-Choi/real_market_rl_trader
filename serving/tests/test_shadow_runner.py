import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

import httpx

from shadow_runner import (
    ManifestWriter, apply_until, can_retry, classify_response,
    execute_grid_slot, fetch_run_context, retry_policy, trading_grid,
)

TZ = "Asia/Seoul"


def test_trading_grid_is_76_bars():
    grid = trading_grid(date(2026, 7, 21))
    assert len(grid) == 76
    assert grid[0] == pd.Timestamp("2026-07-21 09:05", tz=TZ)
    assert grid[-1] == pd.Timestamp("2026-07-21 15:20", tz=TZ)
    assert (grid[1] - grid[0]) == pd.Timedelta(minutes=5)


def test_classify_ok_requires_bar_ts_match():
    scheduled = "2026-07-21T10:05:00+09:00"
    ok = classify_response(scheduled, 200, {"bar_ts": scheduled})
    assert ok == ("ok", None, scheduled)
    prev = "2026-07-21T10:00:00+09:00"
    wrong = classify_response(scheduled, 200, {"bar_ts": prev})
    assert wrong[0] == "wrong_bar" and wrong[2] == prev


def test_classify_error_codes():
    scheduled = "2026-07-21T10:05:00+09:00"
    stale = classify_response(scheduled, 503, {"error": {"code": "STALE_DATA"}})
    assert stale == ("stale", "STALE_DATA", None)
    err = classify_response(scheduled, 503, {"error": {"code": "PROVIDER_ERROR"}})
    assert err == ("error", "PROVIDER_ERROR", None)


@pytest.mark.parametrize("outcome,code,expected", [
    ("error", "PROVIDER_ERROR", (True, 5, False)),
    ("error", "SERVING_ERROR", (True, 5, False)),
    ("error", "ConnectError", (True, 5, False)),        # 연결 거부 (httpx 예외 클래스명)
    ("error", "ConnectTimeout", (True, 5, False)),      # 연결 반쯤 열린 blip (httpx 예외 클래스명)
    ("stale", "STALE_DATA", (True, 15, False)),
    ("wrong_bar", None, (True, 15, False)),             # 노출 지연 — stale과 동일 취급
    ("error", "VALIDATION_ERROR", (False, 0, True)),    # 계약 위반 → run 오류
    ("error", "MODEL_ERROR", (False, 0, True)),
    ("error", "READ_TIMEOUT", (False, 0, False)),       # 서버 처리 중일 수 있음 — 재시도 금지
    ("error", "INSUFFICIENT_HISTORY", (False, 0, False)),  # 정상 데이터 조건 — 재시도 무의미
    ("error", "PARSE_ERROR", (False, 0, True)),         # 응답 파싱 실패 → run 오류
    ("error", "TOTALLY_UNKNOWN", (False, 0, True)),     # allowlist 밖 → run 오류
    ("error", None, (False, 0, True)),                  # code 없는 에러 → run 오류
    ("ok", None, (False, 0, False)),
])
def test_retry_policy(outcome, code, expected):
    assert retry_policy(outcome, code) == expected


def test_can_retry_requires_worst_case_budget():
    now = datetime(2026, 7, 21, 1, 6, 0, tzinfo=timezone.utc)      # 10:06 KST
    next_grid = datetime(2026, 7, 21, 1, 10, 0, tzinfo=timezone.utc)  # 10:10 KST
    # 남은 시간 4분 < worst budget 240초 + wait 15초 → 재시도 불가
    assert can_retry(now, next_grid, wait_seconds=15, worst_budget_seconds=240) is False
    assert can_retry(now, next_grid, wait_seconds=15, worst_budget_seconds=180) is True


def test_apply_until_limits_grid():
    grid = trading_grid(date(2026, 7, 21))
    limited = apply_until(grid, date(2026, 7, 21), "10:00")
    assert len(limited) == 12
    assert limited[-1] == pd.Timestamp("2026-07-21 10:00", tz=TZ)
    assert apply_until(grid, date(2026, 7, 21), None) == grid


class _StubResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubClient:
    def __init__(self, resp):
        self._resp = resp

    def get(self, url):
        return self._resp


@pytest.mark.parametrize("resp", [
    _StubResp(500, {}),                                # non-200
    _StubResp(200, ValueError("bad json")),            # json 파싱 실패
    _StubResp(200, ["not", "a", "dict"]),              # 비-dict JSON
    _StubResp(200, {"artifact_id": "x"}),              # env_params 누락
])
def test_fetch_run_context_failures_return_none(resp):
    assert fetch_run_context(_StubClient(resp), "http://s") is None


def test_fetch_run_context_success():
    resp = _StubResp(200, {"artifact_id": "ppo-fs3-x",
                           "env_params": {"initial_cash": 10000.0}})
    artifact_id, portfolio = fetch_run_context(_StubClient(resp), "http://s")
    assert artifact_id == "ppo-fs3-x"
    assert portfolio["available_cash"] == 10000.0


# ── execute_grid_slot ────────────────────────────────────────────────────

class _PredictStubClient:
    """duck-type client for _call_predict: .post(url, json=...) -> _StubResp,
    or raises an httpx exception if configured to."""

    def __init__(self, responses=None, raises=None):
        # responses: list of _StubResp, consumed in order per .post() call
        # raises: exception instance (or list of them) to raise instead
        self._responses = list(responses) if responses else None
        self._raises = raises if isinstance(raises, list) else (
            [raises] if raises is not None else None)
        self.calls = 0

    def post(self, url, json=None):
        idx = self.calls
        self.calls += 1
        if self._raises is not None:
            exc = self._raises[idx] if idx < len(self._raises) else self._raises[-1]
            if exc is not None:
                raise exc
        return self._responses[idx]


class _SleepRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


SCHEDULED = "2026-07-21T10:05:00+09:00"
NEXT_GRID_AT = pd.Timestamp("2026-07-21 10:10:00", tz=TZ)


def test_execute_grid_slot_stale_then_ok_retries():
    client = _PredictStubClient(responses=[
        _StubResp(503, {"error": {"code": "STALE_DATA"}}),
        _StubResp(200, {"bar_ts": SCHEDULED}),
    ])
    now_calls = iter([
        pd.Timestamp("2026-07-21 10:05:01", tz=TZ),   # can_retry check
    ])
    sleep_fn = _SleepRecorder()
    attempts, outcome, response_bar_ts, run_error = execute_grid_slot(
        client, "http://s", "005930", {}, SCHEDULED, NEXT_GRID_AT,
        now_fn=lambda: next(now_calls), sleep_fn=sleep_fn,
    )
    assert outcome == "ok"
    assert len(attempts) == 2
    assert attempts[0]["outcome"] == "stale"
    assert response_bar_ts == SCHEDULED
    assert run_error is False
    assert sleep_fn.calls == [15]


def test_execute_grid_slot_insufficient_budget_preserves_explicit_outcome():
    client = _PredictStubClient(responses=[
        _StubResp(503, {"error": {"code": "STALE_DATA"}}),
    ])
    # now is right before next_grid_at minus budget → can_retry False
    now_fn = lambda: NEXT_GRID_AT - pd.Timedelta(seconds=1)
    sleep_fn = _SleepRecorder()
    attempts, outcome, response_bar_ts, run_error = execute_grid_slot(
        client, "http://s", "005930", {}, SCHEDULED, NEXT_GRID_AT,
        now_fn=now_fn, sleep_fn=sleep_fn,
    )
    assert outcome == "stale"
    assert len(attempts) == 1
    assert run_error is False
    assert sleep_fn.calls == []


def test_execute_grid_slot_no_response_and_insufficient_budget_is_skipped():
    client = _PredictStubClient(raises=httpx.ConnectError("refused"))
    now_fn = lambda: NEXT_GRID_AT - pd.Timedelta(seconds=1)
    sleep_fn = _SleepRecorder()
    attempts, outcome, response_bar_ts, run_error = execute_grid_slot(
        client, "http://s", "005930", {}, SCHEDULED, NEXT_GRID_AT,
        now_fn=now_fn, sleep_fn=sleep_fn,
    )
    assert outcome == "skipped"
    assert len(attempts) == 1
    assert run_error is False


def test_execute_grid_slot_read_timeout_never_retries():
    client = _PredictStubClient(raises=httpx.ReadTimeout("timed out"))
    now_fn = lambda: pd.Timestamp("2026-07-21 10:05:01", tz=TZ)
    sleep_fn = _SleepRecorder()
    attempts, outcome, response_bar_ts, run_error = execute_grid_slot(
        client, "http://s", "005930", {}, SCHEDULED, NEXT_GRID_AT,
        now_fn=now_fn, sleep_fn=sleep_fn,
    )
    assert len(attempts) == 1
    assert outcome == "error"
    assert attempts[0]["error_code"] == "READ_TIMEOUT"
    assert run_error is False
    assert sleep_fn.calls == []


def test_execute_grid_slot_validation_error_is_run_error_no_retry():
    client = _PredictStubClient(responses=[
        _StubResp(422, {"error": {"code": "VALIDATION_ERROR"}}),
    ])
    now_fn = lambda: pd.Timestamp("2026-07-21 10:05:01", tz=TZ)
    sleep_fn = _SleepRecorder()
    attempts, outcome, response_bar_ts, run_error = execute_grid_slot(
        client, "http://s", "005930", {}, SCHEDULED, NEXT_GRID_AT,
        now_fn=now_fn, sleep_fn=sleep_fn,
    )
    assert len(attempts) == 1
    assert run_error is True
    assert sleep_fn.calls == []


def test_manifest_writer_round_trip(tmp_path):
    writer = ManifestWriter(tmp_path, date(2026, 7, 21), "100000-abcdef")
    writer.write({"kind": "header", "run_id": "100000-abcdef", "date": "2026-07-21",
                  "symbol": "005930", "artifact_id": "ppo-fs3-x",
                  "portfolio": {"units_held": 0, "shares_held": 0.0,
                                "bars_since_entry": 0, "available_cash": 10000.0},
                  "git_sha": "abc1234", "config": {"server": "http://s"}})
    writer.write({"kind": "grid", "scheduled_bar_ts": SCHEDULED,
                  "scheduled_at": SCHEDULED, "attempts": [], "response_bar_ts": None,
                  "outcome": "ok"})

    lines = [json.loads(line) for line in writer.path.read_text(encoding="utf-8")
             .splitlines() if line.strip()]
    header, grid = lines[0], lines[1]
    for key in ("run_id", "date", "symbol", "artifact_id", "portfolio", "git_sha",
                "config"):
        assert key in header
    for key in ("scheduled_bar_ts", "scheduled_at", "attempts", "response_bar_ts",
                "outcome"):
        assert key in grid
