from datetime import date, datetime, timezone

import pandas as pd
import pytest

from shadow_runner import (
    apply_until, can_retry, classify_response, fetch_run_context,
    retry_policy, trading_grid,
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
