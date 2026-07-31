"""Shadow runner — 장중 5분 grid마다 /predict 호출, manifest JSONL 기록.

spec §2. 주문 없음·서버 상태 없음. 서버 audit = 결정 데이터의 진실,
이 manifest = 스케줄 실행의 진실 (연결 실패는 서버 audit에 안 남는다).
운영 불변식: 서버와 같은 checkout에서 실행 — 여기서 기록하는 git SHA가
서버 코드 SHA를 대변한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as time_mod
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _src in (_ROOT / "serving" / "src", _ROOT / "agent" / "src", _ROOT / "env" / "src"):
    _p = str(_src)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import httpx
import pandas as pd

KST = "Asia/Seoul"
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 240.0     # KIS fetch 전체 최악 ~208초 + 여유 (spec §2)
WORST_BUDGET_SECONDS = 240
RETRYABLE_WAIT = {"transient": 5, "late_bar": 15}


def trading_grid(day: date) -> list:
    start = pd.Timestamp(f"{day} 09:05", tz=KST)
    return [start + pd.Timedelta(minutes=5 * i) for i in range(76)]


def classify_response(scheduled_bar_ts: str, status_code: int, body: dict):
    """(outcome, error_code, response_bar_ts). body는 성공/에러 응답 JSON."""
    if status_code == 200:
        if "bar_ts" not in body:
            return "error", "PARSE_ERROR", None
        response_bar_ts = str(body["bar_ts"])
        if response_bar_ts == scheduled_bar_ts:
            return "ok", None, response_bar_ts
        return "wrong_bar", None, response_bar_ts
    error_code = (body.get("error") or {}).get("code")
    if error_code == "STALE_DATA":
        return "stale", error_code, None
    return "error", error_code, None


# 재시도는 명시적 allowlist (spec §2) — 밖은 전부 재시도 금지
_TRANSIENT_RETRYABLE = frozenset(
    {"PROVIDER_ERROR", "SERVING_ERROR", "ConnectError", "ConnectTimeout"})
_NO_RETRY_BENIGN = frozenset({"READ_TIMEOUT", "INSUFFICIENT_HISTORY"})
_RUN_ERROR_CODES = frozenset({"VALIDATION_ERROR", "MODEL_ERROR"})


def retry_policy(outcome: str, error_code):
    """(재시도 여부, 대기 초, run 오류 여부).

    - allowlist만 재시도: transient(연결 거부·PROVIDER/SERVING_ERROR) 5초,
      노출 지연(stale·wrong_bar) 15초.
    - read timeout은 서버가 아직 처리 중일 수 있어 같은 grid 재시도 금지
      (audit 중복 방지). INSUFFICIENT_HISTORY는 정상 데이터 조건 — 재시도 무의미.
    - VALIDATION/MODEL_ERROR·알 수 없는 코드·파싱 실패는 run 오류 (계약 위반 신호).
    """
    if outcome == "ok":
        return False, 0, False
    if outcome in ("stale", "wrong_bar"):
        return True, RETRYABLE_WAIT["late_bar"], False
    if error_code in _RUN_ERROR_CODES:
        return False, 0, True
    if error_code in _NO_RETRY_BENIGN:
        return False, 0, False
    if error_code in _TRANSIENT_RETRYABLE:
        return True, RETRYABLE_WAIT["transient"], False
    return False, 0, True


def can_retry(now, next_grid_at, wait_seconds: int, worst_budget_seconds: int) -> bool:
    remaining = (next_grid_at - now).total_seconds()
    return remaining >= wait_seconds + worst_budget_seconds


def apply_until(grid: list, day: date, until) -> list:
    """--until HH:MM — 이 라벨(포함)까지의 grid만."""
    if not until:
        return grid
    limit = pd.Timestamp(f"{day} {until}", tz=KST)
    return [g for g in grid if g <= limit]


def fetch_run_context(client, server: str):
    """/metadata 핸드셰이크 → (artifact_id, flat portfolio) 또는 None(치명 실패).

    non-200·비-dict JSON·필드 누락은 조용한 기본값 없이 run을 중단시킨다.
    """
    try:
        resp = client.get(f"{server}/metadata")
    except httpx.HTTPError as exc:
        print(f"[fatal] /metadata 요청 실패: {exc}")
        return None
    if resp.status_code != 200:
        print(f"[fatal] /metadata HTTP {resp.status_code}")
        return None
    try:
        meta = resp.json()
        if not isinstance(meta, dict):
            raise TypeError(f"dict 아님: {type(meta).__name__}")
        portfolio = {"units_held": 0, "shares_held": 0.0, "bars_since_entry": 0,
                     "available_cash": float(meta["env_params"]["initial_cash"])}
        return meta["artifact_id"], portfolio
    except (ValueError, KeyError, TypeError) as exc:
        print(f"[fatal] /metadata 응답 파싱 실패: {exc}")
        return None


def _iso_kst(ts: pd.Timestamp) -> str:
    return ts.isoformat()


class ManifestWriter:
    def __init__(self, manifest_dir: Path, day: date, run_id: str) -> None:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        self.path = manifest_dir / f"shadow-{day.isoformat()}-{run_id}.jsonl"

    def write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _call_predict(client: httpx.Client, base_url: str, symbol: str, portfolio: dict):
    started = time_mod.monotonic()
    try:
        resp = client.post(f"{base_url}/predict",
                           json={"symbol": symbol, "portfolio": portfolio})
        latency_ms = int((time_mod.monotonic() - started) * 1000)
        try:
            body = resp.json()
        except ValueError:
            return resp.status_code, {}, latency_ms, "PARSE_ERROR"
        if not isinstance(body, dict):
            # list/str JSON도 json() 성공 후 body.get()에서 죽는다 — 파싱 실패 취급
            return resp.status_code, {}, latency_ms, "PARSE_ERROR"
        return resp.status_code, body, latency_ms, None
    except httpx.ReadTimeout:
        return None, {}, int((time_mod.monotonic() - started) * 1000), "READ_TIMEOUT"
    except httpx.HTTPError as exc:
        return None, {}, int((time_mod.monotonic() - started) * 1000), type(exc).__name__


def run(argv=None) -> int:
    from models.artifact import current_git_sha

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--symbol", default="005930")
    parser.add_argument("--delay-seconds", type=int, default=20)
    parser.add_argument("--manifest-dir",
                        default=str(_ROOT / "serving" / "logs" / "shadow"))
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD (기본: 오늘, KST) — 테스트/재실행용")
    parser.add_argument("--until", default=None,
                        help="HH:MM — 이 라벨까지의 grid만 실행 (기본 15:20)")
    args = parser.parse_args(argv)

    day = (date.fromisoformat(args.date) if args.date
           else pd.Timestamp.now(tz=KST).date())
    run_id = pd.Timestamp.now(tz=KST).strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
    git_sha = current_git_sha()

    client = httpx.Client(timeout=httpx.Timeout(
        connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=10.0, pool=10.0))
    context = fetch_run_context(client, args.server)
    if context is None:
        return 2
    artifact_id, portfolio = context

    writer = ManifestWriter(Path(args.manifest_dir), day, run_id)
    writer.write({"kind": "header", "run_id": run_id, "date": day.isoformat(),
                  "symbol": args.symbol, "artifact_id": artifact_id,
                  "portfolio": dict(portfolio),   # diff의 audit 필터 기준 (역추론 금지)
                  "git_sha": git_sha,
                  "config": {"server": args.server,
                             "delay_seconds": args.delay_seconds}})
    if git_sha.endswith("-dirty") or git_sha == "unknown":
        # preflight: untracked 파일도 dirty 판정 — 로컬 전용 파일은 .git/info/exclude로
        print(f"[warn] dirty/unknown checkout ({git_sha}) — 정식 완료 판정 제외")

    grid = apply_until(trading_grid(day), day, args.until)
    counts: dict = {}
    run_error = False
    for i, bar_ts in enumerate(grid):
        scheduled_at = bar_ts + pd.Timedelta(seconds=args.delay_seconds)
        next_grid_at = grid[i + 1] + pd.Timedelta(seconds=args.delay_seconds) \
            if i + 1 < len(grid) else scheduled_at + pd.Timedelta(seconds=300)
        wait = (scheduled_at - pd.Timestamp.now(tz=KST)).total_seconds()
        if wait > 0:
            time_mod.sleep(wait)
        elif wait < -240:
            # 과거 grid (장중 재시작 등)는 건드리지 않는다
            writer.write({"kind": "grid", "scheduled_bar_ts": _iso_kst(bar_ts),
                          "scheduled_at": _iso_kst(scheduled_at), "attempts": [],
                          "response_bar_ts": None, "outcome": "skipped"})
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue

        attempts: list = []
        outcome, error_code, response_bar_ts = "skipped", None, None
        for attempt in (1, 2):
            status, body, latency_ms, transport_error = _call_predict(
                client, args.server, args.symbol, portfolio)
            if transport_error is not None:
                outcome, error_code = "error", transport_error
                http_status = status  # PARSE_ERROR면 status 있음, transport면 None
                response_bar_ts = None
            else:
                http_status = status
                outcome, error_code, response_bar_ts = classify_response(
                    _iso_kst(bar_ts), status, body)
            attempts.append({"attempt": attempt, "http_status": http_status,
                             "error_code": error_code, "latency_ms": latency_ms,
                             "response_bar_ts": response_bar_ts,
                             "outcome": outcome})
            do_retry, wait_s, is_run_error = retry_policy(outcome, error_code)
            run_error = run_error or is_run_error
            if not do_retry or attempt == 2:
                break
            if not can_retry(pd.Timestamp.now(tz=KST), next_grid_at,
                             wait_s, WORST_BUDGET_SECONDS):
                # 재시도 포기 (spec §2): 명시적 응답을 받았으면 그 outcome 보존,
                # 무응답(transport 실패)이면 skipped
                if http_status is None:
                    outcome = "skipped"
                break
            time_mod.sleep(wait_s)
        writer.write({"kind": "grid", "scheduled_bar_ts": _iso_kst(bar_ts),
                      "scheduled_at": _iso_kst(scheduled_at), "attempts": attempts,
                      "response_bar_ts": response_bar_ts, "outcome": outcome})
        counts[outcome] = counts.get(outcome, 0) + 1

    print(f"shadow run {run_id} done: {counts} → {writer.path}")
    return 1 if run_error else 0


if __name__ == "__main__":
    sys.exit(run())
