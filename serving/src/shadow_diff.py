"""익일 shadow diff — spec §3. 2단 게이트(artifact·코드 고정 → coverage) 통과
후에만 observation·mask 값을 재계산·비교한다. 모델 로드 불필요(load_metadata만).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _src in (_ROOT / "serving" / "src", _ROOT / "agent" / "src", _ROOT / "env" / "src"):
    _p = str(_src)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

EXIT_OK = 0
EXIT_VALUE_MISMATCH = 1
EXIT_COVERAGE = 2
EXIT_ARTIFACT = 3
EXIT_INPUT = 4

_KST = "Asia/Seoul"


def _canonical_ts(value) -> str:
    """timestamp 문자열 정규화 — audit은 str(Timestamp)="… 10:05:00+09:00"(공백),
    manifest/grid는 isoformat="…T10:05:00+09:00"이라 원문 대조는 항상 어긋난다."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware: {value!r}")
    return ts.tz_convert(_KST).isoformat()


def _current_sha() -> str:
    from models.artifact import current_git_sha

    return current_git_sha()


@dataclass
class DiffResult:
    exit_code: int
    report: list = field(default_factory=list)

    def fail(self, code: int, message: str) -> "DiffResult":
        self.report.append(message)
        self.exit_code = max(self.exit_code, code) if self.exit_code else code
        return self


def _load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _select_manifest(manifest_dir: Path, day: date, explicit) -> "Path | None":
    if explicit is not None:
        return Path(explicit)
    candidates = sorted(manifest_dir.glob(f"shadow-{day.isoformat()}-*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    return None  # 0개(입력 없음) 또는 복수(--manifest 필요) — 호출자가 EXIT_INPUT


def run_diff(*, day, audit_dir: Path, manifest_dir: Path, artifact_dir: Path,
             data_dir: Path, warmup_days: int, max_bar_age_minutes: int,
             grid: list, warmup_count: int, explicit_manifest) -> DiffResult:
    from data.feature_engineer import FeatureEngineer
    from friction.friction_model import FrictionModel
    from market_data import HistoricalParquetProvider
    from models.artifact import load_metadata
    from observation_builder import build_decision_inputs

    result = DiffResult(exit_code=EXIT_OK)

    manifest_path = _select_manifest(Path(manifest_dir), day, explicit_manifest)
    if manifest_path is None or not manifest_path.is_file():
        return result.fail(EXIT_INPUT,
                           "manifest 없음 또는 복수 — --manifest로 명시 필요")
    manifest = _load_jsonl(manifest_path)
    header = manifest[0]
    if header.get("kind") != "header":
        return result.fail(EXIT_INPUT, f"manifest 첫 줄이 header가 아님: {manifest_path}")
    grid_records = [r for r in manifest if r.get("kind") == "grid"]

    audit_records = []
    for path in sorted(Path(audit_dir).glob("predict-*.jsonl")):
        audit_records.extend(_load_jsonl(path))
    if not audit_records:
        return result.fail(EXIT_INPUT, f"audit record 없음: {audit_dir}")

    # ── 게이트 1: artifact·코드 고정 ──────────────────────────────
    # 필터는 (symbol, 날짜, portfolio)로만 — artifact_id는 필터가 아니라 검사 대상
    # (artifact_id로 먼저 필터하면 혼합을 감지할 수 없다, spec §3).
    # portfolio 기준은 manifest 헤더 — audit에서 역추론 금지 (수동 호출이 먼저
    # 나오면 잘못된 필터가 된다).
    day_str = day.isoformat()
    flat = header.get("portfolio")
    if flat is None:
        return result.fail(EXIT_INPUT, "manifest header에 portfolio 없음")
    candidates = [
        a for a in audit_records
        if "error" not in a
        and a.get("symbol") == header["symbol"]
        and str(a.get("bar_ts", "")).startswith(day_str)
        and a.get("portfolio") == flat
    ]
    if not candidates:
        return result.fail(EXIT_INPUT, "필터 후 남은 audit record 없음")
    artifact_ids = {a["artifact_id"] for a in candidates}
    if artifact_ids != {header["artifact_id"]}:
        return result.fail(EXIT_ARTIFACT,
                           f"artifact 혼합/불일치: audit={sorted(artifact_ids)} "
                           f"manifest={header['artifact_id']}")
    meta = load_metadata(artifact_dir)
    if meta.artifact_id != header["artifact_id"]:
        return result.fail(EXIT_ARTIFACT,
                           f"config artifact({meta.artifact_id}) != "
                           f"manifest({header['artifact_id']}) — artifact 교체됨?")
    sha = _current_sha()

    def _sha_bad(value: str) -> bool:
        return value.endswith("-dirty") or value == "unknown"

    if _sha_bad(header["git_sha"]) or _sha_bad(sha) or sha != header["git_sha"]:
        return result.fail(EXIT_ARTIFACT,
                           f"git SHA 불일치/dirty/unknown: run={header['git_sha']} now={sha}")

    # ── 게이트 2: coverage — 정확 집합 대조 (부분 성공이 통과하면 안 된다) ──
    # 모든 timestamp 키는 _canonical_ts로 정규화 후 대조 (audit ↔ manifest 형식 상이)
    warmup_grid = [_canonical_ts(g) for g in grid[:warmup_count]]
    success_grid = [_canonical_ts(g) for g in grid[warmup_count:]]
    expected = warmup_grid + success_grid

    scheduled = [_canonical_ts(r["scheduled_bar_ts"]) for r in grid_records]
    if len(scheduled) != len(set(scheduled)):
        result.fail(EXIT_COVERAGE, "manifest에 중복 scheduled_bar_ts")
    if set(scheduled) != set(expected):
        result.fail(EXIT_COVERAGE,
                    f"manifest grid 집합 불일치: 누락={sorted(set(expected) - set(scheduled))} "
                    f"예상외={sorted(set(scheduled) - set(expected))}")
    if result.exit_code == EXIT_COVERAGE:
        return result

    by_scheduled = {_canonical_ts(r["scheduled_bar_ts"]): r for r in grid_records}
    for g in warmup_grid:
        rec = by_scheduled[g]
        if rec["outcome"] != "stale":
            result.fail(EXIT_COVERAGE, f"warmup grid {g}: stale 아님 ({rec['outcome']})")
    audit_by_bar: dict = {}
    for a in candidates:
        audit_by_bar.setdefault(_canonical_ts(a["bar_ts"]), []).append(a)
    # 성공 audit의 bar 집합 == 성공 grid 집합 (예상 외 audit bar 0개)
    if set(audit_by_bar) != set(success_grid):
        result.fail(EXIT_COVERAGE,
                    f"success audit 집합 불일치: 누락={sorted(set(success_grid) - set(audit_by_bar))} "
                    f"예상외={sorted(set(audit_by_bar) - set(success_grid))}")
    for g in success_grid:
        rec = by_scheduled[g]
        response_ts = (_canonical_ts(rec["response_bar_ts"])
                       if rec.get("response_bar_ts") else None)
        if rec["outcome"] != "ok" or response_ts != g:
            result.fail(EXIT_COVERAGE, f"success grid {g}: ok/bar_ts 불일치 ({rec['outcome']})")
            continue
        if len(audit_by_bar.get(g, [])) > 1:
            result.fail(EXIT_COVERAGE, f"success grid {g}: 성공 audit 중복 "
                                       f"({len(audit_by_bar[g])}개)")
    if result.exit_code == EXIT_COVERAGE:
        return result

    # ── 값 비교 ───────────────────────────────────────────────────
    provider = HistoricalParquetProvider(Path(data_dir), warmup_days=warmup_days)
    fe = FeatureEngineer()
    friction = FrictionModel(**meta.friction_params)
    for g in success_grid:
        record = audit_by_bar[g][0]
        bar_ts = pd.Timestamp(g)
        p = record["portfolio"]
        recomputed = build_decision_inputs(
            bars_1m=provider.get_recent_bars(header["symbol"], bar_ts),
            as_of=bar_ts,
            units_held=p["units_held"], shares_held=p["shares_held"],
            bars_since_entry=p["bars_since_entry"],
            available_cash=p["available_cash"],
            env_params=meta.env_params, friction_model=friction,
            max_bar_age=pd.Timedelta(minutes=max_bar_age_minutes),
            feature_engineer=fe,
        )
        live_obs = record["observation"]
        rec_obs = [float(x) for x in recomputed.observation]
        # zip은 짧은 쪽까지만 돌므로 길이 검사가 선행되어야 한다 (13차원 계약)
        if len(live_obs) != len(rec_obs) or len(rec_obs) != 13:
            result.fail(EXIT_VALUE_MISMATCH,
                        f"{g} obs 길이 불일치: live={len(live_obs)} "
                        f"recomputed={len(rec_obs)} (계약 13)")
            continue
        for idx, (a, b) in enumerate(zip(live_obs, rec_obs)):
            if a != b:
                result.fail(EXIT_VALUE_MISMATCH,
                            f"{g} obs[{idx}]: live={a!r} recomputed={b!r}")
        live_mask = record["action_mask"]
        rec_mask = [bool(x) for x in recomputed.action_mask]
        if live_mask != rec_mask:
            result.fail(EXIT_VALUE_MISMATCH,
                        f"{g} mask: live={live_mask} recomputed={rec_mask}")
    if result.exit_code == EXIT_OK:
        result.report.append(
            f"OK: {len(success_grid)} bars 완전 일치 (artifact {meta.artifact_id})")
    return result


def main(argv=None) -> int:
    from config import load_serving_config
    from shadow_runner import trading_grid

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)

    config = load_serving_config(args.config)
    day = date.fromisoformat(args.date)
    result = run_diff(
        day=day, audit_dir=Path(args.audit_dir),
        manifest_dir=Path(args.manifest_dir),
        artifact_dir=config.artifact_dir, data_dir=config.data_dir,
        warmup_days=config.warmup_days,
        max_bar_age_minutes=config.max_bar_age_minutes,
        grid=trading_grid(day), warmup_count=12,
        explicit_manifest=args.manifest,
    )
    for line in result.report:
        print(line)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
