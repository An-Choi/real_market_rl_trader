import json
from pathlib import Path

import pandas as pd
import pytest

import yaml

from data.feature_engineer import FeatureEngineer
from friction.friction_model import FrictionModel
from models.artifact import load_metadata
from observation_builder import build_decision_inputs
from market_data import HistoricalParquetProvider
from shadow_diff import (
    EXIT_ARTIFACT, EXIT_COVERAGE, EXIT_INPUT, EXIT_OK, EXIT_VALUE_MISMATCH,
    main, run_diff,
)
from shadow_runner import trading_grid

TZ = "Asia/Seoul"
FLAT = {"units_held": 0, "shares_held": 0.0, "bars_since_entry": 0,
        "available_cash": 10_000.0}


def _recompute_observation(raw_data_dir, meta, bar_ts):
    provider = HistoricalParquetProvider(raw_data_dir, warmup_days=30)
    result = build_decision_inputs(
        bars_1m=provider.get_recent_bars("005930", bar_ts),
        as_of=bar_ts,
        units_held=0, shares_held=0.0, bars_since_entry=0,
        available_cash=10_000.0,
        env_params=meta.env_params,
        friction_model=FrictionModel(**meta.friction_params),
        max_bar_age=pd.Timedelta(minutes=10),
        feature_engineer=FeatureEngineer(),
    )
    return result


def _fixture_day(minute_data):
    # 합성 데이터의 마지막 날을 "shadow 운영일"로 가정한다.
    return pd.to_datetime(minute_data["Timestamp"]).dt.date.iloc[-1]


@pytest.fixture()
def shadow_fixture(tmp_path, raw_data_dir, minute_data, tiny_artifact_dir, monkeypatch):
    """정합한 manifest+audit 세트 생성. 반환: (audit_dir, manifest_dir, config_stub, day)

    합성 데이터의 featured grid는 실제 76-grid와 다르므로, 테스트는
    trading_grid를 '그 날 실제 featured bar들'로 monkeypatch하지 않고 —
    diff 도구가 grid를 인자로 받는 순수 함수(run_diff(..., grid=))라는 점을
    이용해 축소 grid를 주입한다: warmup 2개 + success 4개.
    """
    meta = load_metadata(tiny_artifact_dir)
    day = _fixture_day(minute_data)
    featured = FeatureEngineer().transform(minute_data)
    day_bars = featured[pd.to_datetime(featured["Timestamp"]).dt.date == day]
    success_grid = [pd.Timestamp(t) for t in day_bars["Timestamp"].iloc[2:6]]
    warmup_grid = [success_grid[0] - pd.Timedelta(minutes=10),
                   success_grid[0] - pd.Timedelta(minutes=5)]
    grid = warmup_grid + success_grid

    audit_dir = tmp_path / "audit"; audit_dir.mkdir()
    manifest_dir = tmp_path / "manifest"; manifest_dir.mkdir()
    git_sha = "abc1234"

    audit_path = audit_dir / "predict-20260616.jsonl"
    with audit_path.open("w", encoding="utf-8") as fh:
        for bar_ts in success_grid:
            result = _recompute_observation(raw_data_dir, meta, bar_ts)
            fh.write(json.dumps({
                "ts": "2026-06-16T01:00:00+00:00", "path": "/predict",
                "as_of": str(bar_ts), "symbol": "005930", "portfolio": dict(FLAT),
                "bar_ts": str(result.bar_ts), "action": 0, "label": "hold",
                "action_mask": [bool(x) for x in result.action_mask],
                "observation": [float(x) for x in result.observation],
                "artifact_id": meta.artifact_id,
            }) + "\n")

    manifest_path = manifest_dir / f"shadow-{day.isoformat()}-120000-abcdef.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "header", "run_id": "120000-abcdef",
                             "date": day.isoformat(), "symbol": "005930",
                             "artifact_id": meta.artifact_id,
                             "portfolio": dict(FLAT),   # diff 필터 기준
                             "git_sha": git_sha, "config": {}}) + "\n")
        for bar_ts in warmup_grid:
            fh.write(json.dumps({"kind": "grid", "scheduled_bar_ts": bar_ts.isoformat(),
                                 "scheduled_at": bar_ts.isoformat(),
                                 "attempts": [{"attempt": 1, "http_status": 503,
                                               "error_code": "STALE_DATA",
                                               "latency_ms": 10,
                                               "response_bar_ts": None,
                                               "outcome": "stale"}],
                                 "response_bar_ts": None,
                                 "outcome": "stale"}) + "\n")
        for bar_ts in success_grid:
            fh.write(json.dumps({"kind": "grid", "scheduled_bar_ts": bar_ts.isoformat(),
                                 "scheduled_at": bar_ts.isoformat(),
                                 "attempts": [{"attempt": 1, "http_status": 200,
                                               "error_code": None, "latency_ms": 10,
                                               "response_bar_ts": bar_ts.isoformat(),
                                               "outcome": "ok"}],
                                 "response_bar_ts": bar_ts.isoformat(),
                                 "outcome": "ok"}) + "\n")

    monkeypatch.setattr("shadow_diff._current_sha", lambda: git_sha)
    return dict(audit_dir=audit_dir, manifest_dir=manifest_dir, day=day,
                grid=grid, warmup_count=2, artifact_dir=tiny_artifact_dir,
                data_dir=raw_data_dir)


def _run(fx, **overrides):
    kwargs = dict(
        day=fx["day"], audit_dir=fx["audit_dir"], manifest_dir=fx["manifest_dir"],
        artifact_dir=fx["artifact_dir"], data_dir=fx["data_dir"], warmup_days=30,
        max_bar_age_minutes=10, grid=fx["grid"], warmup_count=fx["warmup_count"],
        explicit_manifest=None,
    )
    kwargs.update(overrides)
    return run_diff(**kwargs)


def test_all_pass_exit_0(shadow_fixture):
    assert _run(shadow_fixture).exit_code == EXIT_OK


def test_value_mismatch_exit_1(shadow_fixture):
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1]); rec["observation"][0] += 1.0
    lines[-1] = json.dumps(rec)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_VALUE_MISMATCH


def test_missing_bar_coverage_exit_2(shadow_fixture):
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    audit_file.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    # manifest도 마지막 grid를 error로 바꿔야 일관되지만 — audit 누락만으로도 실패해야 한다
    assert _run(shadow_fixture).exit_code == EXIT_COVERAGE


def test_duplicate_success_bar_exit_2(shadow_fixture):
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    audit_file.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_COVERAGE


def test_mid_run_artifact_swap_detected_exit_3(shadow_fixture):
    # 장중 artifact 교체: 뒤쪽 절반의 audit artifact_id가 다르다 — 필터 순서 회귀 방지
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    for i in range(2, len(lines)):
        rec = json.loads(lines[i]); rec["artifact_id"] = "ppo-fs3-swapped"
        lines[i] = json.dumps(rec)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_ARTIFACT


def test_sha_mismatch_exit_3(shadow_fixture, monkeypatch):
    monkeypatch.setattr("shadow_diff._current_sha", lambda: "fff9999")
    assert _run(shadow_fixture).exit_code == EXIT_ARTIFACT


def test_dirty_sha_exit_3(shadow_fixture, monkeypatch):
    monkeypatch.setattr("shadow_diff._current_sha", lambda: "abc1234-dirty")
    assert _run(shadow_fixture).exit_code == EXIT_ARTIFACT


def test_manual_calls_filtered_out(shadow_fixture):
    # 다른 portfolio의 수동 호출이 섞여도 (symbol,날짜,portfolio) 필터로 배제 — 통과 유지
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    with audit_file.open("a", encoding="utf-8") as fh:
        rec = json.loads(audit_file.read_text(encoding="utf-8").splitlines()[-1])
        rec["portfolio"] = dict(rec["portfolio"], units_held=2, shares_held=1.0,
                                bars_since_entry=5)
        fh.write(json.dumps(rec) + "\n")
    assert _run(shadow_fixture).exit_code == EXIT_OK


def test_short_observation_exit_1(shadow_fixture):
    # zip 절단 방지: live obs가 12개(앞부분 동일)여도 반드시 실패해야 한다
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1]); rec["observation"] = rec["observation"][:12]
    lines[-1] = json.dumps(rec)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_VALUE_MISMATCH


def test_long_observation_exit_1(shadow_fixture):
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1]); rec["observation"] = rec["observation"] + [0.0]
    lines[-1] = json.dumps(rec)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_VALUE_MISMATCH


def test_mask_mismatch_exit_1(shadow_fixture):
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1]); rec["action_mask"][1] = not rec["action_mask"][1]
    lines[-1] = json.dumps(rec)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_VALUE_MISMATCH


def test_wrong_bar_in_manifest_exit_2(shadow_fixture):
    manifest_file = next(shadow_fixture["manifest_dir"].glob("*.jsonl"))
    lines = manifest_file.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1])
    rec["outcome"] = "wrong_bar"
    rec["response_bar_ts"] = json.loads(lines[-2])["scheduled_bar_ts"]
    lines[-1] = json.dumps(rec)
    manifest_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_COVERAGE


def test_duplicate_manifest_grid_exit_2(shadow_fixture):
    manifest_file = next(shadow_fixture["manifest_dir"].glob("*.jsonl"))
    lines = manifest_file.read_text(encoding="utf-8").splitlines()
    manifest_file.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_COVERAGE


def test_unexpected_audit_bar_exit_2(shadow_fixture):
    # grid에 없는 bar_ts의 성공 audit (같은 portfolio) → 예상 외 bar로 실패
    audit_file = next(shadow_fixture["audit_dir"].glob("*.jsonl"))
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1])
    rec["bar_ts"] = str(shadow_fixture["grid"][0])   # warmup bar — 성공 grid 밖
    audit_file.write_text("\n".join(lines + [json.dumps(rec)]) + "\n",
                          encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_COVERAGE


def test_sha_unknown_exit_3(shadow_fixture, monkeypatch):
    monkeypatch.setattr("shadow_diff._current_sha", lambda: "unknown")
    assert _run(shadow_fixture).exit_code == EXIT_ARTIFACT


def test_multiple_manifests_require_explicit_exit_4(shadow_fixture):
    src = next(shadow_fixture["manifest_dir"].glob("*.jsonl"))
    (shadow_fixture["manifest_dir"] / src.name.replace("120000", "130000")) \
        .write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    assert _run(shadow_fixture).exit_code == EXIT_INPUT


def test_no_inputs_exit_4(shadow_fixture, tmp_path):
    empty = tmp_path / "none"; empty.mkdir()
    assert _run(shadow_fixture, manifest_dir=empty).exit_code == EXIT_INPUT


def _write_serving_config(path, *, artifact_dir, data_dir, audit_log_dir, warmup_days):
    path.write_text(yaml.safe_dump({
        "artifact_dir": str(artifact_dir), "data_dir": str(data_dir),
        "symbols": ["005930"], "audit_log_dir": str(audit_log_dir),
        "warmup_days": warmup_days, "max_bar_age_minutes": 10,
    }), encoding="utf-8")


@pytest.mark.parametrize("exc", [
    __import__("errors").StaleDataError("no completed minute bars", {}),
    __import__("errors").ProviderError("no 1m parquet files", {}),
    KeyError("bar_ts"),
    FileNotFoundError("manifest missing"),
])
def test_main_recompute_crash_is_reported_as_exit_input(
        shadow_fixture, tmp_path, monkeypatch, capsys, exc):
    """backfill 누락(recompute가 당일 데이터 없는 raw_data_dir을 봄) → StaleDataError 등.

    main()에 가드가 없으면 이 예외가 traceback과 함께 그대로 튀어나가 프로세스가
    unhandled exception으로 죽는다 — 원인(backfill 누락 등 입력 문제)이 "값
    불일치"(exit 1)로 오인되거나 traceback만 남는다. main()은 이를 잡아
    EXIT_INPUT(4)로 매핑하고 원인 메시지를 출력해야 한다.
    """
    day = shadow_fixture["day"]
    config_path = tmp_path / "serving.yaml"
    _write_serving_config(
        config_path, artifact_dir=shadow_fixture["artifact_dir"],
        data_dir=shadow_fixture["data_dir"], audit_log_dir=tmp_path / "audit_log",
        warmup_days=30,
    )

    def _raise(*args, **kwargs):
        raise exc
    monkeypatch.setattr("shadow_diff.run_diff", _raise)

    exit_code = main([
        "--date", day.isoformat(),
        "--audit-dir", str(shadow_fixture["audit_dir"]),
        "--manifest-dir", str(shadow_fixture["manifest_dir"]),
        "--config", str(config_path),
    ])
    assert exit_code == EXIT_INPUT
    out = capsys.readouterr().out
    assert str(exc) in out or type(exc).__name__ in out


def test_main_recompute_real_stale_data_error_is_exit_input(
        shadow_fixture, tmp_path, capsys):
    """backfill 누락(recompute가 당일 데이터 없는 data_dir을 봄)에서 실제로 발생하는
    StaleDataError가 run_diff에서 잡히지 않고 그대로 전파됨을 먼저 확인한다 —
    main()의 가드가 실제로 필요한 시나리오임을 증명 (동반 회귀 방지)."""
    day = shadow_fixture["day"]
    data_dir = Path(shadow_fixture["data_dir"])
    missing_backfill_dir = tmp_path / "missing_backfill"
    for symbol_dir in data_dir.iterdir():
        if not symbol_dir.is_dir():
            continue
        for f in (symbol_dir / "1m").glob("*.parquet"):
            df = pd.read_parquet(f)
            ts = pd.to_datetime(df["Timestamp"])
            trimmed = df[ts.dt.date < day]  # 당일 backfill 누락 재현
            out = missing_backfill_dir / symbol_dir.name / "1m"
            out.mkdir(parents=True, exist_ok=True)
            if trimmed.empty:
                continue
            trimmed.reset_index(drop=True).to_parquet(out / f.name)

    from errors import StaleDataError
    with pytest.raises(StaleDataError):
        run_diff(
            day=day, audit_dir=shadow_fixture["audit_dir"],
            manifest_dir=shadow_fixture["manifest_dir"],
            artifact_dir=shadow_fixture["artifact_dir"], data_dir=missing_backfill_dir,
            warmup_days=30, max_bar_age_minutes=10,
            grid=shadow_fixture["grid"], warmup_count=shadow_fixture["warmup_count"],
            explicit_manifest=None,
        )
