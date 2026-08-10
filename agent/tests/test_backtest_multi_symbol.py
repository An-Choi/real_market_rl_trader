"""backtest.py 다종목 헬퍼 단위 테스트."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.backtest import (
    ensure_oos_artifact,
    resolve_backtest_symbols,
    resolve_boundaries,
    run_artifact_backtests,
)
from models.walk_forward import SplitBoundaries


def _write_synthetic_minute_parquet(root: Path, symbol: str, days: int = 60) -> None:
    """연속 `days` 거래일(주말 제외) 분봉 데이터를 하나의 parquet에 기록한다.

    Task 11의 purge_days=5 설정에서도 val/test 구간이 살아남도록 최소 60일 기준
    (test_backtest_entrypoint_smoke.py와 동일 헬퍼 — 코드 복제, 두 파일 모두
    독립적으로 pytest에 수집되므로 패키지 import 대신 복사).
    """
    all_days = pd.bdate_range("2025-06-02", periods=days)
    rng = np.random.default_rng(11)
    frames = []
    price = 100.0

    for day in all_days:
        ts = pd.date_range(f"{day.date()} 09:00", periods=390, freq="1min", tz="Asia/Seoul")
        closes = price + np.cumsum(rng.normal(0, 0.15, 390))
        price = float(closes[-1])
        frames.append(pd.DataFrame({
            "Timestamp": ts,
            "Open": closes,
            "High": closes + 0.4,
            "Low": closes - 0.4,
            "Close": closes,
            "Volume": rng.integers(500, 5000, 390),
            "TradingValue": np.cumsum(rng.integers(1_000_000, 9_000_000, 390)),
        }))

    out_dir = root / symbol / "1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        out_dir / "synthetic.parquet", engine="pyarrow"
    )


class _Meta:
    """ArtifactMetadata 대역 (필요 필드만)."""

    def __init__(self, version: int, train_data: dict):
        self.artifact_format_version = version
        self.train_data = train_data
        self.artifact_id = "test-artifact"


V4_TRAIN_DATA = {
    "symbols": ["AAA", "BBB"],
    "trained_split": "train",
    "split_boundaries": {
        "train_end_date": "2026-01-14", "validation_end_date": "2026-01-17", "purge_days": 2,
    },
}


def test_cli_symbols_override_artifact_metadata():
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    config = {"data": {"symbols": ["CCC"]}}
    out = resolve_backtest_symbols(config=config, meta=meta, cli_symbol=None, cli_symbols="DDD")
    assert out == ["DDD"]


def test_artifact_symbols_beat_config_default():
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    config = {"data": {"symbols": ["CCC", "DDD", "EEE"]}}
    out = resolve_backtest_symbols(config=config, meta=meta, cli_symbol=None, cli_symbols=None)
    assert out == ["AAA", "BBB"]


def test_single_symbol_v3_artifact_defaults_to_its_own_symbol():
    meta = _Meta(3, {"symbols": ["005930"], "start": "2025-05-23", "end": "2026-03-12"})
    config = {"data": {"symbols": ["005930", "000660", "034220", "066570", "009150"]}}
    out = resolve_backtest_symbols(config=config, meta=meta, cli_symbol=None, cli_symbols=None)
    assert out == ["005930"]


def test_no_artifact_falls_back_to_config():
    config = {"data": {"symbols": ["AAA", "BBB"]}}
    out = resolve_backtest_symbols(config=config, meta=None, cli_symbol=None, cli_symbols=None)
    assert out == ["AAA", "BBB"]


def test_v4_boundaries_come_from_metadata_not_recomputed():
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    boundaries = resolve_boundaries(meta=meta, data_by_symbol={}, config={"data": {}})
    assert boundaries == SplitBoundaries.from_metadata(V4_TRAIN_DATA["split_boundaries"])


def test_v3_artifact_gets_none_for_ratio_fallback():
    meta = _Meta(3, {"symbols": ["005930"]})
    assert resolve_boundaries(meta=meta, data_by_symbol={}, config={"data": {}}) is None


def _frame(start: str, days: int) -> pd.DataFrame:
    ts = [pd.Timestamp(start) + pd.Timedelta(days=d) for d in range(days)]
    return pd.DataFrame({"Timestamp": ts, "Close": [100.0] * days})


def test_baseline_only_multi_symbol_computes_shared_boundaries():
    data = {"AAA": _frame("2026-01-01", 20), "BBB": _frame("2026-01-05", 16)}
    config = {"data": {"split": {"purge_days": 1}}}
    boundaries = resolve_boundaries(meta=None, data_by_symbol=data, config=config)
    assert boundaries is not None
    assert boundaries.purge_days == 1


def test_baseline_only_single_symbol_keeps_legacy_ratio_path():
    data = {"AAA": _frame("2026-01-01", 20)}
    assert resolve_boundaries(meta=None, data_by_symbol=data, config={"data": {}}) is None


def test_non_train_artifact_rejected():
    train_data = dict(V4_TRAIN_DATA)
    train_data["trained_split"] = "all"
    with pytest.raises(SystemExit):
        ensure_oos_artifact(_Meta(4, train_data))


def test_train_artifact_accepted_and_v3_passes():
    ensure_oos_artifact(_Meta(4, dict(V4_TRAIN_DATA)))
    ensure_oos_artifact(_Meta(3, {"symbols": ["005930"]}))


def test_artifact_loaded_exactly_once_for_multi_symbol(monkeypatch):
    calls = {"load": 0, "compat": 0, "run": 0}
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    fake_agent = object()

    monkeypatch.setattr(
        "experiments.backtest.load_artifact",
        lambda path, env=None, **kw: (calls.__setitem__("load", calls["load"] + 1) or (fake_agent, meta)),
    )
    monkeypatch.setattr(
        "experiments.backtest.build_backtest_environment",
        lambda featured_data, config: object(),
    )
    monkeypatch.setattr(
        "experiments.backtest.check_env_compatibility",
        lambda m, e: calls.__setitem__("compat", calls["compat"] + 1),
    )
    monkeypatch.setattr(
        "experiments.backtest.run_agent_backtest",
        lambda **kw: (calls.__setitem__("run", calls["run"] + 1) or {"agent": "x", "metrics": {}}),
    )

    out = run_artifact_backtests(
        artifact_path=Path("dummy"),
        meta=meta,
        data_by_symbol={"AAA": _frame("2026-01-01", 5), "BBB": _frame("2026-01-01", 5)},
        config={},
        boundaries=None,
        split="test",
        max_steps=None,
        seed=1,
    )
    assert calls["load"] == 1          # 모델 로드는 정확히 1회
    assert calls["compat"] == 1        # 두 번째 종목 env만 명시 검증 (첫 env는 load_artifact가 검증)
    assert calls["run"] == 2
    assert set(out) == {"AAA", "BBB"}


def _patch_backtest_internals(monkeypatch, calls, meta):
    fake_agent = object()
    monkeypatch.setattr(
        "experiments.backtest.load_artifact",
        lambda path, env=None, **kw: (calls.__setitem__("load", calls["load"] + 1) or (fake_agent, meta)),
    )
    monkeypatch.setattr(
        "experiments.backtest.build_backtest_environment",
        lambda featured_data, config: object(),
    )
    monkeypatch.setattr(
        "experiments.backtest.check_env_compatibility", lambda m, e: None,
    )
    monkeypatch.setattr(
        "experiments.backtest.run_agent_backtest",
        lambda **kw: (calls.__setitem__("run", calls["run"] + 1) or {"agent": "x", "metrics": {"total_return": 0.0}}),
    )


def test_artifact_loaded_once_with_multi_seed(monkeypatch):
    calls = {"load": 0, "run": 0}
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    _patch_backtest_internals(monkeypatch, calls, meta)

    out = run_artifact_backtests(
        artifact_path=Path("dummy"), meta=meta,
        data_by_symbol={"AAA": _frame("2026-01-01", 5), "BBB": _frame("2026-01-01", 5)},
        config={}, boundaries=None, split="test", max_steps=None,
        seed=1, seeds=[1, 2, 3],
    )
    assert calls["load"] == 1                      # multi-seed도 로드는 1회
    assert calls["run"] == 6                       # 2종목 × 3seed
    assert set(out["AAA"]) >= {"agent", "runs", "mean_metrics", "std_metrics"}


def test_compare_loads_artifact_once(monkeypatch):
    calls = {"load": 0, "run": 0}
    meta = _Meta(4, dict(V4_TRAIN_DATA))
    _patch_backtest_internals(monkeypatch, calls, meta)
    monkeypatch.setattr(
        "experiments.backtest.compare_baselines",
        lambda **kw: [{"agent": "buy_and_hold", "metrics": {"total_return": 0.0}}],
    )

    from experiments.backtest import run_compare_backtests
    out = run_compare_backtests(
        artifact_path=Path("dummy"), meta=meta,
        data_by_symbol={"AAA": _frame("2026-01-01", 5), "BBB": _frame("2026-01-01", 5)},
        config={}, boundaries=None, split="test", max_steps=None, seed=1,
    )
    assert calls["load"] == 1                      # compare 모드도 로드는 1회
    for summaries in out.values():
        assert summaries[-1]["agent"] == "x"       # artifact 요약이 append됨


def test_explicit_output_dir_created_and_file_rejected(tmp_path):
    from experiments.backtest import resolve_output_dir

    target = tmp_path / "a" / "b"                  # 존재하지 않는 중첩 경로
    out = resolve_output_dir(target, base=tmp_path, run_id="rid")
    assert out == target and target.is_dir()

    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x")
    with pytest.raises(SystemExit):                # 일반 파일 경로는 거부
        resolve_output_dir(file_path, base=tmp_path, run_id="rid")


def test_auto_output_dir_suffix_on_collision(tmp_path, monkeypatch):
    from experiments.backtest import resolve_output_dir

    class _FixedDatetime:
        @staticmethod
        def now(tz=None):
            import datetime as _dt
            return _dt.datetime(2026, 8, 8, 9, 30, 15, tzinfo=tz)

    monkeypatch.setattr("experiments.backtest.datetime", _FixedDatetime)
    first = resolve_output_dir(None, base=tmp_path, run_id="rid")
    second = resolve_output_dir(None, base=tmp_path, run_id="rid")
    assert first.name == "rid-20260808-093015"
    assert second.name == "rid-20260808-093015-2"  # 같은 초 재실행 → suffix 증가


def test_baseline_multi_symbol_backtest_smoke(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_synthetic_minute_parquet(raw_dir, "005930")
    _write_synthetic_minute_parquet(raw_dir, "000660")
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)                    # 기존 디렉터리 재사용 시나리오

    proc = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "experiments" / "backtest.py"),
            "--baseline", "random",
            "--symbols", "005930,000660",
            "--max-steps", "5",
            "--raw-dir", str(raw_dir),
            "--processed-dir", str(processed_dir),
            "--output-dir", str(out_dir),
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout)          # stdout은 JSON 1개
    assert set(payload["per_symbol"]) == {"005930", "000660"}
    assert (out_dir / "005930.json").is_file()
    assert (out_dir / "000660.json").is_file()
    assert "005930" in proc.stderr             # 요약 테이블은 stderr
    assert "reusing existing output dir" in proc.stderr  # 기존 디렉터리 재사용 고지
