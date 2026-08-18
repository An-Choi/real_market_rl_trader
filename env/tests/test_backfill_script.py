import importlib.util
from pathlib import Path

import pytest


def _load_backfill():
    script = Path(__file__).resolve().parents[2] / "scripts" / "backfill.py"
    spec = importlib.util.spec_from_file_location("backfill_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_symbols_splits_and_strips() -> None:
    mod = _load_backfill()
    assert mod._parse_symbols("005930") == ["005930"]
    assert mod._parse_symbols("005930, 000660 ,005380") == ["005930", "000660", "005380"]
    assert mod._parse_symbols("005930,,") == ["005930"]


def test_daily_end_defaults_to_today_at_runtime(monkeypatch) -> None:
    mod = _load_backfill()
    monkeypatch.setattr("sys.argv", ["backfill.py"])

    args = mod.parse_args()

    assert args.daily_end is None


def test_parse_force_months_valid_and_dedup() -> None:
    mod = _load_backfill()
    assert mod._parse_force_months("") == []
    assert mod._parse_force_months("2026-05") == ["2026-05"]
    assert mod._parse_force_months(" 2026-05 ,2026-06,2026-05") == ["2026-05", "2026-06"]


def test_parse_force_months_rejects_bad_format() -> None:
    mod = _load_backfill()
    for bad in ("2026-13", "2026/05", "202605", "2026-5", "abcd-ef"):
        with pytest.raises(ValueError):
            mod._parse_force_months(bad)


def test_cli_has_refresh_flags(monkeypatch) -> None:
    mod = _load_backfill()
    monkeypatch.setattr("sys.argv", ["backfill.py", "--refresh-current",
                                     "--force-months", "2026-05"])
    args = mod.parse_args()
    assert args.refresh_current is True
    assert args.force_months == "2026-05"


def test_current_month_label() -> None:
    from datetime import date
    mod = _load_backfill()
    assert mod._current_month_label(date(2026, 7, 6)) == "2026-07"
    assert mod._current_month_label(date(2026, 11, 1)) == "2026-11"


def test_today_bar_count_reads_current_partition(tmp_path) -> None:
    from datetime import date
    import pandas as pd

    mod = _load_backfill()
    d = tmp_path / "005930" / "1m"
    d.mkdir(parents=True)
    ts = (list(pd.date_range("2026-07-03 09:00", periods=3, freq="1min", tz="Asia/Seoul"))
          + list(pd.date_range("2026-07-06 09:00", periods=2, freq="1min", tz="Asia/Seoul")))
    pd.DataFrame({"Timestamp": ts, "Close": [1.0] * 5}).to_parquet(
        d / "2026-07.parquet", index=False)

    assert mod._today_bar_count(tmp_path, "005930", date(2026, 7, 6)) == 2
    assert mod._today_bar_count(tmp_path, "005930", date(2026, 7, 7)) == 0       # 데이터 없음
    assert mod._today_bar_count(tmp_path, "000660", date(2026, 7, 6)) == 0       # 파티션 없음


def test_today_bar_summary_flags_zero_and_partial_counts() -> None:
    mod = _load_backfill()
    assert mod._today_bar_summary(381) == "381"
    assert mod._today_bar_summary(0) == "0 (holiday or missing)"
    assert mod._today_bar_summary(120) == "120 ⚠️ expected 381"


def test_write_github_summary_appends_when_env_set(tmp_path, monkeypatch) -> None:
    mod = _load_backfill()
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    mod._write_github_summary("line1")
    mod._write_github_summary("line2")
    assert summary.read_text() == "line1\nline2\n"

    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    mod._write_github_summary("ignored")     # env 없으면 no-op (예외 없이)


def test_cli_market_defaults_to_j(monkeypatch) -> None:
    mod = _load_backfill()
    monkeypatch.setattr("sys.argv", ["backfill.py"])
    args = mod.parse_args()
    assert args.market == "j"
    assert args.skip_daily is False


def test_cli_market_un_implies_skip_daily(monkeypatch) -> None:
    mod = _load_backfill()
    monkeypatch.setattr("sys.argv", ["backfill.py", "--market", "un"])
    args = mod.parse_args()
    assert args.market == "un"
    assert args.skip_daily is True                         # UN은 분봉만 수집 (스펙 §2)


def test_cli_market_un_rejects_refresh_current(monkeypatch) -> None:
    mod = _load_backfill()
    monkeypatch.setattr("sys.argv", ["backfill.py", "--market", "un", "--refresh-current"])
    with pytest.raises(SystemExit):
        mod.parse_args()


def test_market_profiles_match_spec() -> None:
    mod = _load_backfill()
    un = mod.MARKET_PROFILES["un"]
    assert un["market_code"] == "UN"
    assert un["interval"] == "1m-un"
    assert un["end_hour"] == "200000"
    assert un["first_hour"] == "080000"
    assert un["max_pages"] >= 6                            # 08:00~20:00 실측 6페이지
    # J 프로파일은 기존 하드코딩 값과 정확히 일치해야 함 (동작 불변 가드)
    assert mod.MARKET_PROFILES["j"] == {
        "market_code": "J", "interval": "1m", "end_hour": "153000",
        "first_hour": "090000", "max_pages": 4,
    }
