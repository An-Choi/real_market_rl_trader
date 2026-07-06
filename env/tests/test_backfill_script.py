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
