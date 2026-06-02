import importlib.util
from pathlib import Path


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
