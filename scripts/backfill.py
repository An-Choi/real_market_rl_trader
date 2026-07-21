"""Backfill KIS daily + minute OHLCV for one or more symbols.

Minute data is saved month-by-month (crash-safe, resumable): months whose
Parquet already exists are skipped unless --overwrite is given.
KIS minute history is capped at a rolling ~1 year (see troubleshooting spec).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_sys_path() -> None:
    env_src = _project_root() / "env" / "src"
    if str(env_src) not in sys.path:
        sys.path.insert(0, str(env_src))


def _load_local_env() -> None:
    load_dotenv(_project_root() / ".env", override=False)


_bootstrap_sys_path()
_load_local_env()


from pipeline.data_collector import DataCollector  # noqa: E402
from pipeline.kis_auth import KISAuth, KISConfig  # noqa: E402
from pipeline.kis_historical import KISHistoricalFetcher  # noqa: E402


DEFAULT_SYMBOLS = "005930"
EXPECTED_BARS_PER_DAY = 381


def _parse_symbols(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_force_months(raw: str) -> list[str]:
    """Parse comma-separated YYYY-MM labels; dedup preserving order."""
    months: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if len(token) != 7 or token[4] != "-":
            raise ValueError(f"--force-months expects YYYY-MM, got {token!r}")
        try:
            parsed = datetime.strptime(token, "%Y-%m")
        except ValueError as exc:
            raise ValueError(f"--force-months expects YYYY-MM, got {token!r}") from exc
        label = f"{parsed.year:04d}-{parsed.month:02d}"
        if label not in months:
            months.append(label)
    return months


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill KIS OHLCV for one or more symbols")
    parser.add_argument("--symbols", type=str, default=DEFAULT_SYMBOLS,
                        help="Comma-separated symbol codes, e.g. 005930,000660")
    parser.add_argument("--daily-start", type=str, default="2020-01-01")
    parser.add_argument("--daily-end", type=str, default=None,
                        help="Inclusive daily end date, YYYY-MM-DD (default: today)")
    parser.add_argument("--minute-days", type=int, default=380,
                        help="Minute candles for the last N calendar days (KIS cap ~1yr)")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-minute", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-fetch and overwrite existing Parquet partitions")
    parser.add_argument("--refresh-current", action="store_true",
                        help="Daily-run mode: re-fetch current minute month + full daily refresh")
    parser.add_argument("--force-months", type=str, default="",
                        help="Comma-separated YYYY-MM minute months to force refetch")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--token-cache", type=Path, default=Path("data/.kis_token.json"))
    parser.add_argument("--rate-limit-sleep", type=float, default=0.4,
                        help="Demo 0.4-0.5s, real 0.05s recommended")
    return parser.parse_args()


def _iso(value: str) -> date:
    return date.fromisoformat(value)


def _backfill_daily(collector, fetcher, symbol, start, end, raw_dir, overwrite) -> None:
    partition = f"{start.year}-{end.year}"
    path = raw_dir / symbol / "1d" / f"{partition}.parquet"
    if path.exists() and not overwrite:
        logging.info("Skip existing daily partition: %s", path)
        return
    df = collector.fetch_kis_daily(fetcher=fetcher, start=start, end=end)
    logging.info("[%s] daily rows: %d", symbol, len(df))
    if not df.empty:
        saved = collector.save_raw_parquet(df, symbol=symbol, interval="1d", partition=partition)
        logging.info("Saved daily Parquet: %s", saved)


def _current_month_label(today: date) -> str:
    return f"{today.year:04d}-{today.month:02d}"


def _today_bar_count(raw_dir: Path, symbol: str, today: date) -> int:
    import pandas as pd

    path = raw_dir / symbol / "1m" / f"{_current_month_label(today)}.parquet"
    if not path.exists():
        return 0
    df = pd.read_parquet(path, columns=["Timestamp"])
    return int((df["Timestamp"].dt.date == today).sum())


def _today_bar_summary(bars: int) -> str:
    if bars == EXPECTED_BARS_PER_DAY:
        return str(bars)
    if bars == 0:
        return "0 (holiday or missing)"
    return f"{bars} ⚠️ expected {EXPECTED_BARS_PER_DAY}"


def _write_github_summary(text: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.token_cache.parent.mkdir(parents=True, exist_ok=True)

    config = KISConfig.from_env(token_cache_path=args.token_cache)
    auth = KISAuth(config)
    collector = DataCollector(raw_data_dir=args.raw_dir)

    symbols = _parse_symbols(args.symbols)
    daily_start = _iso(args.daily_start)
    daily_end = _iso(args.daily_end) if args.daily_end else date.today()
    minute_end = date.today()
    minute_start = minute_end - timedelta(days=args.minute_days)

    force_months = _parse_force_months(args.force_months)
    today = date.today()

    if args.refresh_current:
        _write_github_summary("| symbol | daily | minute saved | force | today bars | coverage |")
        _write_github_summary("|---|---|---|---|---|---|")

    for symbol in symbols:
        fetcher = KISHistoricalFetcher(
            auth=auth, symbol=symbol, rate_limit_sleep=args.rate_limit_sleep
        )
        logging.info("=== Backfill %s ===", symbol)

        if args.refresh_current:
            daily_status = collector.refresh_daily_all(
                fetcher, symbol=symbol, start=daily_start, end=daily_end
            )
            saved = collector.backfill_minute_monthly(
                fetcher=fetcher, symbol=symbol,
                start=minute_start, end=minute_end,
                overwrite_partitions={_current_month_label(today)},
            )
            force_results = (
                collector.force_month_refetch(fetcher, symbol=symbol, months=force_months)
                if force_months else {}
            )
            bars = _today_bar_count(args.raw_dir, symbol, today)
            coverage = collector.audit_minute_coverage(symbol=symbol, today=today)
            for warning in coverage:
                logging.warning("[%s] coverage: %s", symbol, warning)
            line = (f"| {symbol} | daily={daily_status} | minute={saved or '-'} "
                    f"| force={force_results or '-'} | today_bars={_today_bar_summary(bars)} "
                    f"| coverage={'⚠️ ' + '; '.join(coverage) if coverage else 'ok'} |")
            logging.info("summary: %s", line)
            _write_github_summary(line)
            continue

        # 기존 일회성 백필 경로
        if not args.skip_daily:
            _backfill_daily(collector, fetcher, symbol, daily_start, daily_end,
                            args.raw_dir, args.overwrite)
        if not args.skip_minute:
            logging.info("[%s] minute backfill: %s ~ %s", symbol, minute_start, minute_end)
            saved = collector.backfill_minute_monthly(
                fetcher=fetcher, symbol=symbol,
                start=minute_start, end=minute_end, overwrite=args.overwrite,
            )
            logging.info("[%s] minute partitions saved: %d", symbol, len(saved))
        if force_months:
            force_results = collector.force_month_refetch(
                fetcher, symbol=symbol, months=force_months
            )
            logging.info("[%s] force months: %s", symbol, force_results)
        for warning in collector.audit_minute_coverage(symbol=symbol, today=today):
            logging.warning("[%s] coverage: %s", symbol, warning)


if __name__ == "__main__":
    main()
