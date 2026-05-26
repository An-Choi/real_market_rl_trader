"""One-shot backfill: KIS daily (2020-2025) + minute (last 30 days) for 005930."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_sys_path() -> None:
    project_root = _project_root()
    env_src = project_root / "env" / "src"
    if str(env_src) not in sys.path:
        sys.path.insert(0, str(env_src))


def _load_local_env() -> None:
    load_dotenv(_project_root() / ".env", override=False)


_bootstrap_sys_path()
_load_local_env()


from pipeline.data_collector import DataCollector  # noqa: E402
from pipeline.kis_auth import KISAuth, KISConfig  # noqa: E402
from pipeline.kis_historical import KISHistoricalFetcher  # noqa: E402


SYMBOL = "005930"
DEFAULT_DAILY_END = "2025-12-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill 005930 OHLCV from KIS API")
    parser.add_argument("--daily-start", type=str, default="2020-01-01")
    parser.add_argument("--daily-end", type=str, default=DEFAULT_DAILY_END)
    parser.add_argument("--minute-days", type=int, default=30,
                        help="Backfill minute candles for the last N calendar days")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-minute", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--token-cache", type=Path, default=Path("data/.kis_token.json"))
    parser.add_argument("--rate-limit-sleep", type=float, default=0.5,
                        help="Demo 0.5s, real 0.05s recommended")
    return parser.parse_args()


def _iso(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.token_cache.parent.mkdir(parents=True, exist_ok=True)

    config = KISConfig.from_env(token_cache_path=args.token_cache)
    auth = KISAuth(config)
    fetcher = KISHistoricalFetcher(auth=auth, symbol=SYMBOL, rate_limit_sleep=args.rate_limit_sleep)
    collector = DataCollector(raw_data_dir=args.raw_dir)

    if not args.skip_daily:
        daily_start = _iso(args.daily_start)
        daily_end = _iso(args.daily_end)
        logging.info("Daily backfill: %s ~ %s", daily_start, daily_end)
        df = collector.fetch_kis_daily(fetcher=fetcher, start=daily_start, end=daily_end)
        logging.info("Daily rows: %d", len(df))
        if not df.empty:
            partition = f"{daily_start.year}-{daily_end.year}"
            path = collector.save_raw_parquet(df, symbol=SYMBOL, interval="1d", partition=partition)
            logging.info("Saved daily Parquet: %s", path)

    if not args.skip_minute:
        minute_end = date.today()
        minute_start = minute_end - timedelta(days=args.minute_days)
        logging.info("Minute backfill: %s ~ %s", minute_start, minute_end)
        df = collector.fetch_kis_minute(fetcher=fetcher, start=minute_start, end=minute_end)
        logging.info("Minute rows: %d", len(df))
        if not df.empty:
            for (year, month), group in df.groupby([df["Timestamp"].dt.year, df["Timestamp"].dt.month]):
                partition = f"{year:04d}-{month:02d}"
                path = collector.save_raw_parquet(group, symbol=SYMBOL, interval="1m", partition=partition)
                logging.info("Saved minute Parquet (%s): %s", partition, path)


if __name__ == "__main__":
    main()
