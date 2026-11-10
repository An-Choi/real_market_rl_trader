"""Recover observed KIS minute rows with backups; never synthesize missing bars."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env/src"))

import pandas as pd
from dotenv import load_dotenv
from data.defect_days import classify_trading_day, sanitize_minute_rows
from pipeline.kis_auth import KISAuth, KISConfig
from pipeline.kis_historical import KISHistoricalFetcher


def merge_observed_rows(existing, fetched, target_date):
    """Keep known observations; add only new timestamps from the requested day."""
    clean = sanitize_minute_rows(existing)
    if fetched.empty:
        return clean
    fetched = fetched.loc[pd.to_datetime(fetched.Timestamp).dt.date == target_date].copy()
    for column in ("Open", "High", "Low", "Close", "Volume", "TradingValue"):
        values = pd.to_numeric(fetched[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ValueError("invalid provider OHLCV")
    if (fetched.Close <= 0).any() or (fetched.High < fetched.Low).any():
        raise ValueError("invalid provider prices")
    new = fetched.loc[~fetched.Timestamp.isin(clean.Timestamp)]
    return sanitize_minute_rows(pd.concat([clean, new], ignore_index=True))


def missing_runs(frame):
    quality = classify_trading_day(frame, require_complete_session=True)
    if quality.trading_date is None or quality.expected_open is None:
        return []
    start = pd.Timestamp.combine(quality.trading_date, quality.expected_open).tz_localize("Asia/Seoul")
    end = pd.Timestamp(str(quality.trading_date) + " 15:19", tz="Asia/Seoul")
    missing = pd.date_range(start, end, freq="min").difference(pd.DatetimeIndex(frame.Timestamp))
    groups = []
    for stamp in missing:
        if not groups or stamp - groups[-1][-1] != pd.Timedelta(minutes=1):
            groups.append([])
        groups[-1].append(stamp)
    return sorted(groups, key=len, reverse=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def run(args):
    if args.output.exists():
        raise ValueError("recovery output exists; preserve it and choose a new revision")
    args.output.mkdir(parents=True)
    load_dotenv(ROOT / ".env", override=False)
    environment = os.environ.get("KIS_ENV", "demo")
    config = KISConfig.from_env(Path(f"/home/an/Desktop/RL_Market/.runtime/credentials/kis-data-{environment}.json"))
    auth = KISAuth(config)
    auth.get_token()  # authenticate before any source mutation
    paths = [path for symbol in args.symbols.split(",")
             for path in sorted((ROOT / "data/raw" / symbol / "1m").glob("*.parquet"))]
    inventory = {str(p.relative_to(ROOT / "data")): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(args.output / "original_inventory.json", inventory)
    records, modified, errors = [], [], 0
    for path in paths:
        source = pd.read_parquet(path)
        groups = dict(tuple(source.groupby(pd.to_datetime(source.Timestamp).dt.date)))
        changes = {}
        for day, old in groups.items():
            original = classify_trading_day(old, require_complete_session=True)
            if original.is_valid:
                continue
            clean = sanitize_minute_rows(old)
            after = classify_trading_day(clean, require_complete_session=True)
            record = {"symbol": path.parents[1].name, "date": str(day),
                "before": original.to_dict(), "deduplicated": after.to_dict(),
                "provider_queries": 0, "added_rows": 0, "applied": False}
            candidate = clean
            if not after.is_valid:
                fetcher = KISHistoricalFetcher(auth, record["symbol"], rate_limit_sleep=args.sleep)
                try:
                    fetched = fetcher.fetch_minute_for_date(day, max_pages=8)
                    record["provider_queries"] += 1
                    candidate = merge_observed_rows(clean, fetched, day)
                    # Directly query the largest missing interval as a separate
                    # check that reverse pagination did not skip a page.
                    for run in missing_runs(candidate)[:2]:
                        time.sleep(args.sleep)
                        extra = fetcher.fetch_minute_for_date(day,
                            end_hour=run[-1].strftime("%H%M%S"), max_pages=1)
                        record["provider_queries"] += 1
                        candidate = merge_observed_rows(candidate, extra, day)
                    errors = 0
                except Exception as error:
                    record["provider_error_type"] = type(error).__name__
                    errors += 1
                time.sleep(args.sleep)
            after = classify_trading_day(candidate, require_complete_session=True)
            record["after"] = after.to_dict()
            record["added_rows"] = len(candidate) - len(clean)
            record["remaining_missing_intervals"] = [
                {"start": str(run[0]), "end": str(run[-1]), "minutes": len(run)}
                for run in missing_runs(candidate)]
            candidate_path = args.output / "candidates" / record["symbol"] / f"{day}.parquet"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.to_parquet(candidate_path, index=False)
            if after.is_valid:
                changes[day] = candidate
                record["status"] = "quality_passed"
            else:
                record["status"] = "unrecoverable_from_provider; excluded"
            records.append(record)
            write_json(args.output / "recovery.json", records)
            print(json.dumps({k: record[k] for k in ("symbol", "date", "status", "added_rows")}), flush=True)
            if errors >= 3:
                raise RuntimeError("three consecutive provider failures; preserve partial audit and retry later")
        if changes and args.apply:
            relative = path.relative_to(ROOT / "data")
            if hashlib.sha256(path.read_bytes()).hexdigest() != inventory[str(relative)]:
                raise ValueError("source changed during recovery")
            backup = args.output / "backups" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            replacement = pd.concat([changes.get(day, frame) for day, frame in groups.items()], ignore_index=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.recovering")
            try:
                replacement.to_parquet(temporary, index=False)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            modified.append(str(relative))
            for record in records:
                if record["symbol"] == path.parents[1].name and record["date"][:7] == path.stem and record["status"] == "quality_passed":
                    record["applied"] = True
            write_json(args.output / "recovery.json", records)
    write_json(args.output / "completion.json", {"completed": True, "apply": args.apply,
        "environment": config.env, "source_partitions": len(paths), "modified_partitions": modified,
        "audited_defective_days": len(records), "quality_passed_days": sum(r["after"]["is_valid"] for r in records),
        "unrecoverable_days": sum(not r["after"]["is_valid"] for r in records),
        "synthetic_rows_added": 0, "orders_submitted": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbols", default="005930,000660,034220,066570,009150")
    parser.add_argument("--sleep", type=float, default=.6)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.sleep < .5:
        raise SystemExit("use at least .5 seconds between demo quotation requests")
    run(args)
