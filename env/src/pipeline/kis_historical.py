"""KIS REST historical OHLCV fetcher (daily + minute)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

from pipeline.kis_auth import KISAuth


DAILY_ENDPOINT = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
DAILY_TR_ID = "FHKST03010100"
DAILY_WINDOW_DAYS = 90  # KIS 100건 한도, 영업일 기준 90 캘린더일이면 안전

DAILY_COLUMN_MAP = {
    "stck_bsop_date": "Date",
    "stck_oprc": "Open",
    "stck_hgpr": "High",
    "stck_lwpr": "Low",
    "stck_clpr": "Close",
    "acml_vol": "Volume",
    "acml_tr_pbmn": "TradingValue",
    "prdy_vrss": "Change",
}

DAILY_OUTPUT_COLUMNS = list(DAILY_COLUMN_MAP.values())

MINUTE_ENDPOINT = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
MINUTE_TR_ID = "FHKST03010230"

MINUTE_COLUMN_MAP = {
    "stck_oprc": "Open",
    "stck_hgpr": "High",
    "stck_lwpr": "Low",
    "stck_prpr": "Close",
    "cntg_vol": "Volume",
    "acml_tr_pbmn": "TradingValue",
}

MINUTE_OUTPUT_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume", "TradingValue"]


@dataclass
class KISHistoricalFetcher:
    auth: KISAuth
    symbol: str
    rate_limit_sleep: float = 0.5  # demo=0.5, real=0.05 권장

    def fetch_daily(self, start: date, end: date) -> pd.DataFrame:
        windows = _make_daily_windows(start, end)
        frames: list[pd.DataFrame] = []
        for window_start, window_end in windows:
            frame = self._fetch_daily_window(window_start, window_end)
            frames.append(frame)
            if self.rate_limit_sleep > 0:
                time.sleep(self.rate_limit_sleep)

        if not frames:
            return pd.DataFrame(columns=DAILY_OUTPUT_COLUMNS)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return combined

    def _fetch_daily_window(self, window_start: date, window_end: date) -> pd.DataFrame:
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": self.symbol,
            "FID_INPUT_DATE_1": window_start.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": window_end.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        }
        body = self._get(DAILY_ENDPOINT, DAILY_TR_ID, params)
        rows = body.get("output2") or []
        if not rows:
            return pd.DataFrame(columns=DAILY_OUTPUT_COLUMNS)
        return _normalize_daily(rows)

    def fetch_minute_for_date(
        self,
        target_date: date,
        end_hour: str = "153000",
        max_pages: int = 4,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        current_hour = end_hour
        for _ in range(max_pages):
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": self.symbol,
                "FID_INPUT_HOUR_1": current_hour,
                "FID_INPUT_DATE_1": target_date.strftime("%Y%m%d"),
                "FID_PW_DATA_INCU_YN": "N",
                "FID_FAKE_TICK_INCU_YN": "",
            }
            body = self._get(MINUTE_ENDPOINT, MINUTE_TR_ID, params)
            rows = body.get("output2") or []
            if not rows:
                break
            frame = _normalize_minute(rows)
            frames.append(frame)
            earliest = frame["Timestamp"].min()
            if earliest.strftime("%H%M%S") <= "090000":
                break
            current_hour = (earliest - pd.Timedelta(minutes=1)).strftime("%H%M%S")
            if self.rate_limit_sleep > 0:
                time.sleep(self.rate_limit_sleep)

        if not frames:
            return pd.DataFrame(columns=MINUTE_OUTPUT_COLUMNS)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        return combined

    def fetch_minute_range(
        self,
        start: date,
        end: date,
        max_pages_per_day: int = 4,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # 월~금만
                frame = self.fetch_minute_for_date(
                    target_date=current,
                    max_pages=max_pages_per_day,
                )
                if not frame.empty:
                    frames.append(frame)
            current += timedelta(days=1)

        if not frames:
            return pd.DataFrame(columns=MINUTE_OUTPUT_COLUMNS)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("Timestamp").reset_index(drop=True)
        return combined

    def _get(self, endpoint: str, tr_id: str, params: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.auth.get_token()}",
            "appkey": self.auth.app_key,
            "appsecret": self.auth.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        response = requests.get(
            self.auth.base_url + endpoint,
            params=params,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("rt_cd") != "0":
            raise RuntimeError(f"KIS API error: {body.get('msg1')} (rt_cd={body.get('rt_cd')})")
        return body


def _make_daily_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=DAILY_WINDOW_DAYS - 1), end)
        windows.append((current, window_end))
        current = window_end + timedelta(days=1)
    return windows


def _normalize_daily(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns=DAILY_COLUMN_MAP)
    df = df[DAILY_OUTPUT_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d").astype("datetime64[ns]")
    for col in ["Open", "High", "Low", "Close", "Change"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["Volume", "TradingValue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
    return df


def _normalize_minute(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    timestamp = pd.to_datetime(
        df["stck_bsop_date"] + df["stck_cntg_hour"],
        format="%Y%m%d%H%M%S",
    )
    timestamp = timestamp.dt.tz_localize("Asia/Seoul")
    df = df.rename(columns=MINUTE_COLUMN_MAP)
    df["Timestamp"] = timestamp
    df = df[MINUTE_OUTPUT_COLUMNS].copy()
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["Volume", "TradingValue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
    return df
