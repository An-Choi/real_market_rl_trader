from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import responses

from pipeline.kis_auth import KISAuth, KISConfig
from pipeline.kis_historical import KISHistoricalFetcher, _make_daily_windows


@pytest.fixture
def auth(tmp_path: Path) -> KISAuth:
    config = KISConfig(
        env="demo",
        app_key="K",
        app_secret="S",
        token_cache_path=tmp_path / "tok.json",
    )
    config.token_cache_path.write_text(json.dumps({
        "access_token": "TEST_TOKEN",
        "expires_at": (datetime.now() + timedelta(hours=12)).isoformat(),
    }))
    return KISAuth(config)


def _daily_row(date_str: str, close: int) -> dict:
    return {
        "stck_bsop_date": date_str,
        "stck_oprc": str(close - 100),
        "stck_hgpr": str(close + 200),
        "stck_lwpr": str(close - 200),
        "stck_clpr": str(close),
        "acml_vol": "1000000",
        "acml_tr_pbmn": "70000000000",
        "prdy_vrss": "100",
        "prdy_vrss_sign": "2",
    }


@responses.activate
def test_fetch_daily_single_window_normalizes_columns(auth: KISAuth) -> None:
    responses.add(
        responses.GET,
        "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        json={
            "rt_cd": "0",
            "msg_cd": "MCA00000",
            "msg1": "정상처리",
            "output1": {"hts_kor_isnm": "삼성전자"},
            "output2": [
                _daily_row("20240105", 75000),
                _daily_row("20240104", 74000),
                _daily_row("20240103", 73500),
            ],
        },
        status=200,
    )

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    df = fetcher.fetch_daily(start=date(2024, 1, 3), end=date(2024, 1, 5))

    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "TradingValue", "Change"]
    assert len(df) == 3
    assert df["Date"].is_monotonic_increasing
    assert df["Close"].iloc[-1] == 75000.0
    assert df["Date"].dtype == "datetime64[ns]"
    assert df["Open"].dtype == "float64"
    assert df["Volume"].dtype == "int64"


@responses.activate
def test_fetch_daily_paginates_over_long_range(auth: KISAuth) -> None:
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    for window_idx in range(3):
        rows = [_daily_row(f"2024{(window_idx + 1):02d}{day:02d}", 70000 + window_idx)
                for day in range(1, 11)]
        responses.add(responses.GET, url, json={
            "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
            "output1": {}, "output2": rows,
        }, status=200)

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    df = fetcher.fetch_daily(start=date(2024, 1, 1), end=date(2024, 9, 1))

    assert len(responses.calls) == 3
    assert df["Date"].is_monotonic_increasing
    assert df["Date"].is_unique


@responses.activate
def test_fetch_daily_sends_correct_headers(auth: KISAuth) -> None:
    responses.add(
        responses.GET,
        "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        json={"rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok", "output1": {}, "output2": []},
        status=200,
    )

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    fetcher.fetch_daily(start=date(2024, 1, 1), end=date(2024, 1, 2))

    call = responses.calls[0]
    assert call.request.headers["authorization"] == "Bearer TEST_TOKEN"
    assert call.request.headers["appkey"] == "K"
    assert call.request.headers["appsecret"] == "S"
    assert call.request.headers["tr_id"] == "FHKST03010100"


@responses.activate
def test_fetch_daily_raises_on_api_error(auth: KISAuth) -> None:
    responses.add(
        responses.GET,
        "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        json={"rt_cd": "1", "msg_cd": "MCA00001", "msg1": "잘못된 종목코드", "output1": {}, "output2": []},
        status=200,
    )

    fetcher = KISHistoricalFetcher(auth=auth, symbol="999999", rate_limit_sleep=0.0)
    with pytest.raises(RuntimeError, match="잘못된 종목코드"):
        fetcher.fetch_daily(start=date(2024, 1, 1), end=date(2024, 1, 2))


def test_make_daily_windows_are_contiguous_and_non_overlapping() -> None:
    windows = _make_daily_windows(date(2024, 1, 1), date(2024, 9, 1))
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == prev_end + timedelta(days=1)
    assert windows[0][0] == date(2024, 1, 1)
    assert windows[-1][1] == date(2024, 9, 1)


def _minute_row(date_str: str, hhmmss: str, close: int) -> dict:
    return {
        "stck_bsop_date": date_str,
        "stck_cntg_hour": hhmmss,
        "stck_prpr": str(close),
        "stck_oprc": str(close - 50),
        "stck_hgpr": str(close + 100),
        "stck_lwpr": str(close - 100),
        "cntg_vol": "10000",
        "acml_tr_pbmn": "750000000",
    }


@responses.activate
def test_fetch_minute_normalizes_columns_and_timestamp(auth: KISAuth) -> None:
    responses.add(
        responses.GET,
        "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice",
        json={
            "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
            "output1": {}, "output2": [
                _minute_row("20240105", "153000", 75000),
                _minute_row("20240105", "152900", 74950),
                _minute_row("20240105", "152800", 74900),
            ],
        },
        status=200,
    )

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    df = fetcher.fetch_minute_for_date(target_date=date(2024, 1, 5), end_hour="153000", max_pages=1)

    assert list(df.columns) == ["Timestamp", "Open", "High", "Low", "Close", "Volume", "TradingValue"]
    assert len(df) == 3
    assert df["Timestamp"].is_monotonic_increasing
    assert str(df["Timestamp"].dt.tz) == "Asia/Seoul"
    assert df["Close"].iloc[-1] == 75000.0


@responses.activate
def test_fetch_minute_for_date_paginates_within_day(auth: KISAuth) -> None:
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    rows_page1 = [_minute_row("20240105", f"{h:02d}{m:02d}00", 75000)
                  for h in range(15, 13, -1) for m in range(59, -1, -1)]
    rows_page2 = [_minute_row("20240105", f"{h:02d}{m:02d}00", 74000)
                  for h in range(13, 11, -1) for m in range(59, -1, -1)]
    responses.add(responses.GET, url, json={
        "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
        "output1": {}, "output2": rows_page1,
    }, status=200)
    responses.add(responses.GET, url, json={
        "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
        "output1": {}, "output2": rows_page2,
    }, status=200)

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    df = fetcher.fetch_minute_for_date(target_date=date(2024, 1, 5), end_hour="153000", max_pages=2)

    assert len(responses.calls) == 2
    assert df["Timestamp"].is_unique


@responses.activate
def test_fetch_minute_range_iterates_dates(auth: KISAuth) -> None:
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    for d in ("20240108", "20240109", "20240110"):
        responses.add(responses.GET, url, json={
            "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
            "output1": {}, "output2": [
                _minute_row(d, "153000", 75000),
                _minute_row(d, "152900", 74950),
            ],
        }, status=200)

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    df = fetcher.fetch_minute_range(start=date(2024, 1, 8), end=date(2024, 1, 10), max_pages_per_day=1)

    assert len(df) == 6
    assert df["Timestamp"].dt.date.nunique() == 3


@responses.activate
def test_get_retries_on_http_429(auth: KISAuth, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.kis_historical.time.sleep", lambda *a, **k: None)
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    responses.add(responses.GET, url, status=429, json={})
    responses.add(responses.GET, url, status=200, json={
        "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
        "output1": {}, "output2": [_daily_row("20240102", 70000)],
    })

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    df = fetcher.fetch_daily(start=date(2024, 1, 2), end=date(2024, 1, 2))

    assert len(responses.calls) == 2
    assert len(df) == 1


@responses.activate
def test_get_retries_on_soft_rate_limit_code(auth: KISAuth, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.kis_historical.time.sleep", lambda *a, **k: None)
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    responses.add(responses.GET, url, status=200, json={
        "rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과하였습니다.",
        "output1": {}, "output2": [],
    })
    responses.add(responses.GET, url, status=200, json={
        "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
        "output1": {}, "output2": [_daily_row("20240102", 70000)],
    })

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    df = fetcher.fetch_daily(start=date(2024, 1, 2), end=date(2024, 1, 2))

    assert len(responses.calls) == 2
    assert len(df) == 1


def _query(call) -> dict:
    return parse_qs(urlparse(call.request.url).query)


@responses.activate
def test_fetch_minute_un_market_sends_un_params(auth: KISAuth) -> None:
    """UN fetcher는 시장구분 UN, 기본 조회시각 minute_end_hour(20:00)를 보낸다."""
    responses.add(
        responses.GET,
        "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice",
        json={"rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
              "output1": {}, "output2": [_minute_row("20260807", "195900", 75000)]},
        status=200,
    )

    fetcher = KISHistoricalFetcher(
        auth=auth, symbol="005930", rate_limit_sleep=0.0,
        market_code="UN", minute_end_hour="200000", minute_first_hour="080000",
    )
    df = fetcher.fetch_minute_for_date(target_date=date(2026, 8, 7), max_pages=1)

    q = _query(responses.calls[0])
    assert q["FID_COND_MRKT_DIV_CODE"] == ["UN"]
    assert q["FID_INPUT_HOUR_1"] == ["200000"]
    assert len(df) == 1


@responses.activate
def test_fetch_minute_un_paginates_below_0900_until_first_hour(auth: KISAuth) -> None:
    """페이징 하한이 first_hour(08:00)로 내려간다 — 09:00 하드코딩이면 1페이지에서 멈춘다."""
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    rows_p1 = [_minute_row("20260807", f"09{m:02d}00", 75000) for m in range(59, -1, -1)]
    rows_p2 = [_minute_row("20260807", f"08{m:02d}00", 74900) for m in range(49, -1, -1)]
    responses.add(responses.GET, url, json={
        "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok", "output1": {}, "output2": rows_p1,
    }, status=200)
    responses.add(responses.GET, url, json={
        "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok", "output1": {}, "output2": rows_p2,
    }, status=200)

    fetcher = KISHistoricalFetcher(
        auth=auth, symbol="005930", rate_limit_sleep=0.0,
        market_code="UN", minute_end_hour="200000", minute_first_hour="080000",
    )
    df = fetcher.fetch_minute_for_date(
        target_date=date(2026, 8, 7), end_hour="100000", max_pages=4)

    assert len(responses.calls) == 2                       # 08:00 도달 후 종료
    assert df["Timestamp"].min().strftime("%H%M%S") == "080000"
    assert df["Timestamp"].max().strftime("%H%M%S") == "095900"


@responses.activate
def test_fetch_minute_default_market_unchanged_j(auth: KISAuth) -> None:
    """기본값 회귀 가드: 시장구분 J, 조회시각 15:30, 09:00 도달 시 조기종료."""
    responses.add(
        responses.GET,
        "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice",
        json={"rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok",
              "output1": {}, "output2": [_minute_row("20240105", "090000", 75000)]},
        status=200,
    )

    fetcher = KISHistoricalFetcher(auth=auth, symbol="005930", rate_limit_sleep=0.0)
    fetcher.fetch_minute_for_date(target_date=date(2024, 1, 5), max_pages=4)

    assert len(responses.calls) == 1                       # 09:00 하한 조기종료 유지
    q = _query(responses.calls[0])
    assert q["FID_COND_MRKT_DIV_CODE"] == ["J"]
    assert q["FID_INPUT_HOUR_1"] == ["153000"]


@responses.activate
def test_fetch_daily_uses_market_code_field(auth: KISAuth) -> None:
    responses.add(
        responses.GET,
        "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        json={"rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok", "output1": {}, "output2": []},
        status=200,
    )

    fetcher = KISHistoricalFetcher(
        auth=auth, symbol="005930", rate_limit_sleep=0.0, market_code="UN")
    fetcher.fetch_daily(start=date(2024, 1, 1), end=date(2024, 1, 2))

    assert _query(responses.calls[0])["FID_COND_MRKT_DIV_CODE"] == ["UN"]
