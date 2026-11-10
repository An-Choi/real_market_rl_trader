"""Independent trading calendar and quality-only historical fold selection."""
from __future__ import annotations

from datetime import date
import importlib.metadata

import pandas as pd

from models.walk_forward import build_expanding_walk_forward_folds

# Explicit, sourced corrections to exchange_calendars 4.13.2. Never infer an
# exchange closure merely from absence of price data.
ADHOC_CLOSURES = {
    date(2026, 6, 3): {"reason": "national local election",
        "source": "https://strn.krx.co.kr/corebbs5/BHPSTRN0401/list"},
    date(2026, 7, 17): {"reason": "Constitution Day public holiday restored in 2026",
        "source": "https://www.mois.go.kr/video/bbs/type019/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000255&nttId=123641&searchCode1="},
}


def reference_calendar(start, end):
    import exchange_calendars
    calendar = exchange_calendars.get_calendar("XKRX", start=str(start), end=str(end))
    # Bounds may themselves be holidays (for example Dec 31). The returned
    # schedule already handles them; sessions_in_range rejects those bounds.
    sessions = [day for day in calendar.schedule.index.date if start <= day <= end]
    corrected = [day for day in sessions if day not in ADHOC_CLOSURES]
    provenance = {"source": "exchange_calendars XKRX plus sourced adhoc closures",
        "package_version": importlib.metadata.version("exchange-calendars"),
        "independent_of_price_availability": True,
        "adhoc_closures": [{"date": str(day), **item} for day, item in ADHOC_CLOSURES.items()
                           if start <= day <= end]}
    return corrected, provenance


def fold_coverage(data, calendar, fold):
    coverage = {}
    observed = {symbol: set(pd.to_datetime(frame.Timestamp).dt.date) for symbol, frame in data.items()}
    for segment in ("train", "validation", "test"):
        expected = {day for day in calendar if getattr(fold, segment + "_start") <= day <= getattr(fold, segment + "_end")}
        if not expected:
            raise ValueError("empty reference segment")
        coverage[segment] = {symbol: {"ratio": len(days & expected) / len(expected),
            "missing_days": [str(day) for day in sorted(expected - days)]}
            for symbol, days in observed.items()}
    return coverage


def quality_only_folds(data, calendar, *, count=3, validation_days=60, test_days=20,
                       purge_days=5, minimum_train_days=80, minimum_coverage=.95):
    """Choose the latest eligible fixed-length windows using availability only.

    Rejected recent periods remain in the output. No price, return, prediction,
    policy metric or test score is used to choose historical windows.
    """
    candidates = (len(calendar) - validation_days - 2*purge_days - minimum_train_days) // test_days
    if candidates < count:
        raise ValueError("insufficient calendar history for requested study")
    folds = build_expanding_walk_forward_folds(data, n_folds=candidates,
        validation_days=validation_days, test_days=test_days, purge_days=purge_days,
        min_train_days=minimum_train_days, reference_days=calendar)
    audit, eligible = [], []
    for fold in folds:
        coverage = fold_coverage(data, calendar, fold)
        passed = all(row["ratio"] >= minimum_coverage for segment in coverage.values() for row in segment.values())
        audit.append({**fold.to_dict(), "coverage": coverage, "passed": passed})
        if passed:
            eligible.append(fold)
    if len(eligible) < count:
        return [], audit
    from dataclasses import replace
    return [replace(fold, index=index+1) for index, fold in enumerate(eligible[-count:])], audit
