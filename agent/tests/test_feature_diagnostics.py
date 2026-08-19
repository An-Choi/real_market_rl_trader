from __future__ import annotations

import numpy as np
import pandas as pd

from models.feature_diagnostics import (
    build_feature_quality_report,
    feature_redundancy_report,
    feature_temporal_stability_report,
)
from models.normalization import FeatureNormalizer


def test_feature_quality_report_detects_constant_and_forward_relation():
    signal = np.linspace(0.0, 1.0, 20)
    frame = pd.DataFrame({
        "Timestamp": pd.date_range("2026-01-02 09:00", periods=20, freq="5min"),
        "Close": np.exp(np.cumsum(0.001 + 0.01 * signal)),
        "signal": signal,
        "constant": 1.0,
    })
    normalizer = FeatureNormalizer.fit(frame, ["signal", "constant"], clip=2.0)
    report = build_feature_quality_report(
        {"AAA": frame}, ["signal", "constant"], normalization=normalizer
    )
    by_feature = {row["feature"]: row for row in report}
    assert by_feature["constant"]["near_constant"] is True
    assert by_feature["signal"]["spearman_forward_1"] > 0.0
    assert by_feature["signal"]["missing_rate"] == 0.0


def test_redundancy_report_ranks_correlated_pair_first():
    frame = pd.DataFrame({
        "a": np.arange(20),
        "b": np.arange(20) * 2,
        "c": np.tile([0, 1], 10),
    })
    report = feature_redundancy_report({"AAA": frame}, ["a", "b", "c"])
    assert {report[0]["feature_a"], report[0]["feature_b"]} == {"a", "b"}
    assert report[0]["spearman_correlation"] == 1.0


def test_temporal_stability_report_exposes_window_sign_consistency():
    rows = []
    for day in pd.date_range("2026-01-05", periods=8, freq="B"):
        signal = np.arange(5, dtype=float)
        log_close = np.r_[0.0, np.cumsum(signal[:-1] * 0.01)]
        for minute, (value, close) in enumerate(zip(signal, np.exp(log_close))):
            rows.append({
                "Timestamp": day + pd.Timedelta(minutes=5 * minute),
                "Close": close,
                "signal": value,
            })
    report = feature_temporal_stability_report(
        {"AAA": pd.DataFrame(rows)}, ["signal"], window_days=4, horizons=(1,)
    )
    assert report[0]["window_count"] == 2
    assert report[0]["spearman_forward_1_mean"] > 0.9
    assert report[0]["spearman_forward_1_sign_consistency"] == 1.0
