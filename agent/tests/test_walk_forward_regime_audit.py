from __future__ import annotations

import pandas as pd

from experiments.walk_forward_train import classify_market_regime


def test_three_percent_twenty_day_move_is_directional_regime():
    assert classify_market_regime(0.03, threshold=0.03) == "bull"
    assert classify_market_regime(-0.03, threshold=0.03) == "bear"
    assert classify_market_regime(0.01, threshold=0.03) == "sideways"


def test_regime_labels_use_only_supplied_return():
    values = pd.Series([0.04, -0.04, 0.0])
    assert [classify_market_regime(value, 0.03) for value in values] == [
        "bull", "bear", "sideways"
    ]
