from __future__ import annotations

import pandas as pd
import pytest

from experiments.train_rl import split_featured_data


def test_split_featured_data_uses_chronological_order() -> None:
    data = pd.DataFrame({"Close": list(range(10))})

    train_data, test_data = split_featured_data(data, train_ratio=0.6)

    assert train_data["Close"].tolist() == [0, 1, 2, 3, 4, 5]
    assert test_data["Close"].tolist() == [6, 7, 8, 9]


def test_split_featured_data_rejects_invalid_ratio() -> None:
    data = pd.DataFrame({"Close": list(range(10))})

    with pytest.raises(ValueError, match="train_ratio"):
        split_featured_data(data, train_ratio=1.0)


def test_split_featured_data_requires_enough_rows_per_side() -> None:
    data = pd.DataFrame({"Close": list(range(3))})

    with pytest.raises(ValueError, match="Not enough featured rows"):
        split_featured_data(data, train_ratio=0.7)
