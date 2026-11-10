from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.run_learning_diagnostics import easy_market
from experiments.run_ablation import data_fingerprints
from experiments.finalize_research import fold_from_dict
from data.feature_engineer import FeatureEngineer


def test_synthetic_features_are_past_price_changes_and_never_regime_or_future_labels():
    for regime in ("rising", "falling", "flat"):
        data = easy_market(regime, 42, days=3)
        expected = np.r_[0., np.diff(np.log(data.Close.to_numpy()))]
        np.testing.assert_allclose(data[FeatureEngineer.FEATURE_COLUMNS[0]], expected, atol=1e-14)
        for column in FeatureEngineer.FEATURE_COLUMNS[1:]:
            assert (data[column] == 0).all()


def test_fold_manifest_parser_ignores_coverage_but_preserves_exact_dates():
    fold = fold_from_dict(dict(index=1, train_start="2025-01-01", train_end="2025-02-28",
        validation_start="2025-03-10", validation_end="2025-03-28",
        test_start="2025-04-07", test_end="2025-04-30", purge_days=5,
        coverage={"test": {}}, passed=True))
    assert str(fold.test_start) == "2025-04-07"
    assert fold.purge_days == 5


def test_final_evaluation_refuses_changed_data_before_creating_freeze(tmp_path, monkeypatch):
    import experiments.finalize_research as finalize
    from experiments.run_ablation import write_json
    data = {"SYNTHETIC": easy_market("rising", 42, days=3)}
    write_json(tmp_path / "study_manifest.json", {"arguments": {"symbols": "SYNTHETIC"},
        "data_fingerprints": data_fingerprints(data)})
    write_json(tmp_path / "completion.json", {"completed": True, "test_evaluated": False})
    write_json(tmp_path / "results.json", [])
    data["SYNTHETIC"].loc[0, "Adv20"] *= 2
    monkeypatch.setattr(finalize, "load_market", lambda symbols: (data, [], {}))
    with pytest.raises(ValueError, match="data changed"):
        finalize.run(tmp_path)
    assert not (tmp_path / "frozen_evaluation").exists()


def test_final_evaluation_refuses_second_selection_after_test(tmp_path):
    import experiments.finalize_research as finalize
    (tmp_path / "frozen_evaluation").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        finalize.run(tmp_path)
