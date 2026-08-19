import json
import shutil
from datetime import date as _date
from datetime import timedelta as _timedelta

import numpy as np
import pytest

from models.artifact import ArtifactError
from predictor import Predictor


def _tampered_copy(tiny_artifact_dir, tmp_path, mutate):
    dst = tmp_path / "artifact"
    shutil.copytree(tiny_artifact_dir, dst)
    meta_path = dst / "metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    mutate(data)
    meta_path.write_text(json.dumps(data), encoding="utf-8")
    return dst


def test_load_happy_path_and_deterministic_predict(tiny_artifact_dir):
    predictor = Predictor.load(tiny_artifact_dir)
    obs = np.zeros(16, dtype=np.float32)
    mask = np.array([True, True, False, False])
    first = predictor.predict(obs, mask)
    assert first in (0, 1, 2)
    assert predictor.predict(obs, mask) == first          # 결정론
    assert bool(mask[first]) is True                       # mask 준수


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(feature_schema_version=2),                          # schema 불일치
    lambda d: d.update(feature_columns=list(reversed(d["feature_columns"])),
                       ),                                                  # 컬럼 순서
    lambda d: d.update(portfolio_state_fields=["units_held_frac", "tod_frac",
                                               "unrealized_pnl_norm",
                                               "holding_duration_norm"]),  # 필드 순서
    lambda d: (d["action_space"].update(labels=["hold", "clear", "add_unit"])),  # label 순서
    lambda d: (d.update(artifact_format_version=2), d.pop("friction_params")),   # v2 이하
    lambda d: d.pop("friction_params"),                                    # v3인데 누락
], ids=["schema", "feature-cols", "portfolio-fields", "labels", "v2", "no-friction"])
def test_startup_rejections(tiny_artifact_dir, tmp_path, mutate):
    tampered = _tampered_copy(tiny_artifact_dir, tmp_path, mutate)
    with pytest.raises(ArtifactError):
        Predictor.load(tampered)


def _v4_train_data(d: dict) -> dict:
    """기존 v3 train_data({"symbols", "start", "end"})를 검증 통과하는 v4 형태로 변형.

    agent/tests/test_artifact_v4.py의 _v4_metadata_dict()를 모델로 함 — symbols/
    trained_split/split_boundaries/per_symbol이 서로 일관되어야 ArtifactMetadata
    검증을 통과한다.
    """
    symbols = d["train_data"]["symbols"]
    start, end = d["train_data"]["start"], d["train_data"]["end"]
    validation_end = (_date.fromisoformat(end) + _timedelta(days=1)).isoformat()
    return {
        "symbols": symbols,
        "start": start,
        "end": end,
        "trained_split": "all",
        "split_boundaries": {
            "train_end_date": end,
            "validation_end_date": validation_end,
            "purge_days": 0,
        },
        "per_symbol": {
            sym: {"start": start, "end": end, "trading_days": 12} for sym in symbols
        },
    }


def test_serving_accepts_v4_artifact_metadata(tiny_artifact_dir, tmp_path):
    def mutate(d: dict) -> None:
        d["artifact_format_version"] = 4
        d["train_data"] = _v4_train_data(d)

    tampered = _tampered_copy(tiny_artifact_dir, tmp_path, mutate)
    predictor = Predictor.load(tampered)
    obs = np.zeros(16, dtype=np.float32)
    mask = np.array([True, True, False, False])
    assert predictor.predict(obs, mask) in (0, 1, 2)


def test_serving_accepts_only_approved_qualified_v5(tiny_artifact_dir, tmp_path):
    def mutate(d: dict) -> None:
        d["artifact_format_version"] = 5
        d["train_data"] = _v4_train_data(d)
        d["train_data"]["trained_split"] = "train"
        d["deployment_status"] = "approved"
        d["training_params"] = {
            "validation": {"best": {"qualified": True}}
        }

    artifact = _tampered_copy(tiny_artifact_dir, tmp_path, mutate)
    assert Predictor.load(artifact).meta.deployment_status == "approved"


def test_serving_rejects_rejected_v5(tiny_artifact_dir, tmp_path):
    def mutate(d: dict) -> None:
        d["artifact_format_version"] = 5
        d["train_data"] = _v4_train_data(d)
        d["train_data"]["trained_split"] = "train"
        d["deployment_status"] = "rejected"
        d["training_params"] = {
            "validation": {"best_candidate": {"qualified": False}}
        }

    artifact = _tampered_copy(tiny_artifact_dir, tmp_path, mutate)
    with pytest.raises(ArtifactError, match="not an approved qualified"):
        Predictor.load(artifact)


def test_serving_rejects_v2_artifact_metadata_with_v3_v4_message(tiny_artifact_dir, tmp_path):
    def mutate(d: dict) -> None:
        d["artifact_format_version"] = 2
        d.pop("friction_params", None)

    tampered = _tampered_copy(tiny_artifact_dir, tmp_path, mutate)
    with pytest.raises(ArtifactError, match="v3/v4"):
        Predictor.load(tampered)
