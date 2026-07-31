import json
import shutil

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
    obs = np.zeros(13, dtype=np.float32)
    mask = np.array([True, True, False])
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
