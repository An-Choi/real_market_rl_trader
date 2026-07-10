from __future__ import annotations

import numpy as np
import pandas as pd

from models.normalization import FeatureNormalizer


def test_feature_normalizer_fits_training_data_and_preserves_portfolio_state(tmp_path) -> None:
    data = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [10.0, 10.0, 10.0]})
    normalizer = FeatureNormalizer.fit(data, ["a", "b"], clip=3.0)
    observation = np.array([4.0, 10.0, 0.5, 0.25], dtype=np.float32)

    transformed = normalizer.transform_observation(observation)

    assert transformed[0] > 1.0
    assert transformed[1] == 0.0
    np.testing.assert_array_equal(transformed[2:], observation[2:])

    path = tmp_path / "normalization.json"
    normalizer.save(path)
    restored = FeatureNormalizer.load(path)
    np.testing.assert_allclose(
        restored.transform_observation(observation),
        transformed,
    )
