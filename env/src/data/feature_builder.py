"""옵트인 parquet 캐시 래퍼. 계산 코어(FeatureEngineer.transform)와 분리."""

from __future__ import annotations

import pandas as pd

from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer


def build_features(
    symbol: str, data_loader: DataLoader, force_rebuild: bool = False
) -> pd.DataFrame:
    """features_v{N}.parquet 있으면 로드, 없으면 transform() 후 저장.

    실시간 트레이딩은 이 래퍼를 건너뛰고 FeatureEngineer().transform()을 직접 호출한다.
    """
    version = FeatureEngineer.FEATURE_SCHEMA_VERSION
    cache_path = data_loader.processed_data_dir / symbol / f"features_v{version}.parquet"

    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path, engine="pyarrow")

    minute_df = data_loader.load_raw_parquet_all(symbol, "1m")
    features = FeatureEngineer().transform(minute_df)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".parquet.tmp")
    features.to_parquet(tmp, engine="pyarrow")
    tmp.replace(cache_path)  # atomic
    return features
