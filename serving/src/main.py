"""uvicorn 엔트리포인트: python3 serving/src/main.py --config serving/configs/serving.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _src in (_ROOT / "serving" / "src", _ROOT / "agent" / "src", _ROOT / "env" / "src"):
    _p = str(_src)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    import uvicorn

    from app import create_app
    from config import load_serving_config
    from market_data import HistoricalParquetProvider
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_ROOT / "serving" / "configs" / "serving.yaml"))
    args = parser.parse_args()

    config = load_serving_config(args.config)
    predictor = Predictor.load(config.artifact_dir)   # 기동 거부는 여기서 raise
    provider = HistoricalParquetProvider(config.data_dir, warmup_days=config.warmup_days)
    app = create_app(config, predictor, provider)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
