import sys
from pathlib import Path

_ENV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_ENV_SRC) not in sys.path:
    sys.path.insert(0, str(_ENV_SRC))
