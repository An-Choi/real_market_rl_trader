import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _src in (_ROOT / "agent" / "src", _ROOT / "env" / "src"):
    _p = str(_src)
    if _p not in sys.path:
        sys.path.insert(0, _p)
