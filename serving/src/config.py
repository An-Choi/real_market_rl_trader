"""serving.yaml 로더 — 미지 키는 거부(fail-closed)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ServingConfig:
    artifact_dir: Path
    data_dir: Path
    symbols: list
    audit_log_dir: Path
    max_bar_age_minutes: int = 10
    warmup_days: int = 30
    host: str = "127.0.0.1"
    port: int = 8000


_PATH_FIELDS = ("artifact_dir", "data_dir", "audit_log_dir")
# 인증/TLS 없는 스펙 범위 — 로컬 바인딩만 허용 (spec §5)
_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")


def load_serving_config(path: "str | Path") -> ServingConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in dataclasses.fields(ServingConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown serving config keys: {sorted(unknown)}")
    for key in _PATH_FIELDS:
        if key in data:
            data[key] = Path(data[key])
    cfg = ServingConfig(**data)
    if not cfg.symbols or not all(isinstance(s, str) and s for s in cfg.symbols):
        raise ValueError(f"symbols must be a non-empty list of strings: {cfg.symbols!r}")
    if cfg.host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"host must be one of {_ALLOWED_HOSTS} (no auth/TLS in scope): {cfg.host!r}"
        )
    return cfg
