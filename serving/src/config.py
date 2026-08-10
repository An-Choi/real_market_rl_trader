"""serving.yaml 로더 — 미지 키는 거부(fail-closed)."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
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
    provider: str = "historical"
    kis_token_cache: Path = field(default_factory=lambda: Path("data/.kis_token.json"))
    kis_rate_limit_sleep: float = 0.5


_PATH_FIELDS = ("artifact_dir", "data_dir", "audit_log_dir", "kis_token_cache")
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
    if (
        not isinstance(cfg.symbols, list)
        or not cfg.symbols
        or not all(isinstance(s, str) and s for s in cfg.symbols)
    ):
        raise ValueError(f"symbols must be a non-empty list of strings: {cfg.symbols!r}")
    if (
        isinstance(cfg.warmup_days, bool)
        or not isinstance(cfg.warmup_days, int)
        or cfg.warmup_days <= 0
    ):
        raise ValueError(f"warmup_days must be a positive int: {cfg.warmup_days!r}")
    if (
        isinstance(cfg.max_bar_age_minutes, bool)
        or not isinstance(cfg.max_bar_age_minutes, int)
        or cfg.max_bar_age_minutes <= 0
    ):
        raise ValueError(
            f"max_bar_age_minutes must be a positive int: {cfg.max_bar_age_minutes!r}"
        )
    if (
        isinstance(cfg.port, bool)
        or not isinstance(cfg.port, int)
        or not 1 <= cfg.port <= 65535
    ):
        raise ValueError(f"port must be an int in 1..65535: {cfg.port!r}")
    if cfg.provider not in ("historical", "live"):
        raise ValueError(f"provider must be 'historical' or 'live': {cfg.provider!r}")
    rate = cfg.kis_rate_limit_sleep
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate < 0
    ):
        raise ValueError(
            f"kis_rate_limit_sleep must be a finite non-negative number: {rate!r}"
        )
    if cfg.host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"host must be one of {_ALLOWED_HOSTS} (no auth/TLS in scope): {cfg.host!r}"
        )
    return cfg
