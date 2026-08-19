"""Feature parquet cache with raw-source provenance validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from data.data_loader import DataLoader
from data.defect_days import QUALITY_POLICY_VERSION
from data.feature_engineer import FeatureEngineer


_MANIFEST_FORMAT_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024


def _raw_source_files(data_loader: DataLoader, symbol: str) -> list[Path]:
    return sorted((data_loader.raw_data_dir / symbol / "1m").glob("*.parquet"))


def _source_fingerprint(files: list[Path], raw_root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash raw parquet bytes and their stable relative names.

    Content hashing costs a sequential read but prevents a stale feature cache
    when a repaired partition is replaced with the same name or file size.
    """
    digest = hashlib.sha256()
    sources: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(raw_root).as_posix()
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                file_digest.update(chunk)
        size = path.stat().st_size
        content_hash = file_digest.hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\0")
        sources.append(
            {"path": relative, "size_bytes": size, "sha256": content_hash}
        )
    return digest.hexdigest(), sources


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".manifest.json")


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _manifest_matches(
    manifest: dict[str, Any] | None,
    *,
    symbol: str,
    schema_version: int,
    source_fingerprint: str,
) -> bool:
    return bool(
        manifest
        and manifest.get("manifest_format_version") == _MANIFEST_FORMAT_VERSION
        and manifest.get("symbol") == symbol
        and manifest.get("feature_schema_version") == schema_version
        and manifest.get("quality_policy_version") == QUALITY_POLICY_VERSION
        and manifest.get("source_fingerprint") == source_fingerprint
    )


def build_features(
    symbol: str, data_loader: DataLoader, force_rebuild: bool = False
) -> pd.DataFrame:
    """Load a valid feature cache or rebuild it from raw one-minute parquet.

    A sidecar manifest binds the cache to the exact raw parquet contents and
    the quality-policy version. If raw data is deliberately unavailable but a
    cache exists, the cache remains usable for backward-compatible offline
    workflows; it simply cannot be revalidated in that situation.
    """
    version = FeatureEngineer.FEATURE_SCHEMA_VERSION
    cache_path = data_loader.processed_data_dir / symbol / f"features_v{version}.parquet"
    manifest_path = _manifest_path(cache_path)
    raw_files = _raw_source_files(data_loader, symbol)

    fingerprint: str | None = None
    sources: list[dict[str, Any]] = []
    if raw_files:
        fingerprint, sources = _source_fingerprint(raw_files, data_loader.raw_data_dir)

    if cache_path.exists() and not force_rebuild:
        if not raw_files:
            return pd.read_parquet(cache_path, engine="pyarrow")
        if _manifest_matches(
            _read_manifest(manifest_path),
            symbol=symbol,
            schema_version=version,
            source_fingerprint=fingerprint,
        ):
            return pd.read_parquet(cache_path, engine="pyarrow")

    minute_df = data_loader.load_raw_parquet_all(symbol, "1m")
    features = FeatureEngineer().transform(minute_df)

    # Never bless a cache built while a collector was replacing its source.
    # The caller can retry once the atomic raw-partition write has completed.
    current_files = _raw_source_files(data_loader, symbol)
    current_fingerprint, current_sources = _source_fingerprint(
        current_files, data_loader.raw_data_dir
    )
    if fingerprint != current_fingerprint:
        raise RuntimeError(
            f"raw one-minute partitions changed while building features for {symbol}"
        )
    sources = current_sources

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".parquet.tmp")
    try:
        features.to_parquet(tmp, engine="pyarrow")
        tmp.replace(cache_path)
    finally:
        if tmp.exists():
            tmp.unlink()

    _write_manifest(
        manifest_path,
        {
            "manifest_format_version": _MANIFEST_FORMAT_VERSION,
            "symbol": symbol,
            "feature_schema_version": version,
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "source_fingerprint": fingerprint,
            "sources": sources,
        },
    )
    return features
