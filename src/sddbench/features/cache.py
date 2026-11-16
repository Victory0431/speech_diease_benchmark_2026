"""Sharded, resumable feature cache (meta / index.csv / shard_*.npz).

Layout for one feature key (type + recipe hash + view fingerprint):
  <base>/cache/features/<feature_key>/meta.json
  <base>/cache/features/<feature_key>/index.csv
  <base>/cache/features/<feature_key>/shard_000.npz   (window_ids[N], x[N,dim])

Reuse is keyed by (window_id, view_hash, feature_hash). A changed crop/Hz gives a
different window_id/view_hash; a changed feature recipe gives a different
feature_key. Deleting the directory is a supported way to force recomputation.

Schema 2.0 binds every row to its scientific identity: `dataset_id`,
`sample_id`, `source_sha256` and the actually processed sample rate. Identity is
stored per row because one cache directory may be shared by several datasets.
Schema 1.0 indexes cannot prove which originals/rows they belong to, so they are
refused (never silently reused, never auto-deleted) with an explicit rebuild
instruction.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import (
    FeatureConfigError,
    feature_cache_dir,
    feature_dim,
    feature_hash,
    feature_key,
    implementation_info,
    recipe_id,
)

INDEX_SCHEMA_VERSION = "2.0"
IDENTITY_COLUMNS = ["dataset_id", "source_sha256", "processed_sample_rate"]
INDEX_COLUMNS = [
    "window_id", *IDENTITY_COLUMNS, "sample_id", "shard_file", "row_index",
    "valid_frames", "view_hash", "feature_type", "recipe_id", "feature_hash", "dim",
]
_MAX_SHARD_ROWS = 2048

_LEGACY_REBUILD = (
    "this cache records no dataset/source/processed-rate identity; rebuild it "
    "explicitly with `sddbench extract-features --dataset <id> --source <scan_id> "
    "--preprocess <yaml> --features <yaml>` (the directory is left untouched so "
    "old numbers stay reproducible; delete it yourself only when disk is needed)")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


class FeatureCache:
    """One feature-key directory; appends new vectors and reuses existing rows."""

    def __init__(self, base_root: Path, cfg: dict, sample_rate: int, *,
                 cache_mode: str, shard_size: int = _MAX_SHARD_ROWS,
                 impl: dict | None = None, view_fingerprint: str | None = None):
        self.cfg = cfg
        self.sample_rate = int(sample_rate)
        self.impl = impl or implementation_info(cfg)
        self.cache_mode = cache_mode
        self.shard_size = int(shard_size)
        self.view_fingerprint = view_fingerprint
        if self.shard_size < 1:
            raise FeatureConfigError(f"shard_size must be positive, got {shard_size!r}")
        self.feature_hash = feature_hash(cfg, self.sample_rate, self.impl,
                                         view_fingerprint)
        self.feature_key = feature_key(cfg, self.sample_rate, view_fingerprint)
        self.recipe_id = recipe_id(cfg)
        self.path = feature_cache_dir(base_root, cfg, self.sample_rate,
                                      view_fingerprint=view_fingerprint)
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path / "meta.json"
        self.index_path = self.path / "index.csv"

        self.dim = feature_dim(cfg)
        self._meta = self._load_meta()
        self._index_rows = self._load_index()
        self._existing = {r["window_id"]: r for r in self._index_rows}
        self._new_rows: list[dict] = []
        self._buffer: list[dict] = []
        self._next_shard = self._detect_next_shard()
        if not self._index_rows and self._next_shard > 0:
            raise FeatureConfigError(
                f"cache {self.path} has shard files but no index.csv (interrupted run); "
                "delete this directory to recompute instead of appending duplicates")
        self._created_at = self._meta.get("created_at") or _now()
        if not self._meta:
            self._write_meta("initialized")

    # ----------------------------------------------------------- load / meta
    def _load_meta(self) -> dict:
        if self.index_path.is_file() and not self.meta_path.is_file():
            raise FeatureConfigError(
                f"cache {self.path} has index.csv but no meta.json: {_LEGACY_REBUILD}")
        if not self.meta_path.is_file():
            return {}
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if str(meta.get("schema_version") or "") != INDEX_SCHEMA_VERSION:
            raise FeatureConfigError(
                f"cache {self.path} uses index schema "
                f"{meta.get('schema_version')!r} (expected {INDEX_SCHEMA_VERSION}); "
                f"{_LEGACY_REBUILD}")
        if meta.get("feature_hash") != self.feature_hash:
            raise FeatureConfigError(
                f"cache {self.path} belongs to feature_hash {meta.get('feature_hash')}, "
                f"requested {self.feature_hash}")
        if int(meta.get("dim", self.dim)) != self.dim:
            raise FeatureConfigError(
                f"cache {self.path} dim={meta.get('dim')} != requested {self.dim}")
        if meta.get("view_fingerprint") != self.view_fingerprint:
            raise FeatureConfigError(
                f"cache {self.path} view_fingerprint="
                f"{meta.get('view_fingerprint')} != requested {self.view_fingerprint}")
        return meta

    def _load_index(self) -> list[dict]:
        if not self.index_path.is_file():
            return []
        df = pd.read_csv(self.index_path, dtype=str)
        missing = [c for c in INDEX_COLUMNS if c not in df.columns]
        if missing:
            raise FeatureConfigError(
                f"cache index missing identity columns {missing}: {self.index_path}; "
                f"{_LEGACY_REBUILD}")
        rows = df.to_dict("records")
        for r in rows:
            r["row_index"] = int(r["row_index"])
            r["valid_frames"] = int(r["valid_frames"])
            r["processed_sample_rate"] = int(r["processed_sample_rate"])
            r["dim"] = int(r["dim"])
        return rows

    def _detect_next_shard(self) -> int:
        indices = []
        for p in self.path.glob("shard_*.npz"):
            try:
                indices.append(int(p.stem.split("_", 1)[1]))
            except ValueError:
                continue
        return (max(indices) + 1) if indices else 0

    def _write_meta(self, status: str) -> None:
        meta = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "feature_key": self.feature_key,
            "feature_hash": self.feature_hash,
            "feature_type": self.cfg["feature_type"],
            "recipe_id": self.recipe_id,
            "status": status,
            "cache_mode": self.cache_mode,
            "view_fingerprint": self.view_fingerprint,
            "feature_config": self.cfg,
            "implementation": self.impl,
            "sample_rate": self.sample_rate,
            "dim": self.dim,
            "shard_size": self.shard_size,
            "n_items": len(self._existing) + len(self._new_rows),
            "n_shards": self._next_shard,
            "created_at": self._created_at,
            "updated_at": _now(),
            "storage": {
                "shards": "shard_%03d.npz",
                "index": "index.csv",
                "dtype": "float32",
                "window_ids": "unicode",
                "allow_pickle": False,
            },
        }
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.meta_path)
        self._meta = meta

    # ----------------------------------------------------------- public API
    def has(self, window_id: str) -> bool:
        return window_id in self._existing

    def existing(self, window_id: str) -> dict | None:
        return self._existing.get(window_id)

    @property
    def n_existing(self) -> int:
        return len(self._existing)

    def add(self, window_id: str, sample_id: str, vector: np.ndarray,
            valid_frames: int, view_hash: str, *, dataset_id: str,
            source_sha256: str | None = None,
            processed_sample_rate: int | None = None) -> bool:
        """Append one vector. Returns False when the window is already cached.

        `dataset_id` and `source_sha256` are mandatory identity inputs: a cached
        vector is useless unless we can prove which dataset/recording it came
        from. `processed_sample_rate` defaults to the cache's own rate.
        """
        if not str(dataset_id or "").strip():
            raise FeatureConfigError("feature cache rows need a non-empty dataset_id")
        if window_id in self._existing or any(b["window_id"] == window_id for b in self._buffer):
            return False
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise FeatureConfigError(f"vector dim {vec.shape[0]} != cached dim {self.dim}")
        if not np.isfinite(vec).all():
            raise FeatureConfigError("refusing to cache non-finite feature values")
        processed_sr = int(processed_sample_rate if processed_sample_rate is not None
                           else self.sample_rate)
        if processed_sr != self.sample_rate:
            raise FeatureConfigError(
                f"row processed_sample_rate={processed_sr} does not match cache "
                f"rate {self.sample_rate}; use one cache per processed rate")
        self._buffer.append({
            "window_id": window_id, "dataset_id": str(dataset_id).strip(),
            "sample_id": sample_id, "source_sha256": str(source_sha256 or ""),
            "processed_sample_rate": processed_sr, "vec": vec,
            "valid_frames": int(valid_frames), "view_hash": str(view_hash),
        })
        if len(self._buffer) >= self.shard_size:
            self._flush()
        return True

    def _flush(self) -> None:
        if not self._buffer:
            return
        shard_name = f"shard_{self._next_shard:03d}.npz"
        ids = np.array([b["window_id"] for b in self._buffer], dtype=str)
        matrix = np.stack([b["vec"] for b in self._buffer]).astype(np.float32)
        tmp = self.path / (shard_name + ".tmp.npz")
        np.savez(tmp, window_ids=ids, x=matrix)
        os.replace(tmp, self.path / shard_name)
        for row_index, item in enumerate(self._buffer):
            self._new_rows.append({
                "window_id": item["window_id"], "dataset_id": item["dataset_id"],
                "source_sha256": item["source_sha256"],
                "processed_sample_rate": item["processed_sample_rate"],
                "sample_id": item["sample_id"],
                "shard_file": shard_name, "row_index": row_index,
                "valid_frames": item["valid_frames"], "view_hash": item["view_hash"],
                "feature_type": self.cfg["feature_type"],
                "recipe_id": self.recipe_id,
                "feature_hash": self.feature_hash,
                "dim": self.dim,
            })
        self._next_shard += 1
        self._buffer = []
        self._write_meta("running")

    def finalize(self) -> dict:
        """Flush buffers, atomically rewrite index.csv and mark the cache done."""
        self._flush()
        rows = self._index_rows + self._new_rows
        df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
        tmp = self.index_path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.index_path)
        self._index_rows = rows
        self._existing = {r["window_id"]: r for r in rows}
        self._new_rows = []
        self._write_meta("completed")
        return self.summary()

    def summary(self) -> dict:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "feature_key": self.feature_key,
            "feature_hash": self.feature_hash,
            "feature_type": self.cfg["feature_type"],
            "recipe_id": self.recipe_id,
            "cache_dir": str(self.path),
            "cache_mode": self.cache_mode,
            "view_fingerprint": self.view_fingerprint,
            "sample_rate": self.sample_rate,
            "dim": self.dim,
            "n_items": len(self._existing) + len(self._new_rows),
            "n_shards": self._next_shard,
            "status": self._meta.get("status"),
        }


class FeatureCacheManager:
    """One cache per actual sample rate (native mixed mode stays explicit)."""

    def __init__(self, base_root: Path, cfg: dict, *, cache_mode: str,
                 shard_size: int = _MAX_SHARD_ROWS,
                 view_fingerprint: str | None = None):
        self.base_root = Path(base_root)
        self.cfg = cfg
        self.cache_mode = cache_mode
        self.shard_size = shard_size
        self.view_fingerprint = view_fingerprint
        self._impl = implementation_info(cfg)
        self._caches: dict[int, FeatureCache] = {}

    def cache_for(self, sample_rate: int) -> FeatureCache:
        sr = int(sample_rate)
        cache = self._caches.get(sr)
        if cache is None:
            cache = FeatureCache(self.base_root, self.cfg, sr,
                                 cache_mode=self.cache_mode,
                                 shard_size=self.shard_size, impl=self._impl,
                                 view_fingerprint=self.view_fingerprint)
            self._caches[sr] = cache
        return cache

    def finalize(self) -> dict:
        caches = [c.finalize() for c in sorted(self._caches.values(),
                                               key=lambda c: c.sample_rate)]
        return {"caches": caches,
                "n_items": sum(c["n_items"] for c in caches),
                "n_shards": sum(c["n_shards"] for c in caches)}
