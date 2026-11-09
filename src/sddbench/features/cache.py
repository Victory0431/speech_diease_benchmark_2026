"""Sharded, resumable feature cache (meta / index.csv / shard_*.npz).

Layout for one feature key:
  <base>/<feature_key>/meta.json
  <base>/<feature_key>/index.csv
  <base>/<feature_key>/shard_000.npz   (window_ids[N], x[N,dim])

Reuse is keyed by (window_id, view_hash, feature_hash). A changed crop/Hz gives a
different window_id/view_hash; a changed MFCC recipe gives a different
feature_key. Deleting the directory is a supported way to force recomputation.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .mfcc import FeatureConfigError, feature_hash, feature_key, implementation_info

INDEX_COLUMNS = [
    "window_id", "sample_id", "shard_file", "row_index", "valid_frames",
    "view_hash", "feature_hash", "n_mfcc", "dim",
]
_MAX_SHARD_ROWS = 2048


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


class FeatureCache:
    """One feature-key directory; appends new vectors and reuses existing rows."""

    def __init__(self, base_root: Path, cfg: dict, sample_rate: int, *,
                 cache_mode: str, shard_size: int = _MAX_SHARD_ROWS,
                 impl: dict | None = None):
        self.cfg = cfg
        self.sample_rate = int(sample_rate)
        self.impl = impl or implementation_info()
        self.cache_mode = cache_mode
        self.shard_size = int(shard_size)
        if self.shard_size < 1:
            raise FeatureConfigError(f"shard_size must be positive, got {shard_size!r}")
        self.feature_hash = feature_hash(cfg, self.sample_rate, self.impl)
        self.feature_key = feature_key(cfg, self.sample_rate)
        self.path = Path(base_root) / self.feature_key
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path / "meta.json"
        self.index_path = self.path / "index.csv"

        self.dim = 2 * int(cfg["n_mfcc"])
        self._meta = self._load_meta()
        self._index_rows = self._load_index()
        self._existing = {r["window_id"]: r for r in self._index_rows}
        self._new_rows: list[dict] = []
        self._buffer: list[tuple[str, str, np.ndarray, int, str]] = []
        self._next_shard = self._detect_next_shard()
        self._created_at = self._meta.get("created_at") or _now()
        if not self._meta:
            self._write_meta("initialized")

    # ----------------------------------------------------------- load / meta
    def _load_meta(self) -> dict:
        if not self.meta_path.is_file():
            return {}
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("feature_hash") != self.feature_hash:
            raise FeatureConfigError(
                f"cache {self.path} belongs to feature_hash {meta.get('feature_hash')}, "
                f"requested {self.feature_hash}")
        if int(meta.get("dim", self.dim)) != self.dim:
            raise FeatureConfigError(
                f"cache {self.path} dim={meta.get('dim')} != requested {self.dim}")
        return meta

    def _load_index(self) -> list[dict]:
        if not self.index_path.is_file():
            return []
        df = pd.read_csv(self.index_path, dtype=str)
        missing = [c for c in INDEX_COLUMNS if c not in df.columns]
        if missing:
            raise FeatureConfigError(f"cache index missing columns {missing}: {self.index_path}")
        rows = df.to_dict("records")
        for r in rows:
            r["row_index"] = int(r["row_index"])
            r["valid_frames"] = int(r["valid_frames"])
            r["n_mfcc"] = int(r["n_mfcc"])
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
            "schema_version": "1.0",
            "feature_key": self.feature_key,
            "feature_hash": self.feature_hash,
            "status": status,
            "cache_mode": self.cache_mode,
            "feature_config": self.cfg,
            "implementation": self.impl,
            "sample_rate": self.sample_rate,
            "n_mfcc": int(self.cfg["n_mfcc"]),
            "dim": self.dim,
            "stats": list(self.cfg["stats"]),
            "std_ddof": int(self.cfg["std_ddof"]),
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
            valid_frames: int, view_hash: str) -> bool:
        """Append one vector. Returns False when the window is already cached."""
        if window_id in self._existing or any(b[0] == window_id for b in self._buffer):
            return False
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise FeatureConfigError(f"vector dim {vec.shape[0]} != cached dim {self.dim}")
        if not np.isfinite(vec).all():
            raise FeatureConfigError("refusing to cache non-finite feature values")
        self._buffer.append((window_id, sample_id, vec, int(valid_frames), str(view_hash)))
        if len(self._buffer) >= self.shard_size:
            self._flush()
        return True

    def _flush(self) -> None:
        if not self._buffer:
            return
        shard_name = f"shard_{self._next_shard:03d}.npz"
        ids = np.array([b[0] for b in self._buffer], dtype=str)
        matrix = np.stack([b[2] for b in self._buffer]).astype(np.float32)
        tmp = self.path / (shard_name + ".tmp.npz")
        np.savez(tmp, window_ids=ids, x=matrix)
        os.replace(tmp, self.path / shard_name)
        for row_index, (window_id, sample_id, _vec, frames, view_hash) in enumerate(self._buffer):
            self._new_rows.append({
                "window_id": window_id, "sample_id": sample_id,
                "shard_file": shard_name, "row_index": row_index,
                "valid_frames": frames, "view_hash": view_hash,
                "feature_hash": self.feature_hash,
                "n_mfcc": int(self.cfg["n_mfcc"]), "dim": self.dim,
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
            "feature_key": self.feature_key,
            "feature_hash": self.feature_hash,
            "cache_dir": str(self.path),
            "cache_mode": self.cache_mode,
            "sample_rate": self.sample_rate,
            "dim": self.dim,
            "n_items": len(self._existing) + len(self._new_rows),
            "n_shards": self._next_shard,
            "status": self._meta.get("status"),
        }


class FeatureCacheManager:
    """One cache per actual sample rate (native mixed mode stays explicit)."""

    def __init__(self, base_root: Path, cfg: dict, *, cache_mode: str,
                 shard_size: int = _MAX_SHARD_ROWS):
        self.base_root = Path(base_root)
        self.cfg = cfg
        self.cache_mode = cache_mode
        self.shard_size = shard_size
        self._impl = implementation_info()
        self._caches: dict[int, FeatureCache] = {}

    def cache_for(self, sample_rate: int) -> FeatureCache:
        sr = int(sample_rate)
        cache = self._caches.get(sr)
        if cache is None:
            cache = FeatureCache(self.base_root, self.cfg, sr,
                                 cache_mode=self.cache_mode,
                                 shard_size=self.shard_size, impl=self._impl)
            self._caches[sr] = cache
        return cache

    def finalize(self) -> dict:
        caches = [c.finalize() for c in sorted(self._caches.values(),
                                               key=lambda c: c.sample_rate)]
        return {"caches": caches,
                "n_items": sum(c["n_items"] for c in caches),
                "n_shards": sum(c["n_shards"] for c in caches)}
