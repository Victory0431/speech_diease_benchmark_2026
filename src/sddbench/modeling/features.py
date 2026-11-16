"""Load frozen feature vectors from the feature cache.

The model layer never decodes audio and never recomputes features. It reads
`index.csv` + `shard_*.npz`, checks the feature hash and one-window-per-sample
constraint, and returns a dense float32 matrix in manifest order. It only knows
`feature_type`/`recipe_id`/`dim`, never a specific transform.

Since schema 2.0 every row also carries `dataset_id`, `source_sha256` and
`processed_sample_rate`. The loader requires the caller to pass the dataset and
the current manifest's source hashes, so a vector is only reused when it provably
belongs to the same dataset, the same recording content and the same processed
rate. A relative path that exists in two datasets can never cross-load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.registry import feature_cache_dir, feature_hash, feature_key

CACHE_SCHEMA_VERSION = "2.0"


class FeatureLoadError(RuntimeError):
    pass


def cache_dir(workspace_root: Path, feature_cfg: dict, sample_rate: int,
              view_fingerprint: str | None = None) -> Path:
    return feature_cache_dir(workspace_root, feature_cfg, sample_rate,
                             view_fingerprint=view_fingerprint)


def _resolve_cache(workspace_root: Path, feature_cfg: dict, sample_ids: list[str],
                   processed_sample_rate: int, *,
                   dataset_id: str,
                   source_hashes: dict[str, str] | None = None,
                   view_fingerprint: str | None = None) -> tuple[dict, pd.DataFrame | None]:
    """Identity + coverage of the cache, without reading shards.

    Returns (info, part). `info["status"]` is `ready`, `missing`, `legacy`,
    `incompatible` or `partial`; `info["reason"]` explains anything non-ready.
    The trainer (`load_features`) and the read-only plan (web/CLI) share this,
    so the browser cannot claim readiness that the loader would refuse.
    """
    if not str(dataset_id or "").strip():
        raise FeatureLoadError("load_features needs a dataset_id")
    dataset_id = str(dataset_id).strip()
    directory = cache_dir(workspace_root, feature_cfg, processed_sample_rate,
                          view_fingerprint)
    meta_file = directory / "meta.json"
    index_file = directory / "index.csv"
    expected_hash = feature_hash(feature_cfg, processed_sample_rate,
                                 view_fingerprint=view_fingerprint)
    info = {
        "cache_dir": str(directory),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "feature_key": feature_key(feature_cfg, processed_sample_rate,
                                   view_fingerprint=view_fingerprint),
        "feature_hash": expected_hash,
        "view_fingerprint": view_fingerprint,
        "sample_rate": int(processed_sample_rate),
        "dim": None,
        "n_indexed": 0,
        "n_missing": len(sample_ids),
        "missing": [],
        "status": "missing",
        "reason": None,
    }

    def _stop(status: str, reason: str) -> tuple[dict, None]:
        info.update(status=status, reason=reason)
        return info, None

    if not meta_file.is_file() or not index_file.is_file():
        return _stop("missing",
                     f"feature cache not found: {directory}; "
                     "run `sddbench extract-features` first")
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _stop("incompatible", f"feature cache {directory} has an invalid meta.json")
    info["dim"] = meta.get("dim")
    info["feature_key"] = meta.get("feature_key") or info["feature_key"]
    info["feature_type"] = meta.get("feature_type")
    info["recipe_id"] = meta.get("recipe_id")
    if str(meta.get("schema_version") or "") != CACHE_SCHEMA_VERSION:
        return _stop(
            "legacy",
            f"feature cache {directory} uses legacy schema "
            f"{meta.get('schema_version')!r}; rebuild it with `sddbench "
            f"extract-features --dataset {dataset_id} ...` (old rows lack dataset/"
            "source identity and cannot be reused)")
    if meta.get("status") != "completed":
        return _stop("partial",
                     f"feature cache {directory} status={meta.get('status')!r}; "
                     "wait or recompute")
    if meta.get("feature_hash") != expected_hash:
        return _stop("incompatible",
                     "feature cache hash does not match the experiment recipe")
    if int(meta.get("sample_rate", processed_sample_rate)) != int(processed_sample_rate):
        return _stop("incompatible", "feature cache sample rate mismatch")
    if meta.get("view_fingerprint") != view_fingerprint:
        return _stop("incompatible", "feature cache view recipe mismatch")

    index = pd.read_csv(index_file, dtype={"source_sha256": str})
    missing_cols = [c for c in ("dataset_id", "sample_id", "source_sha256",
                                "processed_sample_rate", "view_hash")
                    if c not in index.columns]
    if missing_cols:
        return _stop(
            "incompatible",
            f"feature cache {directory} index lacks identity columns {missing_cols}; "
            "rebuild it instead of reusing unbound vectors")
    needed = set(sample_ids)
    matched = index[index["sample_id"].isin(needed)].copy()
    # Scope by dataset before any duplicate/missing check: the same relative path
    # may legitimately exist in two datasets inside one shared cache.
    part = matched[matched["dataset_id"].astype(str) == dataset_id].copy()
    foreign = sorted(set(matched["sample_id"]) - set(part["sample_id"]))
    if foreign:
        return _stop(
            "incompatible",
            f"feature cache {directory} only has rows for another dataset for "
            f"{foreign[:5]}; refusing cross-dataset reuse")
    if part["window_id"].duplicated().any():
        return _stop("incompatible", "cache index has duplicate window_id rows")
    duplicated = part[part.duplicated("sample_id", keep=False)]["sample_id"].unique().tolist()
    if duplicated:
        return _stop(
            "incompatible",
            "multiple windows per sample are not supported by the MLP baseline yet: "
            f"{duplicated[:5]}")
    missing = [s for s in sample_ids if s not in set(part["sample_id"])]
    info["n_indexed"] = int(len(part))
    info["n_missing"] = len(missing)
    info["missing"] = missing[:20]
    if missing:
        return _stop(
            "partial",
            f"{len(missing)} samples missing from feature cache {directory}: {missing[:5]}")
    if part["feature_hash"].nunique() != 1:
        return _stop("incompatible", "cache index mixes multiple feature hashes")
    rates = part["processed_sample_rate"].astype(int).unique().tolist()
    if rates != [int(processed_sample_rate)]:
        return _stop(
            "incompatible",
            f"feature cache rows claim processed rates {rates}, "
            f"experiment uses {processed_sample_rate}")
    if source_hashes is not None:
        mismatched = []
        for sample_id in sample_ids:
            expected_source = str(source_hashes.get(sample_id) or "")
            row = part.loc[part["sample_id"] == sample_id].iloc[0]
            actual_source = str(row["source_sha256"] or "")
            if expected_source != actual_source:
                mismatched.append(sample_id)
        if mismatched:
            return _stop(
                "incompatible",
                "cached vectors came from different source content than the current "
                f"manifest for {len(mismatched)} sample(s) {mismatched[:5]}; "
                "rebuild features for the changed originals")
    info["status"] = "ready"
    info["n_indexed"] = int(len(part))
    part = part.set_index("sample_id").loc[sample_ids]
    return info, part


def inspect_cache(workspace_root: Path, feature_cfg: dict, sample_ids: list[str],
                  processed_sample_rate: int, *,
                  dataset_id: str,
                  source_hashes: dict[str, str] | None = None,
                  view_fingerprint: str | None = None) -> dict:
    """Read-only cache identity/coverage facts for the shared execution plan."""
    info, _ = _resolve_cache(workspace_root, feature_cfg, sample_ids,
                             processed_sample_rate, dataset_id=dataset_id,
                             source_hashes=source_hashes,
                             view_fingerprint=view_fingerprint)
    return info


def load_features(workspace_root: Path, feature_cfg: dict, sample_ids: list[str],
                  processed_sample_rate: int,
                  *,
                  dataset_id: str,
                  source_hashes: dict[str, str] | None = None,
                  view_fingerprint: str | None = None) -> tuple[np.ndarray, dict]:
    """Return one vector per sample, bound to dataset + source content + view.

    `processed_sample_rate` is the rate the frontend actually produced after
    resampling (not the original Hz stored in the manifest). `source_hashes`
    maps sample_id -> sha256 of the original file; rows whose cached source hash
    differs are refused.
    """
    info, part = _resolve_cache(workspace_root, feature_cfg, sample_ids,
                                processed_sample_rate, dataset_id=dataset_id,
                                source_hashes=source_hashes,
                                view_fingerprint=view_fingerprint)
    if info["status"] != "ready" or part is None:
        raise FeatureLoadError(str(info.get("reason") or "feature cache is not usable"))

    directory = Path(info["cache_dir"])
    dim = int(info["dim"])
    rows_by_shard: dict[str, list[tuple[int, int]]] = {}
    for row, (_, item) in enumerate(part.iterrows()):
        rows_by_shard.setdefault(item["shard_file"], []).append((row, int(item["row_index"])))
    matrix = np.empty((len(part), dim), dtype=np.float32)
    for shard, pairs in rows_by_shard.items():
        with np.load(directory / shard, allow_pickle=False) as data:
            x = data["x"]
            for out_row, in_row in pairs:
                matrix[out_row] = x[in_row]
    if not np.isfinite(matrix).all():
        raise FeatureLoadError("feature matrix contains NaN/Inf")
    result = {
        **info,
        "status": "ready",
        "n_samples": len(matrix),
        # Ordered like `sample_ids` / the manifest: the trainer must report the
        # real window identity and the actually valid (non-padding) duration.
        "window_ids": part["window_id"].astype(str).tolist(),
        "valid_frames": part["valid_frames"].astype(int).tolist(),
    }
    return matrix, result
