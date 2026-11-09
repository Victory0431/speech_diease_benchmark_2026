"""Load the frozen MFCC statistics from the D3 sharded cache.

The model layer never decodes audio and never recomputes features. It reads
`index.csv` + `shard_*.npz`, checks the feature hash and one-window-per-sample
constraint, and returns a dense float32 matrix in manifest order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.mfcc import feature_hash


class FeatureLoadError(RuntimeError):
    pass


def cache_dir(workspace_root: Path, feature_cfg: dict, sample_rate: int,
              view_fingerprint: str | None = None) -> Path:
    key = feature_hash(feature_cfg, sample_rate, view_fingerprint=view_fingerprint)
    return (Path(workspace_root) / "cache" / "features" / feature_cfg["name"]
            / f"{feature_cfg['name']}_{key[:16]}")


def load_features(workspace_root: Path, feature_cfg: dict, sample_ids: list[str],
                  sample_rate: int,
                  view_fingerprint: str | None = None) -> tuple[np.ndarray, dict]:
    directory = cache_dir(workspace_root, feature_cfg, sample_rate, view_fingerprint)
    meta_file = directory / "meta.json"
    index_file = directory / "index.csv"
    if not meta_file.is_file() or not index_file.is_file():
        raise FeatureLoadError(
            f"feature cache not found: {directory}; run `sddbench extract-features` first")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("status") != "completed":
        raise FeatureLoadError(
            f"feature cache {directory} status={meta.get('status')!r}; wait or recompute")
    expected_hash = feature_hash(feature_cfg, sample_rate,
                                 view_fingerprint=view_fingerprint)
    if meta.get("feature_hash") != expected_hash:
        raise FeatureLoadError("feature cache hash does not match the experiment recipe")
    if int(meta.get("sample_rate", sample_rate)) != int(sample_rate):
        raise FeatureLoadError("feature cache sample rate mismatch")
    if meta.get("view_fingerprint") != view_fingerprint:
        raise FeatureLoadError("feature cache view recipe mismatch")

    index = pd.read_csv(index_file)
    needed = set(sample_ids)
    part = index[index["sample_id"].isin(needed)].copy()
    duplicated = part[part.duplicated("sample_id", keep=False)]["sample_id"].unique().tolist()
    if duplicated:
        raise FeatureLoadError(
            "multiple windows per sample are not supported by the MLP baseline yet: "
            f"{duplicated[:5]}")
    missing = [s for s in sample_ids if s not in set(part["sample_id"])]
    if missing:
        raise FeatureLoadError(
            f"{len(missing)} samples missing from feature cache {directory}: {missing[:5]}")
    if part["feature_hash"].nunique() != 1:
        raise FeatureLoadError("cache index mixes multiple feature hashes")
    part = part.set_index("sample_id").loc[sample_ids]

    dim = int(meta["dim"])
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
    info = {
        "cache_dir": str(directory),
        "feature_key": meta.get("feature_key"),
        "feature_hash": expected_hash,
        "view_fingerprint": view_fingerprint,
        "sample_rate": int(sample_rate),
        "dim": dim,
        "n_samples": len(matrix),
    }
    return matrix, info
