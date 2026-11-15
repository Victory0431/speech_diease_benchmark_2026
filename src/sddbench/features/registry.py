"""Feature recipe registry: the only place that knows feature types.

A feature recipe owns everything that defines the vector: its type name, the
frame geometry (frame_ms/hop_ms/n_fft), the transform parameters, the output
dimension and the hash. Nothing outside `features/` may assume MFCC.

Adding a new feature type (e.g. log-mel) means:
  1. add a module with the same five functions as `mfcc.py`;
  2. register one entry below.

Cache paths, index columns, run metadata and the model input dimension all read
`feature_type` / `recipe_id` / `dim` from here, so no other layer changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import mfcc
from .mfcc import FeatureConfigError

__all__ = [
    "FeatureConfigError", "Recipe", "RECIPES",
    "feature_type", "recipe", "normalize_feature_config",
    "recipe_id", "feature_dim", "implementation_info",
    "feature_hash", "feature_key", "feature_cache_dir", "extract",
]


@dataclass(frozen=True)
class Recipe:
    feature_type: str
    normalize: Callable[[dict], dict]
    dim: Callable[[dict], int]
    hash: Callable[..., str]
    impl: Callable[[], dict]
    extract: Callable[..., object]


RECIPES: dict[str, Recipe] = {
    "mfcc": Recipe(
        feature_type="mfcc",
        normalize=mfcc.normalize_feature_config,
        dim=mfcc.dim,
        hash=mfcc.feature_hash,
        impl=mfcc.implementation_info,
        extract=mfcc.extract_mfcc,
    ),
}


def feature_type(cfg: dict) -> str:
    value = str((cfg or {}).get("feature_type") or "").strip()
    if not value:
        raise FeatureConfigError(
            "feature config needs an explicit feature_type (e.g. mfcc); "
            "implicit types make caches and result packs unidentifiable")
    return value


def recipe(cfg: dict) -> Recipe:
    ftype = feature_type(cfg)
    if ftype not in RECIPES:
        raise FeatureConfigError(
            f"unknown feature_type {ftype!r}; registered: {sorted(RECIPES)}")
    return RECIPES[ftype]


def normalize_feature_config(raw: dict) -> dict:
    cfg = dict(raw or {})
    return recipe(cfg).normalize(cfg)


def recipe_id(cfg: dict) -> str:
    return str(cfg.get("name") or feature_type(cfg))


def feature_dim(cfg: dict) -> int:
    return int(recipe(cfg).dim(cfg))


def implementation_info(cfg: dict) -> dict:
    return recipe(cfg).impl()


def feature_hash(cfg: dict, sample_rate: int, impl: dict | None = None,
                 view_fingerprint: str | None = None) -> str:
    return recipe(cfg).hash(cfg, int(sample_rate), impl, view_fingerprint)


def feature_key(cfg: dict, sample_rate: int,
                view_fingerprint: str | None = None) -> str:
    return f"{feature_type(cfg)}_{feature_hash(cfg, sample_rate, view_fingerprint=view_fingerprint)[:16]}"


def feature_cache_dir(base_root: Path | str, cfg: dict, sample_rate: int,
                      view_fingerprint: str | None = None) -> Path:
    """<base>/cache/features/<feature_type>_<hash16> -- one layout for all types."""
    key = feature_key(cfg, sample_rate, view_fingerprint=view_fingerprint)
    return Path(base_root) / "cache" / "features" / key


def extract(waveform: np.ndarray, sample_rate: int, cfg: dict,
            impl: dict | None = None):
    return recipe(cfg).extract(waveform, int(sample_rate), cfg, impl)
