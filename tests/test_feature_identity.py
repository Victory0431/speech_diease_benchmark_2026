"""Layer boundaries: preprocessing never owns frame geometry; features are typed."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sddbench.data.windows import PreprocessConfigError, normalize_preprocess
from sddbench.features.cache import FeatureCache
from sddbench.features.registry import (
    FeatureConfigError,
    feature_dim,
    feature_key,
    normalize_feature_config,
    recipe_id,
)


def test_preprocessing_rejects_frame_geometry():
    with pytest.raises(PreprocessConfigError, match="feature recipe"):
        normalize_preprocess({"frame_ms": 25.0})
    with pytest.raises(PreprocessConfigError, match="feature recipe"):
        normalize_preprocess({"hop_ms": 10.0})


def test_feature_type_must_be_explicit():
    with pytest.raises(FeatureConfigError, match="feature_type"):
        normalize_feature_config({"name": "mfcc_stats", "n_mfcc": 20})


def test_key_dim_and_cache_are_derived_from_the_recipe(tmp_path):
    cfg = normalize_feature_config(
        {"feature_type": "mfcc", "name": "quick", "n_mfcc": 13})
    assert feature_dim(cfg) == 26
    assert recipe_id(cfg) == "quick"

    key = feature_key(cfg, 16000)
    assert key.startswith("mfcc_"), "cache directory must show the feature type"

    cache = FeatureCache(tmp_path, cfg, 16000, cache_mode="shared")
    assert cache.path.name == key
    assert cache.dim == 26
    cache.add("w1", "s1", np.zeros(26, dtype=np.float32), 100, "viewhash",
              dataset_id="demo", source_sha256="sha1")
    cache.finalize()

    meta = json.loads((cache.path / "meta.json").read_text(encoding="utf-8"))
    assert meta["feature_type"] == "mfcc" and meta["recipe_id"] == "quick"
    index = pd.read_csv(cache.path / "index.csv")
    assert {"feature_type", "recipe_id", "dim"} <= set(index.columns)
    assert "n_mfcc" not in index.columns
