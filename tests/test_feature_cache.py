"""D3: MFCC formula, padding invariance, cache reuse and invalidation."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from sddbench.cli import main
from sddbench.data.decoders import file_sha256
from sddbench.data.schema import (
    COL_CATEGORY,
    COL_DECODE_STATUS,
    COL_DIRECTORY_GROUP,
    COL_DURATION_DECODED_SEC,
    COL_N_SAMPLES,
    COL_RELATIVE_PATH,
    COL_SAMPLE_RATE,
    COL_SHA256,
    COL_SPEAKER_TOKEN,
)
from sddbench.data.windows import build_view_index, normalize_preprocess
from sddbench.features.cache import FeatureCache
from sddbench.features.mfcc import (
    FeatureError,
    extract_mfcc,
    feature_hash,
    normalize_feature_config,
)

REPO = Path(__file__).resolve().parents[1]
FEAT_YAML = REPO / "configs" / "features" / "mfcc20_stats.yaml"
PRE_YAML = REPO / "configs" / "preprocessing" / "uaspeech_native_smoke.yaml"


def _feat(**over) -> dict:
    raw = yaml.safe_load(FEAT_YAML.read_text(encoding="utf-8"))
    raw.update(over)
    return normalize_feature_config(raw)


def _pre(**over) -> dict:
    raw = yaml.safe_load(PRE_YAML.read_text(encoding="utf-8"))
    raw.update(over)
    return normalize_preprocess(raw)


def _sine(seconds: float, sr: int = 16000) -> np.ndarray:
    return (0.4 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * sr)) / sr)).astype(np.float32)


def test_mfcc_shape_and_finiteness():
    res = extract_mfcc(_sine(1.0), 16000, _feat())
    assert res.vector.shape == (40,)
    assert res.n_mfcc == 20 and res.n_frames == 1 + (16000 - 512) // 160
    assert np.isfinite(res.vector).all()


def test_n_mfcc_40_gives_80_dims():
    res = extract_mfcc(_sine(1.0), 16000, _feat(n_mfcc=40))
    assert res.vector.shape == (80,)


def test_padding_never_changes_mfcc_stats(tmp_path):
    root = tmp_path / "ds"
    (root / "Disorder").mkdir(parents=True)
    wav = root / "Disorder" / "CF02_B1_C1_M2.wav"
    sf.write(str(wav), _sine(3.0), 16000, subtype="PCM_16")
    rec = {
        COL_RELATIVE_PATH: "Disorder/CF02_B1_C1_M2.wav",
        COL_CATEGORY: "audio_candidate", COL_DECODE_STATUS: "ok",
        COL_SHA256: file_sha256(wav), COL_SAMPLE_RATE: 16000,
        COL_N_SAMPLES: 48000, COL_DURATION_DECODED_SEC: 3.0,
        COL_DIRECTORY_GROUP: "Disorder", COL_SPEAKER_TOKEN: "CF02",
    }
    feat = _feat()
    captured = {}

    def run(cfg, tag):
        vectors = []

        def cb(row, valid, sr):
            vectors.append(extract_mfcc(valid, sr, feat).vector)

        table = build_view_index(root, [rec], cfg, source_scan_id="scan1", on_view=cb)
        assert len(table.rows) == 1
        captured[tag] = (table.rows[0], vectors[0])
        return captured[tag]

    row3, vec3 = run(_pre(observe_seconds=3.0, window_seconds=3.0), "3s")
    row4, vec4 = run(_pre(observe_seconds=4.0, window_seconds=4.0), "4s")
    assert row3["pad_num_samples"] == 0
    assert row4["pad_num_samples"] == 16000
    assert row3["valid_num_samples"] == row4["valid_num_samples"] == 48000
    assert row3["view_hash"] != row4["view_hash"]
    assert np.array_equal(vec3, vec4)  # padding does not enter the MFCC frames


def test_cache_roundtrip_reuse_and_delete(tmp_path):
    cfg = _feat()
    base = tmp_path / "cache"
    cache = FeatureCache(base, cfg, 16000, cache_mode="run")
    vec = np.arange(40, dtype=np.float32)
    for i in range(5):
        assert cache.add(f"w{i}", f"s{i}", vec + i, 100 + i, f"vh{i}") is True
    summary = cache.finalize()
    assert summary["n_items"] == 5 and summary["dim"] == 40
    assert (cache.path / "shard_000.npz").is_file()
    assert (cache.path / "index.csv").is_file()
    meta = json.loads((cache.path / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["dim"] == 40
    index = pd.read_csv(cache.path / "index.csv")
    assert len(index) == 5 and list(index["window_id"]) == [f"w{i}" for i in range(5)]
    data = np.load(cache.path / "shard_000.npz", allow_pickle=False)
    assert data["x"].shape == (5, 40)

    reopened = FeatureCache(base, cfg, 16000, cache_mode="run")
    assert reopened.n_existing == 5
    assert reopened.has("w0")
    assert reopened.add("w0", "s0", vec, 100, "vh0") is False

    # different recipe -> different key/dir, no accidental reuse
    other = FeatureCache(base, _feat(n_mfcc=40), 16000, cache_mode="run")
    assert other.feature_key != cache.feature_key
    assert other.n_existing == 0 and other.dim == 80

    # deleting the cache is the supported way to recompute
    shutil.rmtree(cache.path)
    fresh = FeatureCache(base, cfg, 16000, cache_mode="run")
    assert fresh.n_existing == 0
    assert fresh.add("w0", "s0", vec, 100, "vh0") is True


def test_feature_hash_ignores_labels_but_tracks_recipe_and_hz():
    cfg = _feat()
    h = feature_hash(cfg, 16000)
    assert feature_hash({**cfg, "name": "renamed", "description": "x"}, 16000) == h
    assert feature_hash(_feat(n_mfcc=40), 16000) != h
    assert feature_hash(cfg, 8000) != h


def test_too_short_valid_audio_is_rejected():
    with pytest.raises(FeatureError):
        extract_mfcc(np.zeros(100, dtype=np.float32), 16000, _feat())


def _latest_job(workspace: Path) -> dict:
    jobs = sorted((workspace / "feature_jobs").iterdir(), key=lambda p: p.name)
    return json.loads((jobs[-1] / "job.json").read_text(encoding="utf-8"))


def _run_cli(workspace, paths_file, feat_yaml, limit=4):
    return main([
        "extract-features", "--dataset", "uaspeech", "--source", "scan1",
        "--preprocess", str(PRE_YAML), "--features", str(feat_yaml),
        "--limit", str(limit), "--cache-mode", "shared", "--previews", "0",
        "--paths", str(paths_file),
    ])


def test_cli_computes_features_and_reuses_cache(scan_workspace, tmp_path, monkeypatch):
    workspace, paths_file, _ = scan_workspace
    monkeypatch.chdir(tmp_path)
    assert _run_cli(workspace, paths_file, FEAT_YAML) == 0
    job1 = _latest_job(workspace)
    assert job1["status"] == "completed" and job1["feature"]["dim"] == 40
    n_built = job1["counts"]["built"]
    assert n_built > 0
    assert job1["feature"]["n_new"] == n_built
    cache_dir = Path(job1["outputs"]["cache_dirs"][0])
    shards = sorted(cache_dir.glob("shard_*.npz"))
    x1 = np.concatenate([np.load(p, allow_pickle=False)["x"] for p in shards])
    assert x1.shape == (n_built, 40) and np.isfinite(x1).all()
    index1 = pd.read_csv(cache_dir / "index.csv")
    assert index1["window_id"].is_unique and len(index1) == n_built
    inspection = (Path(job1["job_dir"]) / "inspection.md").read_text(encoding="utf-8")
    assert "MFCC 特征" in inspection and "命中缓存" in inspection

    # second run: same source/view/feature -> full cache hit, same numbers
    assert _run_cli(workspace, paths_file, FEAT_YAML) == 0
    job2 = _latest_job(workspace)
    assert job2["job_id"] != job1["job_id"]
    assert job2["feature"]["n_from_cache"] == n_built
    assert job2["feature"]["n_new"] == 0
    x2 = np.concatenate([np.load(p, allow_pickle=False)["x"]
                         for p in sorted(cache_dir.glob("shard_*.npz"))])
    assert np.array_equal(x1, x2)

    # changing n_mfcc changes the feature key -> new cache, 80 dims
    cfg40 = tmp_path / "mfcc40.yaml"
    raw = yaml.safe_load(FEAT_YAML.read_text(encoding="utf-8"))
    raw.update({"name": "mfcc40_stats", "n_mfcc": 40})
    cfg40.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert _run_cli(workspace, paths_file, cfg40) == 0
    job3 = _latest_job(workspace)
    assert job3["feature"]["dim"] == 80
    assert job3["feature"]["n_new"] == n_built
    assert Path(job3["outputs"]["cache_dirs"][0]) != cache_dir

    # deleting the cache forces recomputation
    shutil.rmtree(workspace / "cache" / "features" / "mfcc_stats")
    assert _run_cli(workspace, paths_file, FEAT_YAML) == 0
    job4 = _latest_job(workspace)
    assert job4["feature"]["n_new"] == n_built
    assert job4["feature"]["n_from_cache"] == 0


def test_cli_dry_run_with_features_does_not_compute(scan_workspace, tmp_path, monkeypatch):
    workspace, paths_file, _ = scan_workspace
    monkeypatch.chdir(tmp_path)
    code = main([
        "extract-features", "--dataset", "uaspeech", "--source", "scan1",
        "--preprocess", str(PRE_YAML), "--features", str(FEAT_YAML),
        "--limit", "2", "--dry-run", "--paths", str(paths_file),
    ])
    assert code == 0
    job = _latest_job(workspace)
    assert job["status"] == "planned" and job["feature"]["planned"] is True
    assert not (workspace / "cache").exists()
