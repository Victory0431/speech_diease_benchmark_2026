"""F04/F07: prepared/split/feature identity closure and processed-Hz binding.

These tests reproduce the audit probes:
- changing manifest content (source hash) while keeping sample ids must be rejected;
- changing task.yaml snapshot must be rejected;
- changing split.csv must be rejected;
- two datasets with the same relative path must not share vectors;
- 16k->8k and 44.1k->16k must generate, store and load under the *processed* Hz;
- a re-scan of unchanged bytes must reuse the same window id;
- legacy (schema 1.0) caches must be refused with a rebuild instruction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from sddbench.data.decoders import file_sha256
from sddbench.data.identity import (
    manifest_fingerprint,
    split_fingerprint,
    task_fingerprint,
)
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
from sddbench.data.windows import (
    build_view_index,
    normalize_preprocess,
    preprocess_fingerprint,
)
from sddbench.features.cache import FeatureCache
from sddbench.features.mfcc import extract_mfcc, normalize_feature_config
from sddbench.modeling.experiment import load_experiment
from sddbench.modeling.features import FeatureLoadError, load_features
from sddbench.modeling.trainer import RunError, load_run_data

REPO = Path(__file__).resolve().parents[1]
FEAT_YAML = REPO / "configs" / "features" / "mfcc20_stats.yaml"


def _feat() -> dict:
    return normalize_feature_config(yaml.safe_load(FEAT_YAML.read_text(encoding="utf-8")))


def _pre(target_sr: int | None) -> dict:
    raw = {
        "name": "demo", "sample_rate_policy": "resample" if target_sr else "native",
        "target_sample_rate": target_sr, "allow_mixed_native": False,
        "observe_seconds": 1.0, "window_seconds": 1.0, "position": "head",
        "short_policy": "right_zero_pad", "channel_policy": "select",
        "channel_index": 0, "amplitude_normalization": "none", "crop_seed": 0,
    }
    return normalize_preprocess(raw)


def _sine(seconds: float, sr: int) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _rec(sr: int, root: Path, rel: str = "Healthy/a.wav") -> tuple[dict, str]:
    wav = root / rel
    wav.parent.mkdir(parents=True, exist_ok=True)
    if not wav.is_file():
        sf.write(str(wav), _sine(2.0, sr), sr, subtype="PCM_16")
    digest = file_sha256(wav)
    return {
        COL_RELATIVE_PATH: rel, COL_CATEGORY: "audio_candidate",
        COL_DECODE_STATUS: "ok", COL_SHA256: digest, COL_SAMPLE_RATE: sr,
        COL_N_SAMPLES: int(2.0 * sr), COL_DURATION_DECODED_SEC: 2.0,
        COL_DIRECTORY_GROUP: "Healthy", COL_SPEAKER_TOKEN: "S1",
    }, digest


@pytest.mark.parametrize("source_sr,target_sr", [(16000, 8000), (44100, 16000)])
def test_resampled_features_are_stored_and_loaded_at_processed_hz(
        tmp_path, source_sr, target_sr):
    root = tmp_path / "ds"
    rec, digest = _rec(source_sr, root)
    cfg = _pre(target_sr)
    feat = _feat()
    workspace = tmp_path / "ws"
    cache = FeatureCache(workspace, feat, target_sr, cache_mode="shared",
                         view_fingerprint=preprocess_fingerprint(cfg))

    def on_view(row, valid, processed_sr):
        res = extract_mfcc(valid, processed_sr, feat)
        cache.add(row["window_id"], row["sample_id"], res.vector, res.n_frames,
                  row["view_hash"], dataset_id="demo", source_sha256=row["source_sha256"],
                  processed_sample_rate=processed_sr)

    table = build_view_index(root, [rec], cfg, dataset_id="demo",
                             source_scan_id="scan1", on_view=on_view)
    assert table.rows[0]["original_sample_rate"] == source_sr
    assert table.rows[0]["processed_sample_rate"] == target_sr
    cache.finalize()
    index = pd.read_csv(cache.path / "index.csv")
    assert index.iloc[0]["processed_sample_rate"] == target_sr
    assert index.iloc[0]["source_sha256"] == digest
    assert index.iloc[0]["dataset_id"] == "demo"

    X, info = load_features(workspace, feat, ["Healthy/a.wav"], target_sr,
                            dataset_id="demo",
                            source_hashes={"Healthy/a.wav": digest},
                            view_fingerprint=preprocess_fingerprint(cfg))
    assert X.shape == (1, 40)
    assert info["sample_rate"] == target_sr
    assert info["cache_schema_version"] == "2.0"


def test_window_id_is_dataset_namespaced_and_rescan_stable(tmp_path):
    root = tmp_path / "ds"
    rec, _ = _rec(16000, root)
    cfg = _pre(None)
    a = build_view_index(root, [rec], cfg, dataset_id="demo",
                         source_scan_id="scan1")
    b = build_view_index(root, [rec], cfg, dataset_id="demo",
                         source_scan_id="scan2-rescan")
    c = build_view_index(root, [rec], cfg, dataset_id="other",
                         source_scan_id="scan1")
    # unchanged bytes re-scanned: same window identity, no fake extra window
    assert a.rows[0]["window_id"] == b.rows[0]["window_id"]
    assert a.rows[0]["view_hash"] == b.rows[0]["view_hash"]
    # a different dataset must not share the window identity
    assert c.rows[0]["window_id"] != a.rows[0]["window_id"]
    # new scan id is still recorded as provenance
    assert b.rows[0]["source_scan_id"] == "scan2-rescan"


def test_cross_dataset_same_relative_path_does_not_cross_load(tmp_path):
    feat = _feat()
    cfg = _pre(None)
    view_fp = preprocess_fingerprint(cfg)
    workspace = tmp_path / "ws"
    cache = FeatureCache(workspace, feat, 16000, cache_mode="shared",
                         view_fingerprint=view_fp)
    cache.add("a|Healthy/a.wav|w0|x", "Healthy/a.wav", np.full(40, 1.0, np.float32),
              100, "vh", dataset_id="a", source_sha256="sha_a")
    cache.add("b|Healthy/a.wav|w0|x", "Healthy/a.wav", np.full(40, 2.0, np.float32),
              100, "vh", dataset_id="b", source_sha256="sha_b")
    cache.finalize()

    Xa, _ = load_features(workspace, feat, ["Healthy/a.wav"], 16000, dataset_id="a",
                          source_hashes={"Healthy/a.wav": "sha_a"},
                          view_fingerprint=view_fp)
    Xb, _ = load_features(workspace, feat, ["Healthy/a.wav"], 16000, dataset_id="b",
                          source_hashes={"Healthy/a.wav": "sha_b"},
                          view_fingerprint=view_fp)
    assert np.allclose(Xa, 1.0) and np.allclose(Xb, 2.0)

    with pytest.raises(FeatureLoadError, match="another dataset"):
        load_features(workspace, feat, ["Healthy/a.wav"], 16000, dataset_id="c",
                      view_fingerprint=view_fp)
    with pytest.raises(FeatureLoadError, match="different source content"):
        load_features(workspace, feat, ["Healthy/a.wav"], 16000, dataset_id="a",
                      source_hashes={"Healthy/a.wav": "sha_CHANGED"},
                      view_fingerprint=view_fp)


def test_legacy_cache_schema_is_refused_with_rebuild_instruction(tmp_path):
    feat = _feat()
    cache = FeatureCache(tmp_path, feat, 16000, cache_mode="shared")
    cache.add("w", "s.wav", np.zeros(40, np.float32), 100, "vh",
              dataset_id="demo", source_sha256="sha")
    cache.finalize()
    meta = json.loads((cache.path / "meta.json").read_text(encoding="utf-8"))
    meta["schema_version"] = "1.0"
    (cache.path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    from sddbench.features.registry import FeatureConfigError
    with pytest.raises(FeatureConfigError, match="schema"):
        FeatureCache(tmp_path, feat, 16000, cache_mode="shared")
    assert (cache.path / "index.csv").is_file(), "legacy cache must not be deleted"


def test_manifest_and_split_fingerprints_track_identity():
    row = {
        "sample_id": "a.wav", "relative_path": "Healthy/a.wav", "subject_id": "S1",
        "group_id": "demo:S1", "origin_id": "o1", "label_raw": "Healthy",
        "label": "control", "variant": "original", "elicitation": "unknown",
        "sample_rate": 16000, "duration_decoded_sec": 1.0, "channels": 1,
        "sha256": "h1", "official_split": "", "label_source": "directory",
    }
    fp = manifest_fingerprint([row], task_hash="t1")
    assert manifest_fingerprint([row], task_hash="t1") == fp
    assert manifest_fingerprint([row], task_hash="t2") != fp
    for field, value in (("group_id", "demo:S2"), ("label", "disorder"),
                         ("origin_id", "o2"), ("sha256", "h2"),
                         ("variant", "clipped"), ("subject_id", "S2")):
        changed = dict(row, **{field: value})
        assert manifest_fingerprint([changed], task_hash="t1") != fp, field

    splits = [{"sample_id": "a.wav", "group_id": "demo:S1", "label": "control",
               "split": "train"}]
    sfp = split_fingerprint(splits)
    assert split_fingerprint(list(splits)) == sfp
    assert split_fingerprint([dict(splits[0], split="test")]) != sfp


# ------------------------------------------------- F04 rejection in training


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "configs" / "datasets").mkdir(parents=True)
    (repo / "configs" / "features").mkdir(parents=True)
    (repo / "configs" / "preprocessing").mkdir(parents=True)
    (repo / "configs" / "datasets" / "demo_v1.yaml").write_text(yaml.safe_dump({
        "task_id": "demo_v1",
        "labels": {"class_order": ["control", "disorder"], "positive_class": "disorder"},
    }), encoding="utf-8")
    (repo / "configs" / "features" / "mfcc20_stats.yaml").write_text(
        (REPO / "configs" / "features" / "mfcc20_stats.yaml").read_text(encoding="utf-8"),
        encoding="utf-8")
    (repo / "configs" / "preprocessing" / "demo.yaml").write_text(yaml.safe_dump({
        "name": "demo", "sample_rate_policy": "native", "allow_mixed_native": False,
        "observe_seconds": 1.0, "window_seconds": 1.0, "position": "head",
    }), encoding="utf-8")

    workspace = tmp_path / "ws"
    prepared = workspace / "prepared" / "demo_v1" / "prep1"
    split_dir = workspace / "splits" / "demo_v1" / "split1"
    prepared.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    rows, split_rows, rates = [], [], [16000, 44100]
    for i in range(8):
        label = "control" if i < 4 else "disorder"
        subject = f"S{i}"
        sample_id = f"{subject}.wav"
        rates_cycle = rates[i % 2]
        rows.append({
            "sample_id": sample_id, "relative_path": sample_id, "subject_id": subject,
            "group_id": f"demo:{subject}", "label": label, "label_raw": label,
            "variant": "original", "elicitation": "unknown",
            "sample_rate": rates_cycle, "duration_decoded_sec": 1.0, "channels": 1,
            "sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
        })
        split_rows.append({"sample_id": sample_id, "group_id": f"demo:{subject}",
                           "label": label,
                           "split": "train" if i < 4 else ("val" if i == 4 else "test")})
    pd.DataFrame(rows).to_csv(prepared / "manifest.csv", index=False)
    task_cfg = {
        "task_id": "demo_v1", "grouping": "subject",
        "labels": {"class_order": ["control", "disorder"], "positive_class": "disorder"},
    }
    (prepared / "task.yaml").write_text(yaml.safe_dump(task_cfg), encoding="utf-8")
    task_hash = task_fingerprint(task_cfg)
    manifest_fp = manifest_fingerprint(rows, task_hash=task_hash)
    (prepared / "admission.json").write_text(json.dumps({
        "manifest_fingerprint": manifest_fp, "task_hash": task_hash,
    }), encoding="utf-8")
    pd.DataFrame(split_rows).to_csv(split_dir / "split.csv", index=False)
    (split_dir / "split.json").write_text(json.dumps({
        "summary": {"n_samples": len(rows)}, "manifest_fingerprint": manifest_fp,
        "task_hash": task_hash, "split_fingerprint": split_fingerprint(split_rows),
    }), encoding="utf-8")
    return repo, workspace


def _cfg(tmp_path: Path) -> dict:
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({
        "experiment_id": "demo",
        "dataset": "demo", "prepared_id": "prep1", "split_id": "split1",
        "features": {"config": "configs/features/mfcc20_stats.yaml",
                     "preprocess": "configs/preprocessing/demo.yaml"},
        "model": {"name": "mlp", "hidden_dims": [8]},
        "training": {"seed": 0, "epochs": 1},
    }), encoding="utf-8")
    return load_experiment(path)


def test_modified_manifest_content_is_rejected(tmp_path):
    repo, workspace = _workspace(tmp_path)
    cfg = _cfg(tmp_path)
    manifest_file = workspace / "prepared" / "demo_v1" / "prep1" / "manifest.csv"
    frame = pd.read_csv(manifest_file)
    frame.loc[0, "sha256"] = "0" * 64
    frame.to_csv(manifest_file, index=False)
    with pytest.raises(RunError, match="manifest.csv content changed"):
        load_run_data(cfg, workspace, repo)


def test_modified_task_snapshot_is_rejected(tmp_path):
    repo, workspace = _workspace(tmp_path)
    cfg = _cfg(tmp_path)
    snapshot = workspace / "prepared" / "demo_v1" / "prep1" / "task.yaml"
    task = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    task["grouping"] = "file"
    snapshot.write_text(yaml.safe_dump(task), encoding="utf-8")
    with pytest.raises(RunError, match="task.yaml snapshot"):
        load_run_data(cfg, workspace, repo)


def test_missing_task_snapshot_is_rejected(tmp_path):
    repo, workspace = _workspace(tmp_path)
    cfg = _cfg(tmp_path)
    (workspace / "prepared" / "demo_v1" / "prep1" / "task.yaml").unlink()
    with pytest.raises(RunError, match="no task.yaml snapshot"):
        load_run_data(cfg, workspace, repo)


def test_modified_split_assignment_is_rejected(tmp_path):
    repo, workspace = _workspace(tmp_path)
    cfg = _cfg(tmp_path)
    split_file = workspace / "splits" / "demo_v1" / "split1" / "split.csv"
    frame = pd.read_csv(split_file)
    frame.loc[0, "split"] = "test"
    frame.to_csv(split_file, index=False)
    with pytest.raises(RunError, match="split.csv content changed"):
        load_run_data(cfg, workspace, repo)


def test_native_mixed_rates_are_rejected_before_training(tmp_path):
    repo, workspace = _workspace(tmp_path)
    cfg = _cfg(tmp_path)
    with pytest.raises(RunError, match="mixed original rates"):
        load_run_data(cfg, workspace, repo)
