"""D6: metrics, MLP, feature loading, config overrides and an end-to-end run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from sddbench.data.windows import normalize_preprocess, preprocess_fingerprint
from sddbench.features.cache import FeatureCache
from sddbench.features.mfcc import normalize_feature_config
from sddbench.modeling.experiment import (
    ExperimentConfigError,
    expand_grid,
    load_experiment,
    set_override,
)
from sddbench.modeling.features import FeatureLoadError, load_features
from sddbench.modeling.metrics import auroc, binary_metrics, confusion
from sddbench.modeling.mlp import MLP
from sddbench.modeling.sweep import run_sweep, summarize_runs
from sddbench.modeling.trainer import RunError, load_run_data, run_experiment

REPO = Path(__file__).resolve().parents[1]


def test_mlp_shape_and_dropout():
    model = MLP(40, [16, 8], 2, dropout=0.5)
    out = model(torch.zeros(7, 40))
    assert out.shape == (7, 2)
    model.eval()
    assert torch.allclose(model(torch.ones(1, 40)), model(torch.ones(1, 40)))
    with pytest.raises(ValueError):
        MLP(4, [2], 2, activation="tanh")


def test_metrics_are_correct_for_known_predictions():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    cm = confusion(y, (probs[:, 1] >= 0.5).astype(int))
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    m = binary_metrics(y, probs, 1)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["auroc"] == pytest.approx(0.5)
    assert m["average_precision"] == pytest.approx(7 / 12)
    assert auroc(np.zeros(5, dtype=int), np.arange(5) / 5) is None


def test_experiment_overrides_and_grid():
    cfg = {
        "dataset": "demo", "prepared_id": "p", "split_id": "s",
        "features": {"config": "configs/features/mfcc20_stats.yaml"},
        "model": {"name": "mlp", "hidden_dims": [128], "dropout": 0.2},
        "training": {"seed": 0, "optimizer": "adamw"},
    }
    cfg = load_experiment_dict(cfg)
    set_override(cfg, "model.dropout=0.0")
    set_override(cfg, "model.hidden_dims=[64, 32]")
    assert cfg["model"]["dropout"] == 0.0
    assert cfg["model"]["hidden_dims"] == [64, 32]
    with pytest.raises(ExperimentConfigError):
        set_override(cfg, "model.nope=1")
    grid = expand_grid(cfg, {"model.dropout": [0.0, 0.3], "training.seed": [0, 1]})
    assert len(grid) == 4
    assert {(g["model"]["dropout"], g["training"]["seed"]) for g in grid} == {
        (0.0, 0), (0.0, 1), (0.3, 0), (0.3, 1)}


def load_experiment_dict(cfg: dict) -> dict:
    """Merge DEFAULTS without touching the filesystem (test helper)."""
    from sddbench.modeling.experiment import DEFAULTS, _merge
    return _merge(DEFAULTS, cfg)


# --------------------------------------------------------------- fixture


def _build_workspace(tmp_path: Path, *, leak: bool = False) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "configs" / "datasets").mkdir(parents=True)
    (repo / "configs" / "features").mkdir(parents=True)
    (repo / "configs" / "preprocessing").mkdir(parents=True)
    (repo / "configs" / "datasets" / "demo_v1.yaml").write_text(yaml.safe_dump({
        "task_id": "demo_v1",
        "labels": {"class_order": ["control", "disorder"], "positive_class": "disorder"},
    }), encoding="utf-8")
    feature_raw = yaml.safe_load((REPO / "configs/features/mfcc20_stats.yaml").read_text(
        encoding="utf-8"))
    (repo / "configs" / "features" / "mfcc20_stats.yaml").write_text(
        yaml.safe_dump(feature_raw), encoding="utf-8")
    preprocess_raw = {
        "name": "demo", "profile": "test", "sample_rate_policy": "native",
        "allow_mixed_native": False, "target_sample_rate": None,
        "channel_policy": "select", "channel_index": 0, "view_mode": "single_segment",
        "observe_seconds": 4.0, "window_seconds": 4.0, "position": "head",
        "crop_seed": 0, "short_policy": "right_zero_pad",
        "amplitude_normalization": "none", "frame_ms": 25.0, "hop_ms": 10.0,
        "crop_algorithm_version": "1.0",
    }
    (repo / "configs" / "preprocessing" / "demo.yaml").write_text(
        yaml.safe_dump(preprocess_raw), encoding="utf-8")

    workspace = tmp_path / "ws"
    prepared = workspace / "prepared" / "demo_v1" / "prep1"
    split_dir = workspace / "splits" / "demo_v1" / "split1"
    prepared.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    subjects = [("C1", "control"), ("C2", "control"), ("C3", "control"), ("C4", "control"),
                ("D1", "disorder"), ("D2", "disorder"), ("D3", "disorder"),
                ("D4", "disorder")]
    split_of = {"C1": "train", "C2": "train", "D1": "train", "D2": "train",
                "C3": "val", "D3": "val", "C4": "test", "D4": "test"}
    rows, split_rows = [], []
    rng = np.random.default_rng(0)
    for subject, label in subjects:
        for i in range(4):
            sample_id = f"{subject}_{i}.wav"
            rows.append({
                "sample_id": sample_id, "relative_path": sample_id,
                "subject_id": subject, "group_id": f"demo:{subject}", "label": label,
                "sample_rate": 16000, "duration_decoded_sec": 4.0,
            })
            split = "test" if (leak and subject == "C1" and i >= 2) else split_of[subject]
            split_rows.append({"sample_id": sample_id, "group_id": f"demo:{subject}",
                               "label": label, "split": split})
    pd.DataFrame(rows).to_csv(prepared / "manifest.csv", index=False)
    (prepared / "admission.json").write_text(json.dumps(
        {"manifest_fingerprint": "fp1"}), encoding="utf-8")
    pd.DataFrame(split_rows).to_csv(split_dir / "split.csv", index=False)
    (split_dir / "split.json").write_text(json.dumps(
        {"summary": {"n_samples": len(rows)}, "manifest_fingerprint": "fp1"}),
        encoding="utf-8")

    feature_cfg = normalize_feature_config(feature_raw)
    preprocess_cfg = normalize_preprocess(preprocess_raw)
    view_fp = preprocess_fingerprint(preprocess_cfg)
    cache = FeatureCache(workspace / "cache" / "features" / "mfcc_stats",
                         feature_cfg, 16000, cache_mode="shared",
                         view_fingerprint=view_fp)
    for row in rows:
        label_shift = 3.0 if row["label"] == "disorder" else 0.0
        vec = rng.normal(label_shift, 0.5, size=40).astype(np.float32)
        cache.add(f"job|{row['sample_id']}|w0|abcd", row["sample_id"], vec, 100,
                  "viewhash")
    cache.finalize()
    return repo, workspace, tmp_path / "runs"


def _experiment_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({
        "experiment_id": "demo_mlp",
        "dataset": "demo", "prepared_id": "prep1", "split_id": "split1",
        "features": {"config": "configs/features/mfcc20_stats.yaml",
                     "preprocess": "configs/preprocessing/demo.yaml"},
        "model": {"name": "mlp", "hidden_dims": [16], "dropout": 0.1},
        "training": {"seed": 0, "epochs": 20, "batch_size": 8,
                     "learning_rate": 0.005, "early_stopping_patience": 5,
                     "class_weighting": "subject_balanced", "device": "cpu"},
        "labels": {"class_order": ["control", "disorder"], "positive_class": "disorder"},
        "evaluation": {"unit": "subject"},
    }), encoding="utf-8")
    return path


def test_feature_loader_rejects_missing_sample(tmp_path):
    repo, workspace, _ = _build_workspace(tmp_path)
    cfg = load_experiment(_experiment_cfg(tmp_path))
    data = load_run_data(cfg, workspace, repo)
    assert data["X"].shape == (32, 40)
    feature_cfg = data["feature_cfg"]
    preprocess = normalize_preprocess(yaml.safe_load(
        (repo / "configs/preprocessing/demo.yaml").read_text(encoding="utf-8")))
    with pytest.raises(FeatureLoadError):
        load_features(workspace, feature_cfg, ["nope.wav"], 16000,
                      view_fingerprint=preprocess_fingerprint(preprocess))


def test_end_to_end_run_writes_result_pack(tmp_path):
    repo, workspace, runs = _build_workspace(tmp_path)
    cfg = load_experiment(_experiment_cfg(tmp_path))
    job = run_experiment(cfg, workspace, repo, run_root=runs)
    out = Path(job["out_dir"])
    for name in ("config.yaml", "run.json", "history.csv", "best.pt", "last.pt",
                 "predictions_windows.csv", "predictions_recordings.csv",
                 "predictions_subjects.csv", "metrics.json", "metrics_per_class.csv",
                 "report.md"):
        assert (out / name).is_file(), name
    assert (out / "figures" / "confusion_test.png").is_file()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_unit"] == "subject"
    assert metrics["class_order"] == ["control", "disorder"]
    assert metrics["coverage"]["test"] == {"n_expected": 8, "n_predicted": 8}
    subject_pred = pd.read_csv(out / "predictions_subjects.csv")
    assert set(subject_pred["split"]) == {"train", "val", "test"}
    assert subject_pred["prob_control"].between(0, 1).all()
    assert (subject_pred["prob_control"] + subject_pred["prob_disorder"]).round(6).eq(1).all()
    # standardisation must be fitted on train only; recorded for reconstruction
    assert len(job["scaler"]["mean"]) == 40
    # test people never appear in training
    assert not (set(subject_pred.loc[subject_pred.split == "train", "subject_id"])
                & set(subject_pred.loc[subject_pred.split == "test", "subject_id"]))


def test_recording_unit_run_is_supported(tmp_path):
    repo, workspace, runs = _build_workspace(tmp_path)
    cfg = load_experiment(_experiment_cfg(tmp_path))
    cfg["evaluation"]["unit"] = "recording"
    job = run_experiment(cfg, workspace, repo, run_root=runs)
    metrics = json.loads((Path(job["out_dir"]) / "metrics.json").read_text(
        encoding="utf-8"))
    assert metrics["evaluation_unit"] == "recording"
    assert metrics["primary_unit"] == "recording"
    assert metrics["metrics"]["test"]["recording"]["n"] == 8
    assert job["best_val_balanced_accuracy"] == job["best_val_subject_balanced_accuracy"]
    assert "val_recording_balanced_accuracy" in pd.read_csv(
        Path(job["out_dir"]) / "history.csv").columns


def test_person_leakage_is_rejected(tmp_path):
    repo, workspace, runs = _build_workspace(tmp_path, leak=True)
    cfg = load_experiment(_experiment_cfg(tmp_path))
    with pytest.raises(RunError, match="leakage"):
        run_experiment(cfg, workspace, repo, run_root=runs)


def test_sweep_and_summarize(tmp_path):
    repo, workspace, runs = _build_workspace(tmp_path)
    cfg = load_experiment(_experiment_cfg(tmp_path))
    sweep = {"sweep_id": "demo_sweep", "axes": {"model.dropout": [0.0, 0.5]}}
    out_dir, jobs = run_sweep(cfg, sweep, workspace, repo, run_root=runs)
    assert len(jobs) == 2
    assert all(job["status"] == "completed" for job in jobs)
    assert (out_dir / "jobs.csv").is_file()
    sweep_doc = json.loads((out_dir / "sweep.json").read_text(encoding="utf-8"))
    assert sweep_doc["axes"] == {"model.dropout": [0.0, 0.5]}

    frame = summarize_runs(runs, tmp_path / "summary")
    assert len(frame) == 2
    assert frame["test_subject_balanced_accuracy"].notna().all()
    assert (tmp_path / "summary" / "summary.csv").is_file()
    assert (tmp_path / "summary" / "summary.md").is_file()
