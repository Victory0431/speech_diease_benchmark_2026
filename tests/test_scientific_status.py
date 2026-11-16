"""F01/F02: label evidence and identity evidence are separate gates.

These tests deliberately start from the situations the audit found:
- a `confirmed` review row that only says "the folder name means the label"
  must not become a formal benchmark candidate;
- `label_folders` + `grouping: file` must not be treated as leak-free;
- exact duplicate bytes across groups/labels must block admission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from sddbench.cli import main
from sddbench.config import ConfigError
from sddbench.data.admission import duplicate_content_issues
from sddbench.data.scientific_status import (
    LEVEL_BENCHMARK,
    LEVEL_TECHNICAL,
    assess_scientific_status,
)


def _wav(path: Path, freq: float = 220.0, seconds: float = 1.0, sr: int = 16000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    sf.write(str(path), 0.4 * np.sin(2 * np.pi * freq * t), sr, subtype="PCM_16")


def _folders_dataset(tmp_path: Path, task_extra: dict | None = None):
    root = tmp_path / "demo_dataset"
    for folder, names, freq in (
            ("Healthy", ["h1", "h2"], 220.0), ("Disorder", ["d1", "d2"], 330.0)):
        (root / folder).mkdir(parents=True)
        for i, name in enumerate(names):
            _wav(root / folder / f"{name}.wav", freq=freq + 10 * i)
    workspace = tmp_path / "ws"
    paths = tmp_path / "paths.yaml"
    paths.write_text(yaml.safe_dump({
        "workspace_root": str(workspace),
        "dataset_roots": {"demo": str(root)},
    }), encoding="utf-8")
    cfg = {
        "task_id": "demo_v1", "dataset_id": "demo",
        "layout": "label_folders", "grouping": "file", "task_version": 1,
        "label_map": {"Healthy": "healthy", "Disorder": "disorder"},
    }
    cfg.update(task_extra or {})
    task = tmp_path / "demo_v1.yaml"
    task.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return workspace, paths, task, root


def _scan(workspace: Path, paths: Path, task: Path) -> str:
    assert main(["stats", "--dataset", "demo", "--mode", "full",
                 "--paths", str(paths), "--task", str(task)]) == 0
    return sorted((workspace / "inventory" / "demo").iterdir())[-1].name


def _prepare(workspace: Path, paths: Path, task: Path) -> int:
    scan_id = _scan(workspace, paths, task)
    return main(["prepare", "--dataset", "demo", "--source", scan_id,
                 "--paths", str(paths), "--task", str(task)])


def _latest_admission(workspace: Path) -> dict:
    path = sorted((workspace / "prepared").glob("*/*/admission.json"))[-1]
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ F02

def test_label_folders_without_identity_evidence_is_technical_only(tmp_path):
    workspace, paths, task, _ = _folders_dataset(tmp_path)
    assert _prepare(workspace, paths, task) == 0

    job = _latest_admission(workspace)
    assert job["status"] == "approved"          # structurally usable
    assert job["benchmark_eligible"] is False   # but not leakage-free
    assert job["scientific_status"]["level"] == LEVEL_TECHNICAL
    assert job["identity_evidence"]["kind"] == "unknown"
    assert job["label_evidence"]["source"] == "directory"


def test_one_recording_per_person_requires_a_reference(tmp_path):
    workspace, paths, task, _ = _folders_dataset(tmp_path, {
        "scientific_status": {"level": LEVEL_BENCHMARK},
        "identity": {"evidence": "one_recording_per_person"},
    })
    assert _prepare(workspace, paths, task) == 0
    job = _latest_admission(workspace)
    assert job["benchmark_eligible"] is False
    assert job["scientific_status"]["reason"] == "identity_unverified"

    cfg = yaml.safe_load(task.read_text(encoding="utf-8"))
    cfg["identity"]["reference"] = "synthetic fixture: one file per speaker"
    task.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    assert _prepare(workspace, paths, task) == 0
    job = _latest_admission(workspace)
    assert job["benchmark_eligible"] is True
    assert job["identity_evidence"]["kind"] == "one_recording_per_person"


def test_invalid_scientific_level_is_rejected():
    with pytest.raises(ConfigError, match="scientific_status.level"):
        assess_scientific_status({"scientific_status": {"level": "trust_me"}},
                                 grouping="file", n_confirmed_groups=2)


# ------------------------------------------------------------------ F01

def test_confirmed_folder_review_does_not_resolve_known_label_conflict(scan_workspace):
    """The audit's F01 case: a user-confirmed folder meaning is not source evidence."""
    ws, paths_file, _ = scan_workspace
    review = ws / "label_review" / "uaspeech_label_review.csv"
    review.parent.mkdir(parents=True)
    pd.DataFrame([
        {"speaker_token": "CF02", "directory_group": "Disorder",
         "confirmed_subject_id": "S001", "confirmed_label": "disorder",
         "evidence": "用户确认目录含义", "review_status": "confirmed"},
        {"speaker_token": "F02", "directory_group": "Healthy",
         "confirmed_subject_id": "S002", "confirmed_label": "control",
         "evidence": "用户确认目录含义", "review_status": "confirmed"},
        {"speaker_token": "F03", "directory_group": "Healthy",
         "confirmed_subject_id": "S003", "confirmed_label": "control",
         "evidence": "用户确认目录含义", "review_status": "confirmed"},
    ]).to_csv(review, index=False)

    task = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "uaspeech_v1.yaml"
    assert main(["prepare", "--dataset", "uaspeech", "--source", "scan1",
                 "--task", str(task), "--paths", str(paths_file), "--mic", "M2"]) == 0
    job = json.loads(sorted(
        (ws / "prepared" / "uaspeech_v1").glob("*/admission.json"))[-1]
        .read_text(encoding="utf-8"))
    assert job["status"] == "approved"
    assert job["benchmark_eligible"] is False
    assert job["scientific_status"]["level"] == LEVEL_TECHNICAL
    assert job["scientific_status"]["reason"] == "label_conflict"
    assert job["scientific_status"]["missing_evidence"]


# ------------------------------------------------------------------ duplicates

def test_duplicate_bytes_across_labels_blocks_manifest(tmp_path):
    root = tmp_path / "dup_dataset"
    (root / "Healthy").mkdir(parents=True)
    (root / "Disorder").mkdir()
    _wav(tmp_path / "_same.wav", freq=200.0)
    same = (tmp_path / "_same.wav").read_bytes()
    (root / "Healthy" / "copy.wav").write_bytes(same)
    (root / "Disorder" / "copy.wav").write_bytes(same)
    workspace = tmp_path / "ws"
    paths = tmp_path / "paths.yaml"
    paths.write_text(yaml.safe_dump({
        "workspace_root": str(workspace),
        "dataset_roots": {"demo": str(root)},
    }), encoding="utf-8")
    task = tmp_path / "demo_v1.yaml"
    task.write_text(yaml.safe_dump({
        "task_id": "demo_v1", "dataset_id": "demo",
        "layout": "label_folders", "grouping": "file", "task_version": 1,
        "label_map": {"Healthy": "healthy", "Disorder": "disorder"},
    }, allow_unicode=True), encoding="utf-8")

    assert _prepare(workspace, paths, task) == 3
    job = _latest_admission(workspace)
    assert job["status"] == "blocked_duplicate_content"
    assert job["benchmark_eligible"] is False
    assert any(i["issue_type"] == "duplicate_content_across_groups"
               for i in job["blockers"])
    assert not list((workspace / "prepared").glob("*/*/manifest.csv"))


def test_duplicate_across_groups_is_error():
    rows = [
        {"sample_id": "a.wav", "sha256": "same", "group_id": "d:S1", "label": "control"},
        {"sample_id": "b.wav", "sha256": "same", "group_id": "d:S1", "label": "control"},
        {"sample_id": "c.wav", "sha256": "same", "group_id": "d:S2", "label": "control"},
    ]
    issues = duplicate_content_issues(rows)
    assert len(issues) == 1
    assert issues[0]["issue_type"] == "duplicate_content_across_groups"
    assert issues[0]["severity"] == "error"


def test_duplicate_within_group_is_warning_only():
    rows = [
        {"sample_id": "a.wav", "sha256": "same", "group_id": "d:S1", "label": "control"},
        {"sample_id": "b.wav", "sha256": "same", "group_id": "d:S1", "label": "control"},
    ]
    issues = duplicate_content_issues(rows)
    assert [i["issue_type"] for i in issues] == ["duplicate_content_within_group"]
    assert issues[0]["severity"] == "warning"


# ------------------------------------------------------------------ publish

def _run_package(root: Path, run_id: str, *, eligible: bool) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    block = {
        "n": 10, "accuracy": 0.8, "balanced_accuracy": 0.8, "sensitivity": 0.8,
        "specificity": 0.8, "precision": 0.8, "f1": 0.8, "auroc": 0.8,
        "average_precision": 0.8, "log_loss": 0.4, "brier": 0.2,
        "confusion": {"tn": 4, "fp": 1, "fn": 1, "tp": 4},
    }
    metrics = {
        "run_id": run_id, "task_id": "demo_v1", "dataset": "demo",
        "class_order": ["control", "disorder"], "evaluation_unit": "subject",
        "positive_class": "disorder", "primary_metric": "balanced_accuracy",
        "metrics": {s: {l: dict(block) for l in ("window", "recording", "subject")}
                    for s in ("train", "val", "test")},
    }
    run = {
        "run_id": run_id, "status": "completed", "experiment_id": "exp",
        "dataset": "demo", "task_id": "demo_v1", "prepared_id": "p",
        "split_id": "s", "class_order": ["control", "disorder"],
        "positive_class": "disorder", "seed": 0,
        "benchmark_eligible": eligible,
        "scientific_status": {"level": LEVEL_BENCHMARK if eligible else LEVEL_TECHNICAL,
                              "reason": "" if eligible else "label_conflict"},
        "model": {"name": "mlp"}, "training": {"learning_rate": 0.001},
        "data": {"n_samples": 10, "n_subjects": 10, "feature_dim": 40,
                 "sample_rate": 16000, "subject_counts": {"test": 2},
                 "label_counts": {}, "split_summary": {},
                 "feature_cache": {"feature_key": "k", "feature_hash": "h"}},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")


def test_leaderboard_ranks_only_benchmark_eligible_runs(tmp_path):
    from sddbench.results.publish import build_leaderboard, publish_all

    runs = tmp_path / "runs"
    _run_package(runs, "eligible_run", eligible=True)
    _run_package(runs, "technical_run", eligible=False)
    publish_all(runs, tmp_path / "results")
    frame = build_leaderboard(tmp_path / "results")
    assert set(frame["run_id"]) == {"eligible_run", "technical_run"}
    assert frame.iloc[0]["run_id"] == "eligible_run"
    assert bool(frame.iloc[0]["benchmark_eligible"]) is True
    text = (tmp_path / "results" / "leaderboard.md").read_text(encoding="utf-8")
    assert "eligible_run" in text and "technical_run" in text
    assert "不排名" in text
