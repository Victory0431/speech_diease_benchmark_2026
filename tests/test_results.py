"""D8: the results contract, digest publishing and the leaderboard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sddbench.results.contract import (
    CONTRACT_VERSION,
    ResultsContractError,
    sanitize_run_meta,
    validate_metrics,
)
from sddbench.results.publish import build_leaderboard, publish_all, publish_run


def _block(value: float = 0.8) -> dict:
    return {
        "n": 10, "accuracy": value, "balanced_accuracy": value,
        "sensitivity": value, "specificity": value, "precision": value,
        "f1": value, "auroc": value, "average_precision": value,
        "log_loss": 0.4, "brier": 0.2,
        "confusion": {"tn": 4, "fp": 1, "fn": 1, "tp": 4},
    }


def _metrics(run_id: str = "run1", dataset: str = "demo") -> dict:
    return {
        "run_id": run_id, "task_id": "demo_v1", "dataset": dataset,
        "class_order": ["control", "disorder"], "evaluation_unit": "subject",
        "positive_class": "disorder", "primary_metric": "balanced_accuracy",
        "metrics": {split: {level: _block() for level in
                            ("window", "recording", "subject")}
                    for split in ("train", "val", "test")},
    }


def _run(run_id: str = "run1") -> dict:
    return {
        "run_id": run_id, "status": "completed", "experiment_id": "exp",
        "dataset": "demo", "task_id": "demo_v1",
        "prepared_id": "prep", "split_id": "split",
        "class_order": ["control", "disorder"], "positive_class": "disorder",
        "seed": 0, "model": {"name": "mlp", "hidden_dims": [8], "dropout": 0.1},
        "training": {"learning_rate": 0.001, "batch_size": 8, "device": "cpu"},
        "best_epoch": 3, "best_val_subject_balanced_accuracy": 0.9,
        "elapsed_sec": 1.0, "code_version": "0.1.0", "config_hash": "abc",
        "git": {"commit": "1234567", "dirty": False},
        "data": {
            "n_samples": 24, "n_subjects": 8, "feature_dim": 40,
            "sample_rate": 16000, "subject_counts": {"train": 4, "val": 2, "test": 2},
            "label_counts": {}, "split_summary": {},
            "manifest": "/abs/manifest.csv",
            "split": "/abs/split.csv",
            "feature_cache": {
                "cache_dir": "/abs/cache", "feature_key": "k1",
                "feature_hash": "h1", "view_fingerprint": "v1",
                "sample_rate": 16000, "dim": 40, "n_samples": 24,
            },
        },
        "out_dir": "/abs/run",
        "outputs": {"report": "/abs/run/report.md"},
    }


def _make_run_dir(root: Path, run_id: str = "run1") -> Path:
    run_dir = root / run_id
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(_metrics(run_id)),
                                          encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(_run(run_id)), encoding="utf-8")
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "metrics_per_class.csv").write_text("split,class\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text("experiment_id: exp\n", encoding="utf-8")
    (run_dir / "figures" / "confusion_test.png").write_bytes(b"png")
    # these must never be published
    (run_dir / "predictions_windows.csv").write_text("subject_id\nC1\n", encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"weights")
    (run_dir / "inputs").mkdir()
    (run_dir / "inputs" / "manifest.csv").write_text("x\n", encoding="utf-8")
    return run_dir


def test_contract_accepts_complete_metrics_and_rejects_missing():
    doc = _metrics()
    validate_metrics(doc)
    broken = _metrics()
    del broken["metrics"]["test"]["subject"]["auroc"]
    with pytest.raises(ResultsContractError, match="auroc"):
        validate_metrics(broken)
    broken = _metrics()
    broken["evaluation_unit"] = "window"
    with pytest.raises(ResultsContractError, match="subject"):
        validate_metrics(broken)


def test_sanitize_meta_drops_paths_and_patient_rows():
    meta = sanitize_run_meta(_run())
    text = json.dumps(meta)
    assert "/abs/" not in text
    assert "manifest.csv" not in text
    assert "predictions" not in text
    assert meta["contract_version"] == CONTRACT_VERSION
    assert meta["data"]["feature_cache"] == {
        "feature_key": "k1", "feature_hash": "h1", "view_fingerprint": "v1",
        "sample_rate": 16000, "dim": 40, "n_samples": 24}


def test_publish_run_only_writes_whitelisted_small_files(tmp_path):
    run_dir = _make_run_dir(tmp_path / "runs")
    dest = publish_run(run_dir, tmp_path / "results")
    written = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert written == {"metrics.json", "run_meta.json", "report.md",
                       "metrics_per_class.csv", "config.yaml",
                       "figures/confusion_test.png"}
    published = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    assert published["contract_version"] == CONTRACT_VERSION
    meta = (dest / "run_meta.json").read_text(encoding="utf-8")
    assert "/abs/" not in meta and "predictions" not in meta


def test_publish_all_builds_leaderboard_and_reports_failures(tmp_path):
    runs = tmp_path / "runs"
    _make_run_dir(runs, "run1")
    broken = runs / "broken"
    broken.mkdir()
    (broken / "metrics.json").write_text("{}", encoding="utf-8")
    published, failures = publish_all(runs, tmp_path / "results")
    assert len(published) == 1
    assert len(failures) == 1
    frame = build_leaderboard(tmp_path / "results")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["dataset"] == "demo"
    assert row["test_subjects"] == 2
    assert row["subject_balanced_accuracy"] == 0.8
    assert row["subject_auroc"] == 0.8
    assert row["window_balanced_accuracy"] == 0.8
    assert (tmp_path / "results" / "leaderboard.md").is_file()
