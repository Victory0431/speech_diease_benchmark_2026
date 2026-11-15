"""label_folders: folder name is the label; no adapter code, no review CSV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from sddbench.cli import main


def _wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    sf.write(str(path), 0.4 * np.sin(2 * np.pi * 220 * t), sr, subtype="PCM_16")


@pytest.fixture
def folders_dataset(tmp_path):
    root = tmp_path / "demo_dataset"
    for folder, names in {"Healthy": ["h1", "h2"], "Disorder": ["d1", "d2"]}.items():
        (root / folder).mkdir(parents=True)
        for name in names:
            _wav(root / folder / f"{name}.wav")
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
    return workspace, paths, task, root


def _scan(workspace: Path, paths: Path, task: Path) -> str:
    assert main(["stats", "--dataset", "demo", "--mode", "full",
                 "--paths", str(paths), "--task", str(task)]) == 0
    scan_dirs = sorted((workspace / "inventory" / "demo").iterdir())
    assert len(scan_dirs) == 1
    return scan_dirs[0].name


def test_stats_and_prepare_need_no_review_file(folders_dataset):
    workspace, paths, task, root = folders_dataset
    scan_id = _scan(workspace, paths, task)

    assert main(["prepare", "--dataset", "demo", "--source", scan_id,
                 "--paths", str(paths), "--task", str(task)]) == 0

    manifests = list((workspace / "prepared").glob("*/*/manifest.csv"))
    assert len(manifests) == 1, "prepare must emit one manifest without a review CSV"
    import pandas as pd
    manifest = pd.read_csv(manifests[0])
    assert sorted(manifest["label"].unique()) == ["disorder", "healthy"]
    assert set(manifest["label_source"]) == {"directory:Disorder", "directory:Healthy"}
    assert set(manifest["group_id"]) == {f"demo:{r}" for r in manifest["relative_path"]}

    admission = json.loads(
        (manifests[0].parent / "admission.json").read_text(encoding="utf-8"))
    assert admission["status"] == "approved"
    assert not (root / "label_review").exists()


def test_unmapped_folder_blocks_instead_of_guessing(folders_dataset):
    workspace, paths, task, _root = folders_dataset
    cfg = yaml.safe_load(task.read_text(encoding="utf-8"))
    cfg["label_map"] = {"Healthy": "healthy"}      # Disorder intentionally missing
    task.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    scan_id = _scan(workspace, paths, task)
    assert main(["prepare", "--dataset", "demo", "--source", scan_id,
                 "--paths", str(paths), "--task", str(task)]) == 3

    admission = json.loads(next(
        (workspace / "prepared").glob("*/*/admission.json")).read_text(encoding="utf-8"))
    assert admission["status"] != "approved"
    assert any(i["issue_type"] == "label_folder_unmapped"
               for i in admission.get("blockers", []))
    assert not list((workspace / "prepared").glob("*/*/manifest.csv"))
