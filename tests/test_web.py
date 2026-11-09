"""D12: schema, catalog and the local web server."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from sddbench.web.catalog import build_catalog
from sddbench.web.schema import build_configs, get_schema, validate_values, write_configs
from sddbench.web.server import make_server


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "configs" / "datasets").mkdir(parents=True)
    (repo / "configs" / "features").mkdir(parents=True)
    (repo / "configs" / "preprocessing").mkdir(parents=True)
    (repo / "configs" / "experiments").mkdir(parents=True)
    (repo / "configs" / "datasets" / "demo_v1.yaml").write_text(yaml.safe_dump({
        "task_id": "demo_v1",
        "dataset_id": "demo",
        "adapter": "demo",
        "grouping": "subject",
        "selection": {"one_per_origin": True},
        "labels": {"class_order": ["control", "disorder"],
                   "positive_class": "disorder"},
    }), encoding="utf-8")
    (repo / "configs" / "preprocessing" / "demo.yaml").write_text(yaml.safe_dump({
        "name": "demo", "profile": "test", "sample_rate_policy": "native",
        "allow_mixed_native": False, "target_sample_rate": None,
        "channel_policy": "select", "channel_index": 0, "view_mode": "single_segment",
        "observe_seconds": 4.0, "window_seconds": 4.0, "position": "head",
        "crop_seed": 0, "short_policy": "right_zero_pad",
        "amplitude_normalization": "none", "frame_ms": 25.0, "hop_ms": 10.0,
    }), encoding="utf-8")
    (repo / "configs" / "features" / "mfcc20.yaml").write_text(yaml.safe_dump({
        "name": "mfcc", "n_mfcc": 20, "n_mels": 64, "frame_ms": 25.0,
        "hop_ms": 10.0, "f_min_hz": 0.0, "f_max_policy": "nyquist",
        "stats": ["mean", "std"], "std_ddof": 0,
    }), encoding="utf-8")
    (repo / "configs" / "experiments" / "demo.yaml").write_text(yaml.safe_dump({
        "experiment_id": "demo_base", "dataset": "demo",
        "prepared_id": "job1", "split_id": "s1",
        "features": {"config": "configs/features/mfcc20.yaml",
                     "preprocess": "configs/preprocessing/demo.yaml"},
        "model": {"name": "mlp", "hidden_dims": [32], "dropout": 0.1},
        "training": {"seed": 0},
    }), encoding="utf-8")

    workspace = tmp_path / "ws"
    _write_json(workspace / "inventory" / "demo" / "scan1" / "meta.json", {
        "scan_id": "scan1", "mode": "full", "status": "completed",
    })
    _write_json(workspace / "prepared" / "demo_v1" / "job1" / "admission.json", {
        "job_id": "job1", "status": "approved", "grouping": "subject",
        "counts": {"n_manifest": 8, "n_subjects": 4, "n_groups": 4},
    })
    (workspace / "prepared" / "demo_v1" / "job1" / "manifest.csv").write_text(
        "sample_id\n", encoding="utf-8")
    _write_json(workspace / "splits" / "demo_v1" / "s1" / "split.json", {
        "split_id": "s1", "seed": 0, "prepared_id": "job1",
        "summary": {"n_samples": 8, "n_groups": 4,
                    "by_split": {"test": {"n_samples": 2}}},
    })
    _write_json(workspace / "runs" / "r1" / "run.json", {
        "run_id": "r1", "dataset": "demo", "experiment_id": "demo_base",
        "seed": 0, "split_id": "s1", "prepared_id": "job1", "primary_unit": "subject",
    })
    _write_json(workspace / "runs" / "r1" / "metrics.json", {
        "dataset": "demo", "primary_unit": "subject",
        "metrics": {"test": {"subject": {"balanced_accuracy": 0.8, "auroc": 0.9}}},
    })
    (workspace / "runs" / "r1" / "figures").mkdir(parents=True)
    (workspace / "runs" / "r1" / "figures" / "confusion_test.png").write_bytes(b"png")

    paths = tmp_path / "paths.yaml"
    paths.write_text(yaml.safe_dump({
        "workspace_root": str(workspace),
        "dataset_roots": {"demo": str(tmp_path / "data")},
    }), encoding="utf-8")
    return repo, workspace, paths


def test_schema_validates_and_covers_pipeline():
    schema = get_schema()
    keys = {f["key"] for section in schema["sections"] for f in section["fields"]}
    for required in ("features.n_mfcc", "preprocess.observe_seconds",
                     "split.seed", "training.seed", "evaluation.unit"):
        assert required in keys
    assert validate_values({"features.n_mfcc": 13})["features.n_mfcc"] == 13
    with pytest.raises(ValueError):
        validate_values({"features.n_mfcc": 999})
    with pytest.raises(ValueError):
        validate_values({"no.such.key": 1})
    with pytest.raises(ValueError):
        validate_values({"preprocess.observe_seconds": 4.0,
                         "preprocess.window_seconds": 2.0})


def test_build_configs_respects_grouping_and_writes_yaml(tmp_path):
    repo, _, _ = _workspace(tmp_path)
    dataset_cfg = {
        "task_id": "demo_v1", "grouping": "file",
        "labels": {"class_order": ["control", "disorder"],
                   "positive_class": "disorder"},
    }
    values = {
        "features.n_mfcc": 13,
        "model.hidden_dims": [64, 32],
        "evaluation.unit": "recording",
        "training.seed": 3,
    }
    experiment, preprocess, features = build_configs(
        repo, "demo", dataset_cfg, "web_demo_1", "job1", "s1", values)
    assert experiment["evaluation"]["unit"] == "recording"
    assert experiment["training"]["selection_metric"] == "recording_balanced_accuracy"
    assert experiment["model"]["hidden_dims"] == [64, 32]
    assert features["n_mfcc"] == 13
    assert preprocess["observe_seconds"] == 4.0
    paths = write_configs(repo, experiment, preprocess, features)
    written = yaml.safe_load((repo / paths["experiment"]).read_text(encoding="utf-8"))
    assert written["experiment_id"] == "web_demo_1"
    assert (repo / paths["preprocess"]).is_file()
    assert (repo / paths["features"]).is_file()


def test_catalog_collects_every_stage(tmp_path):
    repo, workspace, _ = _workspace(tmp_path)
    catalog = build_catalog(repo, workspace)
    assert len(catalog["datasets"]) == 1
    demo = catalog["datasets"][0]
    assert demo["grouping"] == "subject"
    assert demo["scans"][0]["scan_id"] == "scan1"
    assert demo["prepared"][0]["status"] == "approved"
    assert demo["splits"][0]["split_id"] == "s1"
    assert demo["runs"][0]["balanced_accuracy"] == 0.8
    assert demo["defaults"]["__prepared_id"] == "job1"
    assert demo["defaults"]["__split_id"] == "s1"


def test_server_exposes_schema_catalog_and_saves_experiment(tmp_path):
    repo, _, paths = _workspace(tmp_path)
    server = make_server(repo, paths, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path: str):
        with urllib.request.urlopen(base + path, timeout=20) as res:
            return res.status, res.read().decode("utf-8")

    def post(path: str, body: dict):
        req = urllib.request.Request(
            base + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as res:
            return res.status, json.loads(res.read().decode("utf-8"))

    try:
        status, text = get("/")
        assert status == 200 and "语音疾病识别" in text
        status, text = get("/api/schema")
        assert status == 200 and "features.n_mfcc" in text
        status, text = get("/api/catalog")
        assert status == 200 and "demo" in text

        status, saved = post("/api/experiment", {
            "dataset": "demo", "prepared_id": "job1", "split_id": "s1",
            "experiment_id": "web_demo_test",
            "values": {"features.n_mfcc": 13, "model.dropout": 0.3},
        })
        assert status == 201
        assert saved["feature_dim"] == 26
        assert (repo / saved["paths"]["experiment"]).is_file()

        with pytest.raises(urllib.error.HTTPError) as exc:
            post("/api/experiment", {
                "dataset": "demo", "prepared_id": "missing", "split_id": "s1",
                "values": {},
            })
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
