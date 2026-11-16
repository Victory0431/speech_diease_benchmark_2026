"""F08: the web console and the CLI share one read-only execution plan."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from sddbench.cli import main
from sddbench.modeling.experiment import load_experiment
from sddbench.modeling.plan import describe_plan

from test_modeling import _build_workspace, _experiment_cfg


def _paths_file(tmp_path: Path, workspace: Path) -> Path:
    path = tmp_path / "paths.yaml"
    path.write_text(yaml.safe_dump({
        "workspace_root": str(workspace),
        "dataset_roots": {"demo": str(tmp_path / "data")},
    }), encoding="utf-8")
    return path


def test_plan_is_ready_when_cache_matches_the_recipe(tmp_path):
    repo, workspace, _ = _build_workspace(tmp_path)
    cfg = load_experiment(_experiment_cfg(tmp_path))
    plan = describe_plan(cfg, workspace, repo)
    assert plan["ready"] is True, plan["errors"]
    assert plan["feature"]["status"] == "ready"
    assert plan["feature"]["dim"] == 40
    assert plan["feature"]["n_missing"] == 0
    assert plan["identities"]["processed_sample_rate"] == 16000
    assert plan["identities"]["n_samples"] == 32
    assert plan["identities"]["manifest_fingerprint"]
    assert plan["actions"] == []


def test_plan_reports_stale_cache_and_the_exact_build_command(tmp_path):
    repo, workspace, _ = _build_workspace(tmp_path)
    # A different recipe means a different feature hash: the existing cache is
    # not evidence for this plan and must not be reported as ready.
    feature_file = repo / "configs" / "features" / "mfcc20_stats.yaml"
    raw = yaml.safe_load(feature_file.read_text(encoding="utf-8"))
    raw["n_mfcc"] = 13
    feature_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    admission_file = workspace / "prepared" / "demo_v1" / "prep1" / "admission.json"
    admission = json.loads(admission_file.read_text(encoding="utf-8"))
    admission["scan_id"] = "scan1"
    admission_file.write_text(json.dumps(admission), encoding="utf-8")

    plan = describe_plan(load_experiment(_experiment_cfg(tmp_path)), workspace, repo)
    assert plan["ready"] is False
    assert plan["feature"]["status"] == "missing"
    assert plan["feature"]["dim"] == 26
    assert plan["feature"]["n_missing"] == 32
    assert not Path(plan["feature"]["cache_dir"]).exists()
    action = next(a for a in plan["actions"] if a["action"] == "build_features")
    assert "extract-features" in action["command"]
    assert "--manifest" in action["command"]
    assert "--cache-mode" in action["command"]
    assert "shared" in action["command"]
    assert str(workspace / "prepared" / "demo_v1" / "prep1" / "manifest.csv") in action["command"]


def test_cli_plan_exit_codes_match_the_web_verdict(tmp_path, capsys):
    repo, workspace, _ = _build_workspace(tmp_path)
    paths = _paths_file(tmp_path, workspace)
    experiment = _experiment_cfg(tmp_path)
    argv = ["plan", "--config", str(experiment), "--paths", str(paths),
            "--repo-root", str(repo)]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True

    feature_file = repo / "configs" / "features" / "mfcc20_stats.yaml"
    raw = yaml.safe_load(feature_file.read_text(encoding="utf-8"))
    raw["n_mfcc"] = 13
    feature_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert main(argv) == 3
    captured = capsys.readouterr()
    blocked = json.loads(captured.out)
    assert blocked["feature"]["status"] == "missing"
    assert "blocked" in captured.err  # the human-readable reason goes to stderr
