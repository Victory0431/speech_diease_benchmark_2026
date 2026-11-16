"""Read-only execution plan shared by the web console and the CLI.

`describe_plan` answers, before any expensive work:
  * do the prepared manifest / split / frozen task snapshot still verify?
  * which labels, grouping, processed rate and view does the run use?
  * does the feature cache exist for that exact recipe + view + rate + source content?

It calls the same `resolve_run_inputs` gate as training and the same
`inspect_cache` used by the loader, so the browser can never promise a run that
the CLI would refuse. When the cache is missing/stale it returns the exact
`extract-features` command that would build it. Nothing here writes to disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..features.registry import feature_dim
from .features import inspect_cache
from .trainer import resolve_run_inputs

CACHE_READY = "ready"


def _build_command(cfg: dict, repo_root: Path, workspace: Path,
                   admission: dict, manifest_file: Path,
                   paths_file: Path | None) -> list[str] | None:
    scan_id = str(admission.get("scan_id") or "")
    features_rel = (cfg.get("features") or {}).get("config")
    preprocess_rel = (cfg.get("features") or {}).get("preprocess")
    if not scan_id or not features_rel or not preprocess_rel:
        return None
    cmd = [
        sys.executable, "-m", "sddbench.cli", "extract-features",
        "--dataset", str(cfg["dataset"]),
        "--source", scan_id,
        "--preprocess", str((repo_root / preprocess_rel).resolve()),
        "--features", str((repo_root / features_rel).resolve()),
        "--manifest", str(Path(manifest_file).resolve()),
        "--cache-mode", "shared",
    ]
    if paths_file is not None:
        cmd += ["--paths", str(Path(paths_file).resolve()),
                "--repo-root", str(repo_root.resolve())]
    return cmd


def describe_plan(cfg: dict, workspace: str | Path, repo_root: str | Path,
                  *, paths_file: str | Path | None = None) -> dict:
    """Return {ready, status, errors, identities, feature, actions}.

    Expected blockers (missing manifest, stale cache, ...) are reported, not
    raised, so the web form can always show why training is unavailable.
    """
    workspace = Path(workspace)
    repo_root = Path(repo_root)
    plan: dict = {
        "status": "blocked",
        "ready": False,
        "errors": [],
        "identities": {},
        "feature": None,
        "actions": [],
    }
    try:
        data = resolve_run_inputs(cfg, workspace, repo_root)
    except Exception as exc:  # noqa: BLE001 - every blocker must be displayed
        plan["errors"].append(f"{type(exc).__name__}: {exc}")
        return plan

    feature_cfg = data["feature_cfg"]
    cache = inspect_cache(
        workspace, feature_cfg, data["sample_ids"], data["sample_rate"],
        dataset_id=str(cfg["dataset"]),
        source_hashes=data["source_hashes"],
        view_fingerprint=data["view_fingerprint"])
    cache["dim"] = feature_dim(feature_cfg)
    cache["feature_type"] = feature_cfg.get("feature_type")
    cache["recipe_name"] = feature_cfg.get("name")

    plan["identities"] = {
        "task_id": data["task_id"],
        "prepared_id": cfg["prepared_id"],
        "split_id": cfg["split_id"],
        "dataset": str(cfg["dataset"]),
        "n_samples": len(data["sample_ids"]),
        "task_hash": data["task_hash"],
        "manifest_fingerprint": data["manifest_fingerprint"],
        "split_fingerprint": data["split_fingerprint"],
        "preprocess_fingerprint": data["view_fingerprint"],
        "processed_sample_rate": int(data["sample_rate"]),
        "original_sample_rates": data["original_sample_rates"],
        "labels": {
            "class_order": data["class_order"],
            "positive_class": data["positive_class"],
        },
        "grouping": data["grouping"],
        "benchmark_eligible": data["benchmark_eligible"],
        "scientific_status": data["scientific_status"],
    }
    plan["feature"] = {
        "status": cache["status"],
        "reason": cache.get("reason"),
        "cache_dir": cache["cache_dir"],
        "feature_key": cache["feature_key"],
        "feature_hash": cache["feature_hash"],
        "dim": cache["dim"],
        "sample_rate": cache["sample_rate"],
        "view_fingerprint": cache["view_fingerprint"],
        "n_indexed": cache.get("n_indexed"),
        "n_expected": len(data["sample_ids"]),
        "n_missing": cache.get("n_missing"),
        "missing": cache.get("missing") or [],
    }
    if cache["status"] != CACHE_READY:
        plan["errors"].append(str(cache.get("reason") or "feature cache is not ready"))
        cmd = _build_command(cfg, repo_root, workspace, data["admitted"],
                             data["manifest_file"], paths_file)
        if cmd:
            plan["actions"].append({
                "action": "build_features",
                "reason": cache["status"],
                "command": cmd,
                "shell": " ".join(cmd),
            })
        else:
            plan["errors"].append(
                "cannot build features automatically: admission has no scan_id "
                "or the experiment lacks preprocessing/feature config paths")
        return plan

    plan["status"] = "ready"
    plan["ready"] = True
    return plan
