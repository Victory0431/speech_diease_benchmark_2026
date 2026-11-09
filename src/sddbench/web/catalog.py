"""D12: read-only catalog of datasets, prepared jobs, splits, runs and results.

The web console never scans dataset roots itself: it only reads the workspace
artifacts that the CLI already produced.  Every entry keeps the id a human
would type on the command line, so the page and the terminal never diverge.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ..modeling.experiment import _merge, DEFAULTS
from .schema import dataset_defaults


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def _read_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _recent(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)


def _dataset_configs(repo_root: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted((repo_root / "configs" / "datasets").glob("*.yaml")):
        cfg = _read_yaml(path)
        if not cfg:
            continue
        dataset_id = str(cfg.get("dataset_id") or path.stem.replace("_v1", ""))
        cfg["_dataset_id"] = dataset_id
        cfg["_config_path"] = str(path.relative_to(repo_root))
        docs.append(cfg)
    return docs


def _scans(workspace: Path, dataset_id: str) -> list[dict]:
    out = []
    for scan_dir in _recent(workspace / "inventory" / dataset_id):
        if not scan_dir.is_dir():
            continue
        meta = _read_json(scan_dir / "meta.json")
        if not meta:
            continue
        stats = _read_json(scan_dir / "statistics.json")
        out.append({
            "scan_id": meta.get("scan_id", scan_dir.name),
            "mode": meta.get("mode"),
            "status": meta.get("status"),
            "finished_at": meta.get("finished_at"),
            "elapsed_sec": meta.get("elapsed_sec"),
            "counts": (stats or {}).get("counts", {}),
        })
    return out


def _prepared(workspace: Path, task_id: str) -> list[dict]:
    out = []
    for job_dir in _recent(workspace / "prepared" / task_id):
        doc = _read_json(job_dir / "admission.json")
        if not doc:
            continue
        counts = dict(doc.get("counts") or {})
        if "n_groups" not in counts:
            counts["n_groups"] = (counts.get("n_subjects")
                                  if (doc.get("grouping") or "subject") == "subject"
                                  else counts.get("n_manifest"))
        out.append({
            "job_id": doc.get("job_id", job_dir.name),
            "status": doc.get("status"),
            "grouping": doc.get("grouping"),
            "counts": counts,
            "started_at": doc.get("started_at"),
            "finished_at": doc.get("finished_at"),
            "has_manifest": (job_dir / "manifest.csv").is_file(),
            "manifest_csv": str((job_dir / "manifest.csv")),
        })
    return out


def _splits(workspace: Path, task_id: str) -> list[dict]:
    out = []
    for split_dir in _recent(workspace / "splits" / task_id):
        doc = _read_json(split_dir / "split.json")
        if not doc:
            continue
        out.append({
            "split_id": doc.get("split_id", split_dir.name),
            "seed": doc.get("seed"),
            "fractions": doc.get("fractions"),
            "prepared_id": doc.get("prepared_id"),
            "summary": doc.get("summary", {}),
            "created_at": doc.get("created_at"),
            "n_groups": (doc.get("summary") or {}).get(
                "n_groups", (doc.get("summary") or {}).get("n_subjects")),
        })
    return out


def _experiments(repo_root: Path) -> list[dict]:
    out = []
    for path in sorted((repo_root / "configs" / "experiments").rglob("*.yaml")):
        cfg = _read_yaml(path)
        if not cfg.get("dataset"):
            continue
        merged = _merge(DEFAULTS, cfg)
        out.append({
            "experiment_id": merged.get("experiment_id"),
            "dataset": merged.get("dataset"),
            "prepared_id": merged.get("prepared_id"),
            "split_id": merged.get("split_id"),
            "model": merged.get("model"),
            "training": merged.get("training"),
            "evaluation": merged.get("evaluation"),
            "path": str(path.relative_to(repo_root)),
            "generated": "generated" in path.parts,
            "mtime": path.stat().st_mtime,
        })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _runs(workspace: Path, limit: int = 300) -> list[dict]:
    out = []
    for run_dir in _recent(workspace / "runs")[:limit]:
        if not run_dir.is_dir():
            continue
        run = _read_json(run_dir / "run.json")
        if not run:
            continue
        metrics = _read_json(run_dir / "metrics.json")
        unit = metrics.get("primary_unit") or metrics.get("evaluation_unit") or "subject"
        block = ((metrics.get("metrics") or {}).get("test") or {}).get(unit) or {}
        figures = sorted(p.name for p in (run_dir / "figures").glob("*.png")) \
            if (run_dir / "figures").is_dir() else []
        out.append({
            "run_id": run.get("run_id", run_dir.name),
            "dataset": run.get("dataset") or metrics.get("dataset"),
            "experiment_id": run.get("experiment_id"),
            "prepared_id": run.get("prepared_id"),
            "seed": run.get("seed"),
            "split_id": run.get("split_id"),
            "primary_unit": unit,
            "grouping": run.get("grouping"),
            "balanced_accuracy": block.get("balanced_accuracy"),
            "auroc": block.get("auroc"),
            "sensitivity": block.get("sensitivity"),
            "specificity": block.get("specificity"),
            "best_epoch": run.get("best_epoch"),
            "created_at": run.get("created_at") or run.get("finished_at"),
            "figures": figures,
            "report": str(run_dir / "report.md"),
        })
    return out


def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        frame = pd.read_csv(path).fillna("")
        return frame.to_dict("records")
    except Exception:
        return []


def build_catalog(repo_root: str | Path, workspace_root: str | Path) -> dict:
    repo_root = Path(repo_root).resolve()
    workspace = Path(workspace_root).resolve()
    experiments = _experiments(repo_root)
    runs = _runs(workspace)
    datasets = []
    for cfg in _dataset_configs(repo_root):
        dataset_id = cfg["_dataset_id"]
        task_id = str(cfg.get("task_id") or f"{dataset_id}_v1")
        ds_experiments = [e for e in experiments if e["dataset"] == dataset_id]
        defaults = dataset_defaults(repo_root, dataset_id, cfg,
                                    ds_experiments[0] if ds_experiments else None)
        latest_run = next((r for r in runs if r["dataset"] == dataset_id), None)
        defaults["__prepared_id"] = (
            next((p["job_id"] for p in _prepared(workspace, task_id)
                  if p["status"] == "approved" and p["has_manifest"]), "")
            or (latest_run or {}).get("prepared_id", ""))
        splits = _splits(workspace, task_id)
        defaults["__split_id"] = ((latest_run or {}).get("split_id")
                                  or (splits[0]["split_id"] if splits else ""))
        datasets.append({
            "id": dataset_id,
            "task_id": task_id,
            "adapter": cfg.get("adapter"),
            "grouping": cfg.get("grouping") or "subject",
            "evaluation_unit": cfg.get("evaluation_unit"),
            "labels": cfg.get("labels") or {},
            "selection": cfg.get("selection") or {},
            "naming": cfg.get("naming") or {},
            "config_path": cfg["_config_path"],
            "raw_config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "scans": _scans(workspace, dataset_id),
            "prepared": _prepared(workspace, task_id),
            "splits": splits,
            "experiments": ds_experiments,
            "runs": [r for r in runs if r["dataset"] == dataset_id],
            "defaults": defaults,
        })
    return {
        "repo_root": str(repo_root),
        "workspace_root": str(workspace),
        "datasets": datasets,
        "experiments": experiments,
        "runs": runs,
        "results": {
            "leaderboard": _csv_rows(repo_root / "results_summary" / "leaderboard.csv"),
            "aggregates": _csv_rows(repo_root / "results_summary" / "aggregates.csv"),
            "split_sensitivity": _csv_rows(
                repo_root / "results_summary" / "split_sensitivity.csv"),
        },
    }


def find_dataset(catalog: dict, dataset_id: str) -> dict | None:
    return next((d for d in catalog["datasets"] if d["id"] == dataset_id), None)
