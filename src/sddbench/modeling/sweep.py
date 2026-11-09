"""D7: parameter sweeps and cross-run summaries.

A sweep is just an ordered list of `sddbench run` calls with dot-key overrides.
Nothing about the training protocol changes, so every comparison stays fair.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from .experiment import ExperimentConfigError, config_hash, expand_grid
from .trainer import run_experiment


class SweepError(RuntimeError):
    """Raised for invalid sweep definitions."""


def load_sweep(path: str | Path) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise SweepError(f"sweep config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        sweep = yaml.safe_load(f) or {}
    axes = sweep.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise SweepError("sweep config needs a non-empty `axes` mapping")
    for key, values in axes.items():
        if not isinstance(values, list) or not values:
            raise SweepError(f"axis {key!r} must be a non-empty list")
    return sweep


def sweep_id_for(cfg: dict, axes: dict) -> str:
    blob = json.dumps({"experiment": cfg["experiment_id"], "axes": axes,
                       "config_hash": config_hash(cfg)}, sort_keys=True,
                      ensure_ascii=False, default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
    return f"{time.strftime('%Y%m%dT%H%M%S')}_{cfg['experiment_id']}_{digest}"


def run_sweep(cfg: dict, sweep: dict, workspace: Path, repo_root: Path, *,
              run_root: Path | None = None,
              max_runs: int | None = None,
              on_progress=None) -> tuple[Path, list[dict]]:
    """Run every grid point in order and persist jobs.csv after each attempt."""
    axes = dict(sweep["axes"])
    combos = expand_grid(cfg, axes)
    if max_runs is not None:
        combos = combos[:max_runs]
    if not combos:
        raise SweepError("sweep grid produced no combinations")

    sid = str(sweep.get("sweep_id") or sweep_id_for(cfg, axes))
    out_dir = Path(workspace) / "sweeps" / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps({
        "sweep_id": sid,
        "experiment_id": cfg["experiment_id"],
        "axes": axes,
        "n_runs": len(combos),
        "max_runs": max_runs,
        "config_hash": config_hash(cfg),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs: list[dict] = []
    jobs_file = out_dir / "jobs.csv"
    for index, combo in enumerate(combos, start=1):
        overrides = _overrides(cfg, combo, axes)
        row = {"index": index, "status": "running", "run_id": None,
               "run_dir": None, "overrides": json.dumps(overrides, ensure_ascii=False),
               "test_subject_balanced_accuracy": None,
               "test_subject_auroc": None,
               "test_window_balanced_accuracy": None, "error": None}
        if on_progress:
            on_progress(index, len(combos), overrides)
        try:
            job = run_experiment(combo, workspace, repo_root, run_root=run_root)
        except Exception as e:  # keep the sweep going; the failure is recorded
            row["status"] = "failed"
            row["error"] = f"{type(e).__name__}: {e}"
            jobs.append(row)
            _write_jobs(jobs_file, jobs)
            continue
        subject = job["metrics"]["test"]["subject"]
        row.update({
            "status": "completed",
            "run_id": job["run_id"],
            "run_dir": job["out_dir"],
            "test_subject_balanced_accuracy": subject["balanced_accuracy"],
            "test_subject_auroc": subject["auroc"],
            "test_window_balanced_accuracy":
                job["metrics"]["test"]["window"]["balanced_accuracy"],
        })
        jobs.append(row)
        _write_jobs(jobs_file, jobs)
    return out_dir, jobs


def _overrides(base: dict, combo: dict, axes: dict) -> dict:
    """Report only the values that differ from the base config."""
    out = {}
    for key in axes:
        section, field = key.split(".", 1)
        value = combo.get(section, {}).get(field)
        if value != base.get(section, {}).get(field):
            out[key] = value
    return out


def _write_jobs(path: Path, jobs: list[dict]) -> None:
    pd.DataFrame(jobs).to_csv(path, index=False)


def summarize_runs(runs_root: str | Path, out_dir: str | Path) -> pd.DataFrame:
    """Collect every run's metrics into one table (no patient-level rows)."""
    runs_root = Path(runs_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for metrics_file in sorted(runs_root.glob("*/metrics.json")):
        run_dir = metrics_file.parent
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        run = {}
        run_file = run_dir / "run.json"
        if run_file.is_file():
            run = json.loads(run_file.read_text(encoding="utf-8"))
        cfg: dict = {}
        config_file = run_dir / "config.yaml"
        if config_file.is_file():
            cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        if not cfg:
            cfg = {"model": run.get("model") or {},
                   "training": run.get("training") or {}}
        model = cfg.get("model") or {}
        training = cfg.get("training") or {}
        test = metrics.get("metrics", {}).get("test", {})
        subj = test.get("subject") or {}
        win = test.get("window") or {}
        rows.append({
            "run_id": metrics.get("run_id") or run.get("run_id"),
            "experiment_id": metrics.get("experiment_id") or run.get("experiment_id"),
            "task_id": metrics.get("task_id") or run.get("task_id"),
            "prepared_id": run.get("prepared_id"),
            "split_id": run.get("split_id"),
            "hidden_dims": json.dumps(model.get("hidden_dims"), ensure_ascii=False),
            "dropout": model.get("dropout"),
            "activation": model.get("activation"),
            "learning_rate": training.get("learning_rate"),
            "batch_size": training.get("batch_size"),
            "weight_decay": training.get("weight_decay"),
            "seed": training.get("seed"),
            "best_epoch": run.get("best_epoch"),
            "test_subject_balanced_accuracy": subj.get("balanced_accuracy"),
            "test_subject_sensitivity": subj.get("sensitivity"),
            "test_subject_specificity": subj.get("specificity"),
            "test_subject_auroc": subj.get("auroc"),
            "test_subject_f1": subj.get("f1"),
            "test_window_balanced_accuracy": win.get("balanced_accuracy"),
            "test_window_auroc": win.get("auroc"),
            "elapsed_sec": run.get("elapsed_sec"),
            "git_commit": (run.get("git") or {}).get("commit"),
            "git_dirty": (run.get("git") or {}).get("dirty"),
            "status": run.get("status", "unknown"),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["run_id", "test_subject_balanced_accuracy"])
    else:
        frame = frame.sort_values(
            ["test_subject_balanced_accuracy", "test_subject_auroc"],
            ascending=False, na_position="last").reset_index(drop=True)
    frame.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(_render_summary(frame), encoding="utf-8")
    return frame


def _fmt(value, nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _render_summary(frame: pd.DataFrame) -> str:
    lines = [
        "# 实验结果汇总",
        "",
        f"- 运行数：`{len(frame)}`",
        "- 排序：test subject balanced accuracy 降序（仅用于浏览，不替代统计检验）。",
        "- 本文件只含聚合指标；逐录音预测、权重与患者级清单留在本机 `artifacts/runs/`。",
        "",
        "| # | run_id | hidden | dropout | lr | seed | best epoch | test subj bAcc | sens | spec | AUROC | test win bAcc | status |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, row in frame.iterrows():
        lines.append(
            f"| {i + 1} | {row['run_id']} | {_fmt(row['hidden_dims'])} | "
            f"{_fmt(row['dropout'])} | {_fmt(row['learning_rate'])} | "
            f"{_fmt(row['seed'])} | {_fmt(row['best_epoch'])} | "
            f"{_fmt(row['test_subject_balanced_accuracy'])} | "
            f"{_fmt(row['test_subject_sensitivity'])} | "
            f"{_fmt(row['test_subject_specificity'])} | "
            f"{_fmt(row['test_subject_auroc'])} | "
            f"{_fmt(row['test_window_balanced_accuracy'])} | {_fmt(row['status'])} |")
    lines += [
        "",
        "> test 只有个位数说话人时，一两人的差异就可能改变排序；"
        "把这里的排序当作「下一轮实验的线索」，不要当作最终结论。",
        "",
    ]
    return "\n".join(lines)
