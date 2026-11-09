"""Publish small, comparable result digests and build a cross-dataset leaderboard.

Only whitelisted files leave the local workspace:
`metrics.json`, `metrics_per_class.csv`, `report.md`, `config.yaml`, `run_meta.json`
and `figures/*.png`. Weights, window-level predictions and input snapshots stay local.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .contract import (
    CONTRACT_VERSION,
    ResultsContractError,
    attach_contract,
    sanitize_run_meta,
    validate_metrics,
)

PUBLISHED_FILES = ("metrics.json", "metrics_per_class.csv", "report.md", "config.yaml")
FIGURE_SUFFIXES = (".png",)


def publish_run(run_dir: str | Path, dest_root: str | Path) -> Path:
    run_dir = Path(run_dir)
    dest_root = Path(dest_root)
    metrics_file = run_dir / "metrics.json"
    run_file = run_dir / "run.json"
    if not metrics_file.is_file() or not run_file.is_file():
        raise ResultsContractError(f"incomplete run directory: {run_dir}")

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    validate_metrics(metrics)
    run = json.loads(run_file.read_text(encoding="utf-8"))

    dataset = str(metrics["dataset"])
    run_id = str(metrics["run_id"])
    dest = dest_root / dataset / run_id
    dest.mkdir(parents=True, exist_ok=True)

    for name in PUBLISHED_FILES:
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    published_metrics = attach_contract(metrics)
    (dest / "metrics.json").write_text(
        json.dumps(published_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "run_meta.json").write_text(
        json.dumps(sanitize_run_meta(run), ensure_ascii=False, indent=2),
        encoding="utf-8")

    figures = run_dir / "figures"
    if figures.is_dir():
        target = dest / "figures"
        target.mkdir(exist_ok=True)
        for figure in sorted(figures.iterdir()):
            if figure.is_file() and figure.suffix.lower() in FIGURE_SUFFIXES:
                shutil.copy2(figure, target / figure.name)
    return dest


def publish_all(runs_root: str | Path, dest_root: str | Path) -> tuple[list[dict], list[dict]]:
    """Publish every complete run; return (published, failures)."""
    runs_root = Path(runs_root)
    published: list[dict] = []
    failures: list[dict] = []
    for metrics_file in sorted(runs_root.glob("*/metrics.json")):
        run_dir = metrics_file.parent
        try:
            dest = publish_run(run_dir, dest_root)
        except Exception as e:  # a bad run must not stop the archive
            failures.append({"run_dir": str(run_dir), "error": f"{type(e).__name__}: {e}"})
            continue
        published.append({"run_id": run_dir.name, "dest": str(dest)})
    build_leaderboard(dest_root)
    return published, failures


def build_leaderboard(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    rows: list[dict] = []
    for meta_file in sorted(results_root.glob("*/*/run_meta.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        metrics_file = meta_file.parent / "metrics.json"
        if not metrics_file.is_file():
            continue
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        data = meta.get("data") or {}
        counts = data.get("subject_counts") or {}
        subject = metrics["metrics"]["test"]["subject"]
        window = metrics["metrics"]["test"]["window"]
        model = meta.get("model") or {}
        training = meta.get("training") or {}
        row = {
            "dataset": meta.get("dataset"),
            "task_id": meta.get("task_id"),
            "experiment_id": meta.get("experiment_id"),
            "run_id": meta.get("run_id"),
            "seed": meta.get("seed"),
            "prepared_id": meta.get("prepared_id"),
            "split_id": meta.get("split_id"),
            "model": model.get("name"),
            "hidden_dims": json.dumps(model.get("hidden_dims"), ensure_ascii=False),
            "dropout": model.get("dropout"),
            "learning_rate": training.get("learning_rate"),
            "batch_size": training.get("batch_size"),
            "feature_dim": data.get("feature_dim"),
            "feature_key": (data.get("feature_cache") or {}).get("feature_key"),
            "train_subjects": counts.get("train"),
            "val_subjects": counts.get("val"),
            "test_subjects": counts.get("test"),
            "n_samples": data.get("n_samples"),
            "best_epoch": meta.get("best_epoch"),
            "best_val_subject_balanced_accuracy":
                meta.get("best_val_subject_balanced_accuracy"),
            "git_commit": (meta.get("git") or {}).get("commit"),
            "git_dirty": (meta.get("git") or {}).get("dirty"),
            "published_at": meta.get("published_at"),
            "contract_version": meta.get("contract_version"),
        }
        for prefix, block in (("subject", subject), ("window", window)):
            for metric in ("accuracy", "balanced_accuracy", "sensitivity",
                           "specificity", "precision", "f1", "auroc",
                           "average_precision", "log_loss", "brier", "n"):
                row[f"{prefix}_{metric}"] = block.get(metric)
            for key in ("tn", "fp", "fn", "tp"):
                row[f"{prefix}_{key}"] = (block.get("confusion") or {}).get(key)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["dataset", "run_id", "subject_balanced_accuracy"])
    else:
        frame = frame.sort_values(
            ["subject_balanced_accuracy", "subject_auroc"],
            ascending=False, na_position="last").reset_index(drop=True)
    frame.to_csv(results_root / "leaderboard.csv", index=False)
    (results_root / "leaderboard.md").write_text(_render(frame), encoding="utf-8")
    return frame


def _fmt(value, nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _render(frame: pd.DataFrame) -> str:
    lines = [
        "# 跨数据集基准榜（test / subject 为主指标）",
        "",
        f"- 结果契约版本：`{CONTRACT_VERSION}`",
        f"- 运行数：`{len(frame)}`",
        "- 排序：`subject_balanced_accuracy` 降序；同分看 `subject_auroc`。",
        "- 这里只含聚合指标与小体积图；权重、逐录音预测、患者级清单不在仓库。",
        "",
        "| # | dataset | run_id | test subjects | seed | dropout | bAcc | sens | spec | precision | F1 | AUROC | AP | log loss | Brier | window bAcc |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in frame.iterrows():
        lines.append(
            f"| {i + 1} | {row['dataset']} | {row['run_id']} | "
            f"{_fmt(row['test_subjects'])} | {_fmt(row['seed'])} | "
            f"{_fmt(row['dropout'])} | {_fmt(row['subject_balanced_accuracy'])} | "
            f"{_fmt(row['subject_sensitivity'])} | {_fmt(row['subject_specificity'])} | "
            f"{_fmt(row['subject_precision'])} | {_fmt(row['subject_f1'])} | "
            f"{_fmt(row['subject_auroc'])} | {_fmt(row['subject_average_precision'])} | "
            f"{_fmt(row['subject_log_loss'])} | {_fmt(row['subject_brier'])} | "
            f"{_fmt(row['window_balanced_accuracy'])} |")
    lines += [
        "",
        "> 主指标是 test 说话人级 balanced accuracy：同一人的录音先平均，再按人计算。",
        "> 只有个位数测试说话人时，一两人的差异就可能改变排序；跨数据集比较还需要",
        "> 多 seed、置信区间与同一套数据端参数，不能只看单行数字。",
        "",
    ]
    return "\n".join(lines)
