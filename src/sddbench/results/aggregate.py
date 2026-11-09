"""D9: cross-seed aggregation and uncertainty for published runs.

Reads local run packs (``artifacts/runs/<run_id>/``), groups them by
(dataset, experiment_id, primary_unit) and reports:

* per-run test metrics on the primary unit;
* mean/std/median/min/max across seeds;
* a percentile bootstrap CI over runs;
* an optional within-run subject bootstrap CI from ``predictions_subjects.csv``
  (only possible locally: patient-level rows are never published).

Outputs ``aggregate.json``, ``aggregate.md`` and two small figures under
``<out>/<dataset>/aggregates/<group>/`` plus a top-level ``aggregates.csv/md``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..modeling.metrics import binary_metrics

AGGREGATE_VERSION = "1.0"
METRICS = ("balanced_accuracy", "auroc", "sensitivity", "specificity",
           "precision", "f1", "average_precision", "log_loss", "brier")


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_") or "all"


def _signature(run: dict, include_split: bool = True) -> str:
    """Everything that makes a run the same experiment except the training seed."""
    training = dict(run.get("training") or {})
    training.pop("seed", None)
    payload = {
        "model": run.get("model"),
        "training": training,
    }
    if include_split:
        payload["split_id"] = run.get("split_id")
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def load_runs(runs_root: str | Path) -> list[dict]:
    """Every complete run pack, with its primary test block resolved."""
    runs_root = Path(runs_root)
    runs: list[dict] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "metrics.json"
        run_file = run_dir / "run.json"
        if not (metrics_file.is_file() and run_file.is_file()):
            continue
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        run = json.loads(run_file.read_text(encoding="utf-8"))
        unit = metrics.get("primary_unit") or metrics.get("evaluation_unit") or "subject"
        block = metrics["metrics"]["test"][unit]
        runs.append({
            "run_dir": run_dir, "run_id": metrics.get("run_id", run_dir.name),
            "metrics": metrics, "run": run, "unit": unit, "block": block,
            "dataset": metrics.get("dataset"), "experiment_id": run.get("experiment_id"),
            "seed": run.get("seed"), "split_id": run.get("split_id"),
            "signature": _signature(run),
        })
    return runs


def _group_runs(runs: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for item in runs:
        key = (item["dataset"], item["experiment_id"], item["unit"],
               item["signature"])
        groups.setdefault(key, []).append(item)
    for items in groups.values():
        items.sort(key=lambda r: (r["seed"] if r["seed"] is not None else -1,
                                  str(r["run_id"])))
    return groups


def bootstrap_ci(values: list[float | None], b: int, seed: int) -> list[float] | None:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(b, arr.size))
    means = arr[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def subject_bootstrap(run_dir: Path, class_order: list[str], b: int,
                      seed: int) -> dict | None:
    """Resample test subjects with replacement; None when rows are unavailable."""
    path = Path(run_dir) / "predictions_subjects.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    test = frame[frame["split"].astype(str) == "test"]
    if test.empty or len(class_order) != 2:
        return None
    prob_col = f"prob_{class_order[1]}"
    if prob_col not in test.columns:
        return None
    y = (test["y_true"].astype(str) == class_order[1]).astype(int).to_numpy()
    p = test[prob_col].to_numpy(dtype=float)
    if len(y) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(b, len(y)))
    bacc: list[float] = []
    auroc: list[float] = []
    for row in idx:
        m = binary_metrics(y[row], p[row], 1)
        if m["balanced_accuracy"] is not None:
            bacc.append(float(m["balanced_accuracy"]))
        if m["auroc"] is not None:
            auroc.append(float(m["auroc"]))

    def _ci(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "n_groups": int(len(y)),
        "balanced_accuracy_ci95": _ci(bacc),
        "auroc_ci95": _ci(auroc),
    }


def _summary(values: list[float | None], b: int, seed: int) -> dict:
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return {"mean": None, "std": None, "median": None, "min": None,
                "max": None, "ci95_runs": None, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_runs": bootstrap_ci(arr, b, seed),
        "n": len(arr),
    }


def _write_figures(group_dir: Path, runs: list[dict], unit: str,
                   primary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = group_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    values = [r["block"]["balanced_accuracy"] for r in runs]
    labels = [f"s{r['seed']}" if r["seed"] is not None else "s?" for r in runs]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.boxplot([v for v in values if v is not None], widths=0.4)
    xs = np.arange(1, len(values) + 1)
    ax.scatter(xs, values, color="#2563eb", zorder=3, s=28)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(f"test {unit} balanced accuracy")
    ax.set_title("per-seed primary metric")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "seed_boxplot.png", dpi=150)
    plt.close(fig)

    metrics = [m for m in METRICS if primary.get(m, {}).get("mean") is not None]
    means = [primary[m]["mean"] for m in metrics]
    lows = []
    highs = []
    for m in metrics:
        ci = primary[m]["ci95_runs"]
        lows.append(means[metrics.index(m)] - (ci[0] if ci else means[metrics.index(m)]))
        highs.append((ci[1] if ci else means[metrics.index(m)]) - means[metrics.index(m)])
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ys = np.arange(len(metrics))
    ax.errorbar(means, ys, xerr=[lows, highs], fmt="o", capsize=3, color="#1f2937")
    ax.set_yticks(ys)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("mean across seeds with 95% bootstrap interval")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "ci_compare.png", dpi=150)
    plt.close(fig)


def aggregate_runs(runs_root: str | Path, out_root: str | Path,
                   bootstrap: int = 2000, seed: int = 0) -> pd.DataFrame:
    runs_root = Path(runs_root)
    out_root = Path(out_root)
    runs = load_runs(runs_root)
    groups = _group_runs(runs)
    rows: list[dict] = []

    for (dataset, experiment_id, unit, signature), items in sorted(
            groups.items(), key=str):
        group_name = f"{_slug(experiment_id)}__{_slug(unit)}__{signature}"
        group_dir = out_root / str(dataset) / "aggregates" / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        per_run: list[dict] = []
        for item in items:
            doc = {
                "run_id": item["run_id"], "seed": item["seed"],
                "best_epoch": item["run"].get("best_epoch"),
                "test_groups": item["block"].get("n"),
            }
            for metric in METRICS:
                doc[metric] = item["block"].get(metric)
            boot = subject_bootstrap(item["run_dir"],
                                     item["metrics"]["class_order"], bootstrap, seed)
            if boot:
                doc["subject_bootstrap"] = boot
            per_run.append(doc)

        primary = {m: _summary([r[m] for r in per_run], bootstrap, seed)
                   for m in METRICS}
        payload = {
            "aggregate_version": AGGREGATE_VERSION,
            "dataset": dataset, "experiment_id": experiment_id,
            "primary_unit": unit, "signature": signature,
            "split_id": items[0]["split_id"],
            "dropout": (items[0]["run"].get("model") or {}).get("dropout"),
            "contract_version": items[0]["metrics"].get("contract_version"),
            "seeds": [r["seed"] for r in per_run],
            "n_runs": len(per_run),
            "primary": primary,
            "runs": per_run,
            "notes": [
                "ci95_runs is a percentile bootstrap over seeds (descriptive only).",
                "subject_bootstrap resamples test groups locally and is never published.",
            ],
        }
        (group_dir / "aggregate.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (group_dir / "aggregate.md").write_text(
            _render_group(payload), encoding="utf-8")
        _write_figures(group_dir, items, unit, primary)

        row = {
            "dataset": dataset, "experiment_id": experiment_id,
            "primary_unit": unit, "signature": signature,
            "split_id": items[0]["split_id"],
            "dropout": (items[0]["run"].get("model") or {}).get("dropout"),
            "n_runs": len(per_run),
            "seeds": "/".join("?" if s is None else str(s) for s in payload["seeds"]),
            "group_dir": str(group_dir.relative_to(out_root)),
        }
        for metric in ("balanced_accuracy", "auroc", "f1"):
            row[f"{metric}_mean"] = primary[metric]["mean"]
            row[f"{metric}_ci_low"] = (primary[metric]["ci95_runs"] or [None])[0]
            row[f"{metric}_ci_high"] = (primary[metric]["ci95_runs"] or [None, None])[1]
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_root / "aggregates.csv", index=False)
    (out_root / "aggregates.md").write_text(_render_index(frame), encoding="utf-8")
    _write_split_sensitivity(runs, out_root)
    return frame


def _write_split_sensitivity(runs: list[dict], out_root: Path) -> pd.DataFrame:
    """Per-split means for the same config: how much the partition moves results."""
    buckets: dict[tuple, dict[str, list[dict]]] = {}
    for item in runs:
        key = (item["dataset"], item["experiment_id"], item["unit"],
               _signature(item["run"], include_split=False))
        by_split = buckets.setdefault(key, {})
        by_split.setdefault(str(item["split_id"]), []).append(item)

    rows: list[dict] = []
    for (dataset, experiment_id, unit, signature), by_split in \
            sorted(buckets.items(), key=str):
        if len(by_split) < 2:
            continue
        values: list[float] = []
        for split_id, items in sorted(by_split.items()):
            baccs = [float(i["block"]["balanced_accuracy"]) for i in items
                     if i["block"].get("balanced_accuracy") is not None]
            if not baccs:
                continue
            values.extend(baccs)
            rows.append({
                "dataset": dataset, "experiment_id": experiment_id,
                "primary_unit": unit, "signature": signature,
                "split_id": split_id, "n_runs": len(baccs),
                "seeds": "/".join("?" if i["seed"] is None else str(i["seed"])
                                  for i in items),
                "balanced_accuracy_mean": float(np.mean(baccs)),
                "balanced_accuracy_min": float(np.min(baccs)),
                "balanced_accuracy_max": float(np.max(baccs)),
            })
        if values:
            for row in rows:
                if row["signature"] == signature and row["dataset"] == dataset:
                    row["across_splits_mean"] = float(np.mean(values))
                    row["across_splits_range"] = float(np.max(values) - np.min(values))

    frame = pd.DataFrame(rows)
    frame.to_csv(out_root / "split_sensitivity.csv", index=False)
    if frame.empty:
        (out_root / "split_sensitivity.md").write_text(
            "# 划分敏感性\n\n当前没有同一配置下的多个划分，暂无对比。\n",
            encoding="utf-8")
        return frame
    lines = [
        "# 划分敏感性（同一配置，换划分 seed）",
        "",
        "| dataset | experiment | unit | split | runs | seeds | bAcc mean | across-split mean | range |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['experiment_id']} | {row['primary_unit']} | "
            f"{row['split_id']} | {row['n_runs']} | {row['seeds']} | "
            f"{_fmt(row['balanced_accuracy_mean'])} | {_fmt(row['across_splits_mean'])} | "
            f"{_fmt(row['across_splits_range'])} |")
    lines += [
        "",
        "> 同一配置换划分后主指标的位移，是“6 个测试单元够不够”的直接证据；",
        "> 位移大于训练 seed 波动时，优先扩充数据集/换更稳的评价协议，而不是调超参。",
        "",
    ]
    (out_root / "split_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.3f}"


def _render_group(payload: dict) -> str:
    unit = payload["primary_unit"]
    lines = [
        f"# {payload['dataset']} / {payload['experiment_id']}：多 seed 汇总",
        "",
        f"- 主评价单位：**{unit}**；运行数：`{payload['n_runs']}`；"
        f"seeds：`{payload['seeds']}`",
        f"- 划分：`{payload['split_id']}`；配置签名：`{payload['signature']}`；"
        f"dropout：`{_fmt(payload.get('dropout'))}`",
        f"- 结果契约：`{payload['contract_version'] or 'unknown'}`；"
        f"聚合版本：`{payload['aggregate_version']}`",
        "- 跨 seed 的 95% 区间是对 seed 做百分位 bootstrap，只描述波动，不做显著性宣称。",
        "",
        "| run | seed | test groups | bAcc | AUROC | F1 | sens | spec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["runs"]:
        lines.append(
            f"| {run['run_id']} | {_fmt(run['seed'])} | {_fmt(run['test_groups'])} | "
            f"{_fmt(run['balanced_accuracy'])} | {_fmt(run['auroc'])} | "
            f"{_fmt(run['f1'])} | {_fmt(run['sensitivity'])} | "
            f"{_fmt(run['specificity'])} |")
    lines += [
        "",
        "## 跨 seed 汇总",
        "",
        "| metric | mean | std | median | min | max | 95% CI (runs) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in METRICS:
        m = payload["primary"][metric]
        ci = m["ci95_runs"]
        ci_text = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "—"
        lines.append(
            f"| {metric} | {_fmt(m['mean'])} | {_fmt(m['std'])} | "
            f"{_fmt(m['median'])} | {_fmt(m['min'])} | {_fmt(m['max'])} | {ci_text} |")
    lines += [
        "",
        "> 区间宽说明测试单元少：一两个人/录音的变动就足以改变结论。",
        "",
    ]
    return "\n".join(lines)


def _render_index(frame: pd.DataFrame) -> str:
    lines = [
        "# 多 seed 汇总（跨数据集）",
        "",
        "| dataset | experiment | primary unit | split | dropout | runs | seeds | bAcc mean | bAcc 95% CI | AUROC mean |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for _, row in frame.iterrows():
        ci = (f"[{_fmt(row['balanced_accuracy_ci_low'])}, "
              f"{_fmt(row['balanced_accuracy_ci_high'])}]")
        lines.append(
            f"| {row['dataset']} | {row['experiment_id']} | {row['primary_unit']} | "
            f"{row['split_id']} | {_fmt(row['dropout'])} | {row['n_runs']} | {row['seeds']} | "
            f"{_fmt(row['balanced_accuracy_mean'])} | {ci} | "
            f"{_fmt(row['auroc_mean'])} |")
    lines += [
        "",
        "> 只有 dataset、experiment、primary unit、划分与超参签名都相同的运行才会被合并；",
        "> 每个分组的详细结果在同目录 `aggregates/<experiment>__<unit>__<签名>/`。",
        "",
    ]
    return "\n".join(lines)
