"""Train/evaluate the frozen-feature MLP and write a self-contained result pack."""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from .. import __version__
from ..data.windows import normalize_preprocess, preprocess_fingerprint
from ..features.mfcc import normalize_feature_config
from ..results.contract import CONTRACT_VERSION, validate_metrics
from .experiment import config_hash
from .features import load_features
from .metrics import binary_metrics, per_class_rows
from .mlp import MLP
from .report import render_run_report, write_figures


class RunError(RuntimeError):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _git_commit(repo_root: Path) -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _load_task_config(repo_root: Path, dataset: str, task_id: str) -> dict:
    path = repo_root / "configs" / "datasets" / f"{task_id}.yaml"
    if not path.is_file():
        path = repo_root / "configs" / "datasets" / f"{dataset}_v1.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_run_data(cfg: dict, workspace: Path, repo_root: Path) -> dict:
    task_id = cfg.get("task_id") or f"{cfg['dataset']}_v1"
    prepared_dir = workspace / "prepared" / task_id / cfg["prepared_id"]
    split_dir = workspace / "splits" / task_id / cfg["split_id"]
    manifest_file = prepared_dir / "manifest.csv"
    split_file = split_dir / "split.csv"
    if not manifest_file.is_file():
        raise RunError(f"manifest not found: {manifest_file}")
    if not split_file.is_file():
        raise RunError(f"split not found: {split_file}")

    manifest = pd.read_csv(manifest_file)
    split = pd.read_csv(split_file)
    split_meta = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    admission_file = prepared_dir / "admission.json"
    if admission_file.is_file():
        admitted = json.loads(admission_file.read_text(encoding="utf-8"))
        if admitted.get("manifest_fingerprint") and \
                split_meta.get("manifest_fingerprint") != admitted["manifest_fingerprint"]:
            raise RunError("split was built from a different manifest fingerprint; "
                           "rerun `sddbench split`")
    for column in ("sample_id", "group_id", "label"):
        if column not in manifest.columns:
            raise RunError(f"manifest missing {column}")
        if column not in split.columns:
            raise RunError(f"split missing {column}")
    merged = manifest.merge(split[["sample_id", "group_id", "label", "split"]],
                            on="sample_id", suffixes=("", "_split"),
                            validate="one_to_one")
    if len(merged) != len(manifest):
        raise RunError("manifest/split join lost rows")
    if (merged["group_id"] != merged["group_id_split"]).any():
        raise RunError("split group_id disagrees with manifest")
    if (merged["label"] != merged["label_split"]).any():
        raise RunError("split label disagrees with manifest")

    rates = merged["sample_rate"].dropna().astype(int).unique().tolist()
    if len(rates) != 1:
        raise RunError(f"MLP baseline needs one actual sample rate, found {rates}")
    sample_rate = int(rates[0])

    task_cfg = _load_task_config(repo_root, cfg["dataset"], task_id)
    labels_cfg = task_cfg.get("labels") or {}
    grouping = str(task_cfg.get("grouping") or "subject")
    if grouping not in ("subject", "file"):
        raise RunError(f"task {task_id} grouping must be subject|file, got {grouping!r}")
    class_order = list((cfg.get("labels") or {}).get("class_order")
                       or labels_cfg.get("class_order")
                       or ["control", "disorder"])
    observed = sorted(merged["label"].astype(str).unique().tolist())
    if set(observed) != set(class_order):
        raise RunError(f"labels in manifest {observed} != class_order {class_order}")
    positive = (cfg.get("labels") or {}).get("positive_class") \
        or labels_cfg.get("positive_class") or class_order[-1]
    if positive not in class_order:
        raise RunError(f"positive_class {positive!r} not in {class_order}")

    with (repo_root / cfg["features"]["config"]).open("r", encoding="utf-8") as f:
        feature_cfg = normalize_feature_config(yaml.safe_load(f))
    preprocess_rel = (cfg.get("features") or {}).get("preprocess")
    if not preprocess_rel:
        raise RunError("experiment config needs features.preprocess so the loader can "
                       "bind the exact observation window used by the D3 job")
    with (repo_root / preprocess_rel).open("r", encoding="utf-8") as f:
        preprocess_cfg = normalize_preprocess(yaml.safe_load(f) or {})
    view_fingerprint = preprocess_fingerprint(preprocess_cfg)
    sample_ids = merged["sample_id"].astype(str).tolist()
    X, cache_info = load_features(workspace, feature_cfg, sample_ids, sample_rate,
                                  view_fingerprint=view_fingerprint)

    return {
        "task_id": task_id, "manifest": manifest, "split": split,
        "merged": merged, "X": X, "cache_info": cache_info,
        "class_order": class_order, "positive_class": positive,
        "positive_index": class_order.index(positive),
        "sample_rate": sample_rate, "feature_cfg": feature_cfg,
        "grouping": grouping,
        "manifest_file": manifest_file, "split_file": split_file,
        "split_dir": split_dir,
    }


def _subject_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    n_subjects = len(counts)
    weights = np.array([1.0 / (n_subjects * counts[g]) for g in groups], dtype=np.float64)
    return weights * len(weights) / weights.sum()


def _predict(model: nn.Module, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _aggregate_subjects(frame: pd.DataFrame, probs: np.ndarray,
                        class_order: list[str]) -> pd.DataFrame:
    tmp = frame[["group_id", "label", "split"]].copy()
    for i, name in enumerate(class_order):
        tmp[f"prob_{name}"] = probs[:, i]
    agg = tmp.groupby(["group_id", "label", "split"], as_index=False).agg(
        n_recordings=("group_id", "size"),
        **{f"prob_{name}": (f"prob_{name}", "mean") for name in class_order})
    return agg


def _subject_metric(frame: pd.DataFrame, probs: np.ndarray,
                    class_order: list[str], positive_index: int) -> dict:
    agg = _aggregate_subjects(frame, probs, class_order)
    y = (agg["label"].to_numpy() == class_order[positive_index]).astype(int)
    p = agg[[f"prob_{name}" for name in class_order]].to_numpy()
    return binary_metrics(y, p, positive_index)


def _level_metric(frame: pd.DataFrame, probs: np.ndarray, class_order: list[str],
                  positive_index: int, unit: str) -> dict:
    if unit == "recording":
        y = (frame["label"].to_numpy() == class_order[positive_index]).astype(int)
        return binary_metrics(y, probs, positive_index)
    return _subject_metric(frame, probs, class_order, positive_index)


def _evaluate(frame: pd.DataFrame, probs: np.ndarray, class_order: list[str],
              positive_index: int) -> dict:
    y = (frame["label"].to_numpy() == class_order[positive_index]).astype(int)
    return {
        "window": binary_metrics(y, probs, positive_index),
        "recording": binary_metrics(y, probs, positive_index),
        "subject": _subject_metric(frame, probs, class_order, positive_index),
    }


def run_experiment(cfg: dict, workspace: Path, repo_root: Path,
                   run_root: Path | None = None) -> dict:
    started = _now()
    data = load_run_data(cfg, workspace, repo_root)
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    device = str(cfg["training"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RunError("device=cuda requested but CUDA is not available")

    unit = str((cfg.get("evaluation") or {}).get("unit", "subject"))
    if unit not in ("subject", "recording"):
        raise RunError(f"evaluation.unit must be subject|recording, got {unit!r}")
    grouping = data["grouping"]
    if grouping == "file" and unit != "recording":
        raise RunError("grouping=file means one recording per group; "
                       "set evaluation.unit=recording")
    default_selection = f"{unit}_balanced_accuracy"
    selection_metric = str(cfg["training"].get("selection_metric", default_selection))
    if selection_metric != default_selection:
        raise RunError(f"selection_metric must be {default_selection!r} for "
                       f"evaluation.unit={unit!r}, got {selection_metric!r}")

    merged = data["merged"]
    X = data["X"]
    class_order = data["class_order"]
    positive_index = data["positive_index"]
    y = np.array([class_order.index(v) for v in merged["label"].astype(str)], dtype=int)
    groups = merged["group_id"].astype(str).to_numpy()
    splits = merged["split"].astype(str).to_numpy()
    train_mask, val_mask, test_mask = splits == "train", splits == "val", splits == "test"
    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not mask.any():
            raise RunError(f"split {name} is empty")
    if len(set(groups[train_mask]) & set(groups[val_mask])) or \
            len(set(groups[train_mask]) & set(groups[test_mask])) or \
            len(set(groups[val_mask]) & set(groups[test_mask])):
        raise RunError("group leakage detected between splits")

    scaler_mean = X[train_mask].mean(axis=0)
    scaler_std = X[train_mask].std(axis=0)
    scaler_std[scaler_std == 0] = 1.0
    Z = ((X - scaler_mean) / scaler_std).astype(np.float32)

    weights = np.ones(len(y), dtype=np.float32)
    if cfg["training"].get("class_weighting") == "subject_balanced":
        weights = _subject_weights(groups).astype(np.float32)

    hidden = [int(v) for v in cfg["model"]["hidden_dims"]]
    model = MLP(X.shape[1], hidden, len(class_order),
                float(cfg["model"].get("dropout", 0.0)),
                str(cfg["model"].get("activation", "relu"))).to(device)
    if str(cfg["training"]["optimizer"]) == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]))
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(
        [1.0] * len(class_order), dtype=torch.float32, device=device))
    y_t = torch.from_numpy(y)
    w_t = torch.from_numpy(weights)

    train_idx = np.where(train_mask)[0]
    batch_size = int(cfg["training"]["batch_size"])
    rng = np.random.default_rng(seed)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_loss, best_epoch = -1.0, float("inf"), 0
    patience = int(cfg["training"].get("early_stopping_patience", 15))
    stale = 0
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        order = rng.permutation(train_idx)
        total_loss, correct, seen = 0.0, 0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            xb = torch.from_numpy(Z[batch]).to(device)
            yb = y_t[batch].to(device)
            wb = w_t[batch].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            per_sample = nn.functional.cross_entropy(logits, yb, reduction="none")
            # Weights have mean 1 over the full training set; the batch mean is
            # therefore an unbiased estimate of the full weighted objective.
            # Re-normalising by the batch weight sum would change the target.
            loss = (per_sample * wb).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            correct += int((logits.argmax(dim=1) == yb).sum())
            seen += len(batch)
        val_probs = _predict(model, Z[val_mask], device, batch_size)
        val_frame = merged.loc[val_mask, ["group_id", "label", "split"]]
        val_m = _level_metric(val_frame, val_probs, class_order, positive_index, unit)
        val_loss = float(-np.mean(np.log(np.clip(
            val_probs[np.arange(val_probs.shape[0]), y[val_mask]], 1e-12, 1.0))))
        score = val_m["balanced_accuracy"] if val_m["balanced_accuracy"] is not None else -1.0
        history.append({
            "epoch": epoch, "train_loss": total_loss / max(seen, 1),
            "train_accuracy": correct / max(seen, 1), "val_loss": val_loss,
            f"val_{unit}_balanced_accuracy": score})
        improved = score > best_score + 1e-9 or (
            abs(score - best_score) <= 1e-9 and val_loss < best_loss - 1e-9)
        if improved:
            best_score, best_loss, best_epoch = score, val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)

    batch_size = int(cfg["training"]["batch_size"])
    probs = _predict(model, Z, device, batch_size)
    metrics = {}
    predictions = {"window": [], "recording": [], "subject": []}
    for split in ("train", "val", "test"):
        mask = splits == split
        frame = merged.loc[mask, ["sample_id", "group_id", "label", "split"]]
        part_probs = probs[mask]
        metrics[split] = _evaluate(frame, part_probs, class_order, positive_index)
        pred = part_probs.argmax(axis=1)
        durations = merged.loc[mask, "duration_decoded_sec"] \
            if "duration_decoded_sec" in merged.columns else pd.Series([None] * int(mask.sum()))
        for pos, (_, row) in enumerate(frame.iterrows()):
            duration = durations.iloc[pos]
            item = {
                "split": split,
                "window_id": row["sample_id"],  # one w0 window per recording in this baseline
                "sample_id": row["sample_id"],
                "subject_id": row["group_id"],
                "y_true": row["label"],
                "y_pred": class_order[pred[pos]],
                "valid_seconds": None if pd.isna(duration) else float(duration),
            }
            for i, name in enumerate(class_order):
                item[f"prob_{name}"] = float(part_probs[pos, i])
            item["correct"] = int(class_order[pred[pos]] == row["label"])
            predictions["window"].append(item)
            predictions["recording"].append(dict(item))
        agg = _aggregate_subjects(frame, part_probs, class_order)
        for _, row in agg.iterrows():
            probs_row = [row[f"prob_{name}"] for name in class_order]
            item = {"split": split, "subject_id": row["group_id"],
                    "y_true": row["label"], "n_recordings": int(row["n_recordings"])}
            for name, value in zip(class_order, probs_row):
                item[f"prob_{name}"] = float(value)
            item["y_pred"] = class_order[int(np.argmax(probs_row))]
            item["correct"] = int(item["y_pred"] == row["label"])
            predictions["subject"].append(item)

    run_id = _make_run_id(cfg, run_root)
    out_dir = (run_root or (workspace / "runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir()
    (out_dir / "inputs").mkdir()

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame(predictions["window"]).to_csv(out_dir / "predictions_windows.csv", index=False)
    pd.DataFrame(predictions["recording"]).to_csv(
        out_dir / "predictions_recordings.csv", index=False)
    pd.DataFrame(predictions["subject"]).to_csv(
        out_dir / "predictions_subjects.csv", index=False)

    label_counts = {s: dict(sorted(Counter(
        merged.loc[splits == s, "label"].astype(str)).items()))
        for s in ("train", "val", "test")}
    subject_counts = {s: len(set(groups[splits == s])) for s in ("train", "val", "test")}
    elapsed = (_dt.datetime.fromisoformat(_now())
               - _dt.datetime.fromisoformat(started)).total_seconds()
    job = {
        "run_id": run_id, "status": "completed",
        "experiment_id": cfg["experiment_id"],
        "experiment_name": cfg.get("experiment_name") or cfg["experiment_id"],
        "experiment_group": cfg.get("experiment_group"),
        "purpose": cfg.get("purpose"), "stage": cfg.get("stage"),
        "task_id": data["task_id"], "dataset": cfg["dataset"],
        "out_dir": str(out_dir), "elapsed_sec": round(elapsed, 2),
        "prepared_id": cfg["prepared_id"], "split_id": cfg["split_id"],
        "config_hash": config_hash(cfg), "seed": seed,
        "started_at": started, "finished_at": _now(),
        "code_version": __version__, "git": _git_commit(repo_root),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "data": {
            "manifest": str(data["manifest_file"]), "split": str(data["split_file"]),
            "split_summary": json.loads((data["split_dir"] / "split.json").read_text(
                encoding="utf-8"))["summary"],
            "n_samples": len(merged), "n_subjects": len(set(groups)),
            "label_counts": label_counts, "subject_counts": subject_counts,
            "sample_rate": data["sample_rate"], "feature_dim": int(X.shape[1]),
            "feature_cache": data["cache_info"],
        },
        "class_order": class_order, "positive_class": data["positive_class"],
        "model": cfg["model"], "training": cfg["training"],
        "evaluation_unit": unit, "primary_unit": unit, "grouping": grouping,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_score,
        "best_val_subject_balanced_accuracy": best_score,
        "metrics": metrics,
        "scaler": {"mean": scaler_mean.tolist(), "std": scaler_std.tolist()},
        "outputs": {
            "history_csv": str(out_dir / "history.csv"),
            "metrics_json": str(out_dir / "metrics.json"),
            "predictions_subjects": str(out_dir / "predictions_subjects.csv"),
            "predictions_recordings": str(out_dir / "predictions_recordings.csv"),
            "best_model": str(out_dir / "best.pt"),
            "report": str(out_dir / "report.md"),
        },
    }
    checkpoint = {"state_dict": model.state_dict(), "model": cfg["model"],
                  "input_dim": int(X.shape[1]), "class_order": class_order,
                  "scaler_mean": scaler_mean.tolist(), "scaler_std": scaler_std.tolist(),
                  "best_epoch": best_epoch}
    torch.save(checkpoint, out_dir / "best.pt")
    torch.save({**checkpoint, "epoch": int(history[-1]["epoch"]),
                "history": history}, out_dir / "last.pt")
    coverage = {
        split: {"n_expected": int((splits == split).sum()),
                "n_predicted": int(sum(1 for p in predictions["window"]
                                       if p["split"] == split))}
        for split in ("train", "val", "test")}
    warnings = []
    for split in ("train", "val", "test"):
        for level in ("window", "recording", "subject"):
            if job["metrics"][split][level]["auroc"] is None:
                warnings.append(f"{split}/{level}: AUROC undefined (single class)")
    metrics_doc = {
        "run_id": run_id,
        "task_id": data["task_id"],
        "dataset": cfg["dataset"],
        "class_order": class_order,
        "evaluation_unit": unit,
        "primary_unit": unit,
        "grouping": grouping,
        "positive_class": data["positive_class"],
        "primary_metric": "balanced_accuracy",
        "primary_split": "test",
        "probability_threshold": 0.5,
        "metrics": job["metrics"],
        "coverage": coverage,
        "warnings": warnings,
        "config_hash": job["config_hash"],
        "contract_version": CONTRACT_VERSION,
    }
    validate_metrics(metrics_doc)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    per_class = []
    for split in ("train", "val", "test"):
        part = merged.loc[splits == split, ["sample_id", "group_id", "label", "split"]]
        rows = per_class_rows(
            (part["label"].to_numpy() == data["positive_class"]).astype(int),
            probs[splits == split], class_order, positive_index)
        for row in rows:
            per_class.append({"split": split, "level": "window", **row})
        subj = _aggregate_subjects(part, probs[splits == split], class_order)
        rows = per_class_rows(
            (subj["label"].to_numpy() == data["positive_class"]).astype(int),
            subj[[f"prob_{n}" for n in class_order]].to_numpy(), class_order, positive_index)
        for row in rows:
            per_class.append({"split": split, "level": "subject", **row})
    pd.DataFrame(per_class).to_csv(out_dir / "metrics_per_class.csv", index=False)
    (out_dir / "feature_cache.json").write_text(
        json.dumps(data["cache_info"], ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    (out_dir / "run.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    shutil.copy2(data["manifest_file"], out_dir / "inputs" / "manifest.csv")
    shutil.copy2(data["split_file"], out_dir / "inputs" / "split.csv")
    task_file = repo_root / "configs" / "datasets" / f"{data['task_id']}.yaml"
    if task_file.is_file():
        shutil.copy2(task_file, out_dir / "inputs" / "task.yaml")
    write_figures(out_dir, job, probs, splits, merged, class_order)
    (out_dir / "report.md").write_text(render_run_report(job), encoding="utf-8")
    return job


def _make_run_id(cfg: dict, run_root: Path | None) -> str:
    base = (f"{_stamp()}_{cfg['experiment_id']}_s{cfg['training']['seed']}_"
            f"{config_hash(cfg)[:6]}")
    root = run_root or Path("runs")
    candidate, n = base, 0
    while (root / candidate).exists():
        n += 1
        candidate = f"{base}_{n:02d}"
    return candidate
