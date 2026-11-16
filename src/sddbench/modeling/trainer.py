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
from ..data.identity import manifest_fingerprint, split_fingerprint, task_fingerprint
from ..data.windows import normalize_preprocess, preprocess_fingerprint
from ..features.registry import normalize_feature_config, recipe_id
from ..results.contract import CONTRACT_VERSION, validate_metrics
from .experiment import config_hash, validate_experiment
from .features import FeatureLoadError, load_features
from .metrics import binary_metrics, decide_binary, per_class_rows
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


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "command": list(sys.argv),
        "torch_threads": torch.get_num_threads(),
    }


def _git_commit(repo_root: Path) -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True)
        dirty = bool(status.strip())
        stash = None
        if dirty:
            # `git stash create` snapshots tracked changes into a dangling commit
            # without touching the worktree, so a dirty run still has a
            # recoverable code state (apply with `git stash apply <sha>`).
            stash = subprocess.check_output(
                ["git", "stash", "create"], cwd=repo_root, text=True).strip() or None
        changed = [line[3:] for line in status.splitlines() if line.strip()]
        return {"commit": sha, "dirty": dirty, "stash_commit": stash,
                "dirty_files": changed[:50], "n_dirty_files": len(changed)}
    except Exception as e:
        return {"commit": None, "dirty": None, "error": f"{type(e).__name__}: {e}"}


REQUIRED_RUN_ARTIFACTS = (
    "config.yaml", "metrics.json", "history.csv", "metrics_per_class.csv",
    "predictions_windows.csv", "predictions_recordings.csv",
    "predictions_subjects.csv", "feature_cache.json", "report.md",
    "best.pt", "last.pt",
    "inputs/manifest.csv", "inputs/split.csv", "inputs/task.yaml",
    "inputs/preprocess.yaml", "inputs/features.yaml",
)


def _verify_run_artifacts(out_dir: Path, coverage: dict) -> list[str]:
    """Return human-readable problems; empty means the pack may be completed."""
    problems = []
    for name in REQUIRED_RUN_ARTIFACTS:
        path = out_dir / name
        if not path.is_file():
            problems.append(f"missing artifact: {name}")
        elif path.stat().st_size == 0:
            problems.append(f"empty artifact: {name}")
    for split, counts in coverage.items():
        if counts["n_predicted"] != counts["n_expected"]:
            problems.append(
                f"{split}: predicted {counts['n_predicted']} windows, "
                f"expected {counts['n_expected']}")
        if counts["n_predicted_subjects"] != counts["n_expected_subjects"]:
            problems.append(
                f"{split}: predicted {counts['n_predicted_subjects']} groups, "
                f"expected {counts['n_expected_subjects']}")
    return problems


def resolve_run_inputs(cfg: dict, workspace: Path, repo_root: Path) -> dict:
    """Validate + resolve every non-vector input the run depends on.

    This is the single identity gate shared by training (`load_run_data`) and
    the read-only plan used by the web console / `sddbench plan`: frozen task
    snapshot, manifest/split content fingerprints, labels, processed rate,
    feature/preprocess recipe and view fingerprint. It never reads vectors.
    """
    validate_experiment(cfg)
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
    if not admission_file.is_file():
        raise RunError(f"admission.json not found: {admission_file}; re-run `sddbench prepare`")
    admitted = json.loads(admission_file.read_text(encoding="utf-8"))

    # ---- frozen task snapshot (F04): never read the mutable repo task YAML ----
    task_snapshot_file = prepared_dir / "task.yaml"
    if not task_snapshot_file.is_file():
        raise RunError(
            f"prepared job {prepared_dir} has no task.yaml snapshot; its task definition "
            "was never frozen, so a changed grouping/filter/label map could silently pass. "
            "Re-run `sddbench prepare` to create a verifiable job")
    task_cfg = yaml.safe_load(task_snapshot_file.read_text(encoding="utf-8")) or {}
    task_hash = task_fingerprint(task_cfg)
    recorded_task_hash = str(admitted.get("task_hash") or "")
    if not recorded_task_hash:
        raise RunError(
            "admission.json is legacy (no task_hash); it cannot prove which task "
            "definition produced the manifest. Re-run `sddbench prepare`")
    if recorded_task_hash != task_hash:
        raise RunError(
            "task.yaml snapshot content no longer matches admission.json "
            f"(recorded {recorded_task_hash[:12]}, computed {task_hash[:12]}); "
            "re-run `sddbench prepare` instead of editing a frozen task snapshot")

    # ---- manifest content fingerprint, recomputed from the current CSV ----
    recorded_manifest_fp = str(admitted.get("manifest_fingerprint") or "")
    if not recorded_manifest_fp:
        raise RunError(
            "admission.json has no manifest fingerprint; re-run `sddbench prepare`")
    manifest_fp = manifest_fingerprint(manifest.to_dict("records"), task_hash=task_hash)
    if manifest_fp != recorded_manifest_fp:
        raise RunError(
            "manifest.csv content changed after admission "
            f"(recorded {recorded_manifest_fp[:12]}, computed {manifest_fp[:12]}); "
            "refusing to train on a modified manifest")
    if split_meta.get("manifest_fingerprint") != recorded_manifest_fp:
        raise RunError("split was built from a different manifest fingerprint; "
                       "rerun `sddbench split`")
    recorded_split_fp = str(split_meta.get("split_fingerprint") or "")
    if not recorded_split_fp:
        raise RunError(
            f"split {split_dir} is legacy (no split_fingerprint); re-run `sddbench split`")
    split_fp = split_fingerprint(split.to_dict("records"))
    if split_fp != recorded_split_fp:
        raise RunError(
            "split.csv content changed after it was created "
            f"(recorded {recorded_split_fp[:12]}, computed {split_fp[:12]})")
    if str(split_meta.get("task_hash") or "") != task_hash:
        raise RunError("split task_hash disagrees with the prepared task snapshot")

    if "benchmark_eligible" in admitted:
        benchmark_eligible = bool(admitted["benchmark_eligible"])
        scientific_status = dict(admitted.get("scientific_status") or {})
    else:
        # Historical prepared jobs predate the eligibility gate. Keep their
        # numbers reproducible, but never promote them to the formal set.
        benchmark_eligible = False
        scientific_status = {
            "level": "technical_only",
            "reason": "admission_missing" if not admitted else "legacy_admission_unverified",
        }
    for column in ("sample_id", "group_id", "label"):
        if column not in manifest.columns:
            raise RunError(f"manifest missing {column}")
        if column not in split.columns:
            raise RunError(f"split missing {column}")
    for column in ("sha256", "sample_rate"):
        if column not in manifest.columns:
            raise RunError(f"manifest missing {column}")
    merged = manifest.merge(split[["sample_id", "group_id", "label", "split"]],
                            on="sample_id", suffixes=("", "_split"),
                            validate="one_to_one")
    if len(merged) != len(manifest):
        raise RunError("manifest/split join lost rows")
    if (merged["group_id"] != merged["group_id_split"]).any():
        raise RunError("split group_id disagrees with manifest")
    if (merged["label"] != merged["label_split"]).any():
        raise RunError("split label disagrees with manifest")

    # Original Hz is a property of the source; the cache is keyed by the rate the
    # frontend actually produced (F07). Resolve that rate before loading vectors.
    original_rates = merged["sample_rate"].dropna().astype(int).unique().tolist()

    labels_cfg = task_cfg.get("labels") or {}
    grouping = str(task_cfg.get("grouping") or "subject")
    if grouping not in ("subject", "file"):
        raise RunError(f"task {task_id} grouping must be subject|file, got {grouping!r}")
    class_order = list((cfg.get("labels") or {}).get("class_order")
                       or labels_cfg.get("class_order")
                       or ["control", "disorder"])
    observed = sorted(merged["label"].astype(str).unique().tolist())
    if len(observed) != 2:
        raise RunError(
            f"the MLP baseline is binary only; manifest has {len(observed)} "
            f"classes: {observed}")
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
    if preprocess_cfg["sample_rate_policy"] == "resample":
        processed_rate = int(preprocess_cfg["target_sample_rate"])
    elif len(original_rates) == 1:
        processed_rate = int(original_rates[0])
    else:
        raise RunError(
            f"sample_rate_policy=native with mixed original rates {original_rates}; "
            "the MLP baseline needs one processed rate (use resample(target_sample_rate=...) "
            "or a single-rate dataset)")
    sample_ids = merged["sample_id"].astype(str).tolist()
    source_hashes = {str(s): str(h) for s, h in
                     zip(merged["sample_id"], merged["sha256"])}
    return {
        "task_id": task_id, "manifest": manifest, "split": split,
        "merged": merged,
        "class_order": class_order, "positive_class": positive,
        "positive_index": class_order.index(positive),
        "sample_rate": processed_rate,
        "original_sample_rates": original_rates,
        "feature_cfg": feature_cfg,
        "features_rel": cfg["features"]["config"],
        "preprocess_rel": preprocess_rel, "view_fingerprint": view_fingerprint,
        "grouping": grouping,
        "prepared_dir": prepared_dir,
        "manifest_file": manifest_file, "split_file": split_file,
        "split_dir": split_dir,
        "task_snapshot_file": task_snapshot_file,
        "admission_file": admission_file,
        "task_cfg": task_cfg,
        "preprocess_cfg": preprocess_cfg,
        "task_hash": task_hash,
        "manifest_fingerprint": manifest_fp,
        "split_fingerprint": split_fp,
        "sample_ids": sample_ids,
        "source_hashes": source_hashes,
        "benchmark_eligible": benchmark_eligible,
        "scientific_status": scientific_status,
        "admitted": admitted,
    }


def load_run_data(cfg: dict, workspace: Path, repo_root: Path) -> dict:
    data = resolve_run_inputs(cfg, workspace, repo_root)
    try:
        X, cache_info = load_features(
            workspace, data["feature_cfg"], data["sample_ids"], data["sample_rate"],
            dataset_id=str(cfg["dataset"]),
            source_hashes=data["source_hashes"],
            view_fingerprint=data["view_fingerprint"])
    except FeatureLoadError as e:
        raise RunError(str(e)) from e
    return {**data, "X": X, "cache_info": cache_info}


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


def _sample_weights(groups: np.ndarray, train_mask: np.ndarray,
                    strategy: str) -> np.ndarray:
    """Weights for the training objective, fitted on train rows only (F12)."""
    weights = np.ones(len(groups), dtype=np.float64)
    if strategy == "subject_balanced":
        weights[train_mask] = _subject_weights(groups[train_mask])
    return weights.astype(np.float32)


def _subject_metric(frame: pd.DataFrame, probs: np.ndarray,
                    class_order: list[str], positive_index: int,
                    threshold: float = 0.5) -> dict:
    agg = _aggregate_subjects(frame, probs, class_order)
    y = (agg["label"].to_numpy() == class_order[positive_index]).astype(int)
    p = agg[[f"prob_{name}" for name in class_order]].to_numpy()
    return binary_metrics(y, p, positive_index, threshold)


def _level_metric(frame: pd.DataFrame, probs: np.ndarray, class_order: list[str],
                  positive_index: int, unit: str, threshold: float = 0.5) -> dict:
    if unit == "recording":
        y = (frame["label"].to_numpy() == class_order[positive_index]).astype(int)
        return binary_metrics(y, probs, positive_index, threshold)
    return _subject_metric(frame, probs, class_order, positive_index, threshold)


def _evaluate(frame: pd.DataFrame, probs: np.ndarray, class_order: list[str],
              positive_index: int, threshold: float = 0.5) -> dict:
    y = (frame["label"].to_numpy() == class_order[positive_index]).astype(int)
    return {
        "window": binary_metrics(y, probs, positive_index, threshold),
        "recording": binary_metrics(y, probs, positive_index, threshold),
        "subject": _subject_metric(frame, probs, class_order, positive_index, threshold),
    }


def run_experiment(cfg: dict, workspace: Path, repo_root: Path,
                   run_root: Path | None = None) -> dict:
    """Create the run record first, then train into it.

    The run directory and a `run.json` with status=running exist before any data
    is loaded, so a crash or interrupt still leaves the configuration, code
    revision and failure reason on disk. Status only becomes `completed` after
    every required artifact exists and the prediction coverage check passes.
    """
    validate_experiment(cfg)
    started = _now()
    run_id = _make_run_id(cfg, run_root)
    out_dir = (run_root or (workspace / "runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir()
    (out_dir / "inputs").mkdir()
    meta = {
        "run_id": run_id, "status": "running", "phase": "created",
        "attempt": 1,
        "experiment_id": cfg["experiment_id"],
        "experiment_name": cfg.get("experiment_name") or cfg["experiment_id"],
        "experiment_group": cfg.get("experiment_group"),
        "purpose": cfg.get("purpose"), "stage": cfg.get("stage"),
        "dataset": cfg["dataset"], "task_id": cfg.get("task_id"),
        "prepared_id": cfg["prepared_id"], "split_id": cfg["split_id"],
        "seed": int(cfg["training"]["seed"]),
        "seed_role": "training",
        "started_at": started,
        "config_hash": config_hash(cfg),
        "code_version": __version__,
        "git": _git_commit(repo_root),
        "environment": _environment(),
        "config": copy.deepcopy(cfg),
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "run.json", meta)
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    try:
        return _train_into(cfg, workspace, repo_root, out_dir, started, run_id, meta)
    except BaseException as e:  # keep configuration/status even on Ctrl-C
        interrupted = isinstance(e, (KeyboardInterrupt, SystemExit))
        meta.update({
            "status": "interrupted" if interrupted else "failed",
            "phase": "interrupted" if interrupted else "failed",
            "finished_at": _now(),
            "error": f"{type(e).__name__}: {e}",
        })
        _write_json(out_dir / "run.json", meta)
        raise


def _train_into(cfg: dict, workspace: Path, repo_root: Path, out_dir: Path,
                started: str, run_id: str, meta: dict) -> dict:
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
    selection_metric = str(cfg["training"].get("selection_metric")
                           or default_selection)
    if selection_metric != default_selection:
        raise RunError(f"selection_metric must be {default_selection!r} for "
                       f"evaluation.unit={unit!r}, got {selection_metric!r}")

    merged = data["merged"]
    X = data["X"]
    class_order = data["class_order"]
    positive_index = data["positive_index"]
    if len(class_order) != 2:
        raise RunError(f"the MLP baseline is binary only; class_order has "
                       f"{len(class_order)} classes: {class_order}")
    threshold = float((cfg.get("evaluation") or {}).get("binary_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise RunError(f"evaluation.binary_threshold must be in [0, 1], got {threshold!r}")
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

    weights = _sample_weights(
        groups, train_mask, str(cfg["training"].get("class_weighting")))

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
        val_m = _level_metric(val_frame, val_probs, class_order, positive_index,
                              unit, threshold)
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
    last_state = copy.deepcopy(model.state_dict())
    last_epoch = int(history[-1]["epoch"]) if history else 0
    model.load_state_dict(best_state)

    batch_size = int(cfg["training"]["batch_size"])
    probs = _predict(model, Z, device, batch_size)
    metrics = {}
    predictions = {"window": [], "recording": [], "subject": []}
    window_ids = list(data["cache_info"]["window_ids"])
    valid_frames = list(data["cache_info"]["valid_frames"])
    processed_rate = int(data["sample_rate"])
    for split in ("train", "val", "test"):
        mask = splits == split
        frame = merged.loc[mask, ["sample_id", "group_id", "label", "split"]]
        part_probs = probs[mask]
        metrics[split] = _evaluate(frame, part_probs, class_order, positive_index, threshold)
        pred = decide_binary(part_probs[:, positive_index], threshold)
        positions = np.where(mask)[0]
        for pos, global_pos in enumerate(positions):
            row = merged.iloc[global_pos]
            # Window facts come from the feature cache, not from sample_id or the
            # original file duration: valid_seconds excludes zero padding.
            item = {
                "split": split,
                "window_id": window_ids[global_pos],
                "sample_id": row["sample_id"],
                "subject_id": row["group_id"],
                "y_true": row["label"],
                "valid_seconds": round(int(valid_frames[global_pos]) / processed_rate, 6),
            }
            for i, name in enumerate(class_order):
                item[f"prob_{name}"] = float(part_probs[pos, i])
            predicted_index = positive_index if int(pred[pos]) == 1 else 1 - positive_index
            item["y_pred"] = class_order[int(predicted_index)]
            item["correct"] = int(item["y_pred"] == row["label"])
            predictions["window"].append(item)
            predictions["recording"].append(dict(item))
        agg = _aggregate_subjects(frame, part_probs, class_order)
        for _, row in agg.iterrows():
            probs_row = [row[f"prob_{name}"] for name in class_order]
            item = {"split": split, "subject_id": row["group_id"],
                    "y_true": row["label"], "n_recordings": int(row["n_recordings"])}
            for name, value in zip(class_order, probs_row):
                item[f"prob_{name}"] = float(value)
            p_pos = float(row[f"prob_{class_order[positive_index]}"])
            predicted_index = positive_index if float(p_pos) >= threshold \
                else 1 - positive_index
            item["y_pred"] = class_order[int(predicted_index)]
            item["correct"] = int(item["y_pred"] == row["label"])
            predictions["subject"].append(item)

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
    identity = {
        "task_snapshot": str(data["task_snapshot_file"]),
        "task_hash": data["task_hash"],
        "manifest_fingerprint": data["manifest_fingerprint"],
        "split_fingerprint": data["split_fingerprint"],
        "prepared_id": cfg["prepared_id"], "split_id": cfg["split_id"],
    }
    scientific_context = {
        **identity,
        "view_fingerprint": data["view_fingerprint"],
        "feature_hash": data["cache_info"]["feature_hash"],
        "feature_dim": int(X.shape[1]),
        "processed_sample_rate": data["sample_rate"],
        "class_order": class_order,
        "positive_class": data["positive_class"],
        "binary_threshold": threshold,
        "evaluation_unit": unit,
        "selection_metric": selection_metric,
    }
    scientific_hash = config_hash(cfg, context=scientific_context)
    attempt = _attempt_number(out_dir.parent, run_id, seed, scientific_hash)
    meta.update({
        "config_hash": scientific_hash, "attempt": attempt,
        "phase": "data_loaded", "identity": identity,
        "task_hash": data["task_hash"],
        "benchmark_eligible": bool(data["benchmark_eligible"]),
        "scientific_status": data["scientific_status"],
        "task": {"config": data["task_cfg"], "grouping": grouping,
                 "labels": (data["task_cfg"].get("labels") or {})},
        "preprocess": {"path": data["preprocess_rel"],
                       "fingerprint": data["view_fingerprint"],
                       "config": data["preprocess_cfg"]},
        "feature": {"path": cfg["features"]["config"],
                    "config": data["feature_cfg"],
                    "recipe_id": recipe_id(data["feature_cfg"]),
                    "feature_key": data["cache_info"].get("feature_key"),
                    "feature_hash": data["cache_info"].get("feature_hash"),
                    "sample_rate": data["sample_rate"],
                    "dim": int(X.shape[1])},
        "labels": {"class_order": class_order,
                   "positive_class": data["positive_class"],
                   "source": "experiment_config" if (cfg.get("labels") or {})
                   else "task_snapshot"},
        "model": cfg["model"], "training": cfg["training"],
        "evaluation": cfg["evaluation"],
        "evaluation_unit": unit, "grouping": grouping,
        "selection_metric": selection_metric,
    })
    coverage = {}
    for split in ("train", "val", "test"):
        expected_subjects = len(set(groups[splits == split]))
        predicted_subjects = len({p["subject_id"] for p in predictions["subject"]
                                  if p["split"] == split})
        coverage[split] = {
            "n_expected": int((splits == split).sum()),
            "n_predicted": int(sum(1 for p in predictions["window"]
                                   if p["split"] == split)),
            "n_expected_subjects": expected_subjects,
            "n_predicted_subjects": predicted_subjects,
        }
    job = {
        "run_id": run_id, "attempt": attempt, "phase": "training_done",
        "status": "running",
        "experiment_id": cfg["experiment_id"],
        "experiment_name": cfg.get("experiment_name") or cfg["experiment_id"],
        "experiment_group": cfg.get("experiment_group"),
        "purpose": cfg.get("purpose"), "stage": cfg.get("stage"),
        "task_id": data["task_id"], "dataset": cfg["dataset"],
        "out_dir": str(out_dir), "elapsed_sec": round(elapsed, 2),
        "prepared_id": cfg["prepared_id"], "split_id": cfg["split_id"],
        "config_hash": scientific_hash, "seed": seed, "seed_role": "training",
        "started_at": started, "finished_at": _now(),
        "code_version": __version__, "git": meta["git"],
        "environment": meta["environment"],
        "identity": identity,
        "task": meta["task"], "preprocess": meta["preprocess"],
        "feature": meta["feature"], "labels": meta["labels"],
        "data": {
            "manifest": str(data["manifest_file"]), "split": str(data["split_file"]),
            "split_summary": json.loads((data["split_dir"] / "split.json").read_text(
                encoding="utf-8"))["summary"],
            "n_samples": len(merged), "n_subjects": len(set(groups)),
            "label_counts": label_counts, "subject_counts": subject_counts,
            "sample_rate": data["sample_rate"],
            "original_sample_rates": data["original_sample_rates"],
            "feature_dim": int(X.shape[1]),
            "feature_cache": data["cache_info"],
            "identity": identity,
        },
        "class_order": class_order, "positive_class": data["positive_class"],
        "benchmark_eligible": bool(data["benchmark_eligible"]),
        "scientific_status": data["scientific_status"],
        "features": {
            "feature_type": data["feature_cfg"]["feature_type"],
            "recipe_id": recipe_id(data["feature_cfg"]),
            "feature_config": cfg["features"]["config"],
            "preprocess_config": data["preprocess_rel"],
            "preprocess_fingerprint": data["view_fingerprint"],
            "feature_key": data["cache_info"].get("feature_key"),
            "feature_hash": data["cache_info"].get("feature_hash"),
            "sample_rate": data["sample_rate"],
            "dim": int(X.shape[1]),
        },
        "model": cfg["model"], "training": cfg["training"],
        "evaluation": cfg["evaluation"],
        "evaluation_unit": unit, "primary_unit": unit, "grouping": grouping,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_score,
        "best_val_subject_balanced_accuracy": best_score,
        "trained_epochs": last_epoch,
        "metrics": metrics,
        "scaler": {"mean": scaler_mean.tolist(), "std": scaler_std.tolist()},
        "outputs": {
            "history_csv": str(out_dir / "history.csv"),
            "metrics_json": str(out_dir / "metrics.json"),
            "predictions_subjects": str(out_dir / "predictions_subjects.csv"),
            "predictions_recordings": str(out_dir / "predictions_recordings.csv"),
            "best_model": str(out_dir / "best.pt"),
            "last_model": str(out_dir / "last.pt"),
            "feature_index": str(out_dir / "inputs" / "feature_index.csv"),
            "report": str(out_dir / "report.md"),
        },
        "coverage": coverage,
        "checkpoints": {
            "best": {"file": str(out_dir / "best.pt"), "role": "best_val_selection",
                     "best_epoch": best_epoch, "val_score": best_score},
            "last": {"file": str(out_dir / "last.pt"), "role": "last_epoch",
                     "epoch": last_epoch, "resume_supported": False},
        },
    }
    selection = {
        "selection_metric": selection_metric, "selection_direction": "max",
        "val_score": best_score, "best_epoch": best_epoch,
        "trained_epochs": last_epoch,
    }
    checkpoint = {"state_dict": model.state_dict(), "model": cfg["model"],
                  "input_dim": int(X.shape[1]), "class_order": class_order,
                  "scaler_mean": scaler_mean.tolist(), "scaler_std": scaler_std.tolist(),
                  "best_epoch": best_epoch, "role": "best_val_selection",
                  **selection}
    # Semantics are explicit: best.pt holds the early-stopping-selected weights,
    # last.pt holds the weights after the final epoch actually trained. Resume is
    # not implemented, so it is declared unsupported instead of faking it.
    torch.save(checkpoint, out_dir / "best.pt")
    torch.save({
        **checkpoint, "state_dict": last_state,
        "role": "last_epoch", "epoch": last_epoch,
        "best_epoch": best_epoch, "history": history,
        "resume_supported": False,
        "checkpoint_note": "last epoch weights; not a resumable optimizer state",
    }, out_dir / "last.pt")
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
        "label_definition": {"class_order": class_order,
                             "positive_class": data["positive_class"]},
        "evaluation_unit": unit,
        "primary_unit": unit,
        "grouping": grouping,
        "positive_class": data["positive_class"],
        "primary_metric": "balanced_accuracy",
        "primary_split": "test",
        "probability_threshold": threshold,
        "metrics": job["metrics"],
        "coverage": coverage,
        "warnings": warnings,
        "config_hash": scientific_hash,
        "identity": identity,
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
            probs[splits == split], class_order, positive_index, threshold)
        for row in rows:
            per_class.append({"split": split, "level": "window", **row})
        subj = _aggregate_subjects(part, probs[splits == split], class_order)
        rows = per_class_rows(
            (subj["label"].to_numpy() == data["positive_class"]).astype(int),
            subj[[f"prob_{n}" for n in class_order]].to_numpy(), class_order,
            positive_index, threshold)
        for row in rows:
            per_class.append({"split": split, "level": "subject", **row})
    pd.DataFrame(per_class).to_csv(out_dir / "metrics_per_class.csv", index=False)
    _write_json(out_dir / "feature_cache.json", data["cache_info"])
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    # Inputs: the frozen task snapshot is the scientific source of truth; the
    # mutable repo copy is only kept as provenance.
    shutil.copy2(data["manifest_file"], out_dir / "inputs" / "manifest.csv")
    shutil.copy2(data["split_file"], out_dir / "inputs" / "split.csv")
    shutil.copy2(data["task_snapshot_file"], out_dir / "inputs" / "task.yaml")
    shutil.copy2(repo_root / data["preprocess_rel"], out_dir / "inputs" / "preprocess.yaml")
    shutil.copy2(repo_root / cfg["features"]["config"],
                 out_dir / "inputs" / "features.yaml")
    shutil.copy2(Path(data["cache_info"]["cache_dir"]) / "index.csv",
                 out_dir / "inputs" / "feature_index.csv")
    write_figures(out_dir, job, probs, splits, merged, class_order)
    (out_dir / "report.md").write_text(render_run_report(job), encoding="utf-8")

    problems = _verify_run_artifacts(out_dir, coverage)
    if problems:
        meta.update({"phase": "incomplete", "coverage": coverage,
                     "completion_errors": problems})
        raise RunError("run pack failed the completion check: " + "; ".join(problems))
    job.update({
        "status": "completed", "phase": "completed", "finished_at": _now(),
        "completion": {"verified": True, "coverage_ok": True,
                       "required_artifacts": len(REQUIRED_RUN_ARTIFACTS)},
    })
    _write_json(out_dir / "run.json", job)
    meta.update(job)
    return job


def _attempt_number(run_parent: Path, run_id: str, seed: int, config_hash_value: str) -> int:
    """Count earlier runs of the same science+seed as attempts, not new seeds."""
    attempt = 1
    if not run_parent.is_dir():
        return attempt
    for directory in run_parent.iterdir():
        if not directory.is_dir() or directory.name == run_id:
            continue
        meta_file = directory / "run.json"
        if not meta_file.is_file():
            continue
        try:
            other = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(other.get("seed", -1)) != int(seed):
            continue
        if other.get("config_hash") != config_hash_value:
            continue
        if other.get("status") not in ("running", "completed"):
            continue
        attempt += 1
    return attempt


def _make_run_id(cfg: dict, run_root: Path | None) -> str:
    base = (f"{_stamp()}_{cfg['experiment_id']}_s{cfg['training']['seed']}_"
            f"{config_hash(cfg)[:6]}")
    root = run_root or Path("runs")
    candidate, n = base, 0
    while (root / candidate).exists():
        n += 1
        candidate = f"{base}_{n:02d}"
    return candidate
