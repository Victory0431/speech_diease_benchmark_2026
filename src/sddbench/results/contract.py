"""Stable results contract shared by every dataset and model.

The contract is deliberately small: it says which numbers every run must
report, at which splits/units, so a leaderboard can compare datasets without
touching run internals.
"""
from __future__ import annotations

import datetime as _dt

CONTRACT_VERSION = "1.1"

SPLITS = ("train", "val", "test")
LEVELS = ("window", "recording", "subject")
UNITS = ("subject", "recording")

REQUIRED_METRICS = (
    "n", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "precision", "f1", "auroc", "average_precision", "log_loss", "brier",
    "confusion",
)
REQUIRED_CONFUSION = ("tn", "fp", "fn", "tp")


class ResultsContractError(RuntimeError):
    """Raised when a run result does not satisfy the benchmark contract."""


def validate_metrics(doc: dict) -> None:
    """Fail loudly if a metrics document cannot be compared across datasets."""
    required = ("run_id", "task_id", "dataset", "class_order", "evaluation_unit",
                "positive_class", "primary_metric", "metrics")
    missing = [key for key in required if key not in doc]
    if missing:
        raise ResultsContractError(f"metrics.json missing top-level keys: {missing}")
    order = list(doc["class_order"])
    if len(order) != 2 or len(set(order)) != 2:
        raise ResultsContractError("class_order must contain exactly two distinct classes")
    unit = doc["evaluation_unit"]
    if unit not in UNITS:
        raise ResultsContractError(
            f"evaluation_unit must be one of {UNITS}, got {unit!r}")
    primary_unit = doc.get("primary_unit") or unit
    if primary_unit not in UNITS:
        raise ResultsContractError(
            f"primary_unit must be one of {UNITS}, got {primary_unit!r}")
    if primary_unit != unit:
        raise ResultsContractError(
            f"primary_unit {primary_unit!r} must match evaluation_unit {unit!r}")
    for split in SPLITS:
        if split not in doc["metrics"]:
            raise ResultsContractError(f"metrics.json missing split {split!r}")
        for level in LEVELS:
            if level not in doc["metrics"][split]:
                raise ResultsContractError(f"metrics.json missing {split}/{level}")
            block = doc["metrics"][split][level]
            if block is None:
                raise ResultsContractError(f"metrics {split}/{level} is null")
            absent = [key for key in REQUIRED_METRICS if key not in block]
            if absent:
                raise ResultsContractError(
                    f"metrics {split}/{level} missing {absent}")
            confusion_missing = [k for k in REQUIRED_CONFUSION
                                 if k not in block["confusion"]]
            if confusion_missing:
                raise ResultsContractError(
                    f"metrics {split}/{level}/confusion missing {confusion_missing}")


def attach_contract(doc: dict) -> dict:
    out = dict(doc)
    out["contract_version"] = CONTRACT_VERSION
    out["primary_unit"] = out.get("primary_unit") or out.get("evaluation_unit")
    return out


def sanitize_run_meta(run: dict) -> dict:
    """Keep reproducibility facts, drop absolute paths and patient-level rows."""
    data = run.get("data") or {}
    cache = data.get("feature_cache") or {}
    return {
        "contract_version": CONTRACT_VERSION,
        "published_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "experiment_id": run.get("experiment_id"),
        "experiment_name": run.get("experiment_name"),
        "experiment_group": run.get("experiment_group"),
        "purpose": run.get("purpose"),
        "stage": run.get("stage"),
        "task_id": run.get("task_id"),
        "dataset": run.get("dataset"),
        "prepared_id": run.get("prepared_id"),
        "split_id": run.get("split_id"),
        "class_order": run.get("class_order"),
        "positive_class": run.get("positive_class"),
        "seed": run.get("seed"),
        "model": run.get("model"),
        "training": _sanitize_training(run.get("training")),
        "best_epoch": run.get("best_epoch"),
        "best_val_balanced_accuracy":
            run.get("best_val_balanced_accuracy",
                    run.get("best_val_subject_balanced_accuracy")),
        "best_val_subject_balanced_accuracy":
            run.get("best_val_subject_balanced_accuracy"),
        "evaluation_unit": run.get("evaluation_unit"),
        "primary_unit": run.get("primary_unit") or run.get("evaluation_unit"),
        "grouping": run.get("grouping"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "elapsed_sec": run.get("elapsed_sec"),
        "code_version": run.get("code_version"),
        "git": run.get("git"),
        "config_hash": run.get("config_hash"),
        "environment": run.get("environment"),
        "data": {
            "n_samples": data.get("n_samples"),
            "n_subjects": data.get("n_subjects"),
            "feature_dim": data.get("feature_dim"),
            "sample_rate": data.get("sample_rate"),
            "subject_counts": data.get("subject_counts"),
            "label_counts": data.get("label_counts"),
            "split_summary": data.get("split_summary"),
            "feature_cache": {
                key: cache.get(key) for key in
                ("feature_key", "feature_hash", "view_fingerprint",
                 "sample_rate", "dim", "n_samples")
            },
        },
    }


def _sanitize_training(training) -> dict:
    if not isinstance(training, dict):
        return {}
    keep = ("seed", "epochs", "batch_size", "optimizer", "learning_rate",
            "weight_decay", "early_stopping_patience", "selection_metric",
            "class_weighting", "device")
    return {key: training.get(key) for key in keep}

