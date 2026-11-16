"""Experiment config loading, dot-key overrides and small grid expansion."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


class ExperimentConfigError(RuntimeError):
    pass


DEFAULTS = {
    "experiment_id": "experiment",
    "task_id": None,
    "dataset": None,
    "prepared_id": None,
    "split_id": None,
    "features": {"config": "configs/features/mfcc20_stats.yaml"},
    "model": {
        "name": "mlp", "hidden_dims": [128], "dropout": 0.2,
        "activation": "relu",
    },
    "training": {
        "seed": 0, "epochs": 100, "batch_size": 128,
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "optimizer": "adamw", "early_stopping_patience": 15,
        "class_weighting": "subject_balanced", "device": "cpu",
        "selection_metric": None,  # resolved from evaluation.unit when omitted
        "selection_direction": "max",
    },
    "evaluation": {"unit": None, "primary_metric": "balanced_accuracy",
                   "binary_threshold": 0.5},
    "labels": {"class_order": ["control", "disorder"], "positive_class": "disorder"},
}

# Keys that are accepted.  Anything else is rejected early instead of being
# merged, saved and silently ignored (audit F06).
TOP_KEYS = {
    "experiment_id", "experiment_name", "experiment_group", "purpose", "stage",
    "notes", "description", "task_id", "dataset", "prepared_id", "split_id",
    "features", "model", "training", "evaluation", "labels",
}
SECTION_KEYS = {
    "features": {"config", "preprocess"},
    "model": {"name", "hidden_dims", "dropout", "activation"},
    "training": {
        "seed", "epochs", "batch_size", "learning_rate", "weight_decay",
        "optimizer", "early_stopping_patience", "class_weighting", "device",
        "selection_metric", "selection_direction",
    },
    "evaluation": {"unit", "primary_metric", "binary_threshold"},
    "labels": {"class_order", "positive_class"},
}
SELECTION_METRICS = {
    "window_balanced_accuracy", "recording_balanced_accuracy",
    "subject_balanced_accuracy",
}


def _require_int(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExperimentConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_number(value, name: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentConfigError(f"{name} must be a number, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ExperimentConfigError(f"{name} must be {bound}, got {value!r}")
    return float(value)


def validate_experiment(cfg: dict) -> dict:
    """Strict, shared validation for CLI, YAML, sweep and web.

    Every accepted key must have a real effect.  Features that are not
    implemented yet (multi-class, `selection_direction=min`, other models)
    are rejected before any data is loaded.
    """
    if not isinstance(cfg, dict):
        raise ExperimentConfigError("experiment config must be a mapping")
    unknown = sorted(set(cfg) - TOP_KEYS)
    if unknown:
        raise ExperimentConfigError(f"unknown experiment key(s): {unknown}")
    for section, allowed in SECTION_KEYS.items():
        node = cfg.get(section)
        if not isinstance(node, dict):
            raise ExperimentConfigError(f"experiment.{section} must be a mapping")
        extra = sorted(set(node) - allowed)
        if extra:
            raise ExperimentConfigError(
                f"unknown key(s) in experiment.{section}: {extra}")

    for key in ("dataset", "prepared_id", "split_id"):
        if not cfg.get(key):
            raise ExperimentConfigError(f"experiment config needs {key!r}")

    stage = cfg.get("stage")
    if stage is not None and stage not in ("exploration", "baseline", "confirmatory"):
        raise ExperimentConfigError(
            f"stage must be exploration/baseline/confirmatory, got {stage!r}")

    features = cfg["features"]
    for key in ("config", "preprocess"):
        if features.get(key) is not None and not isinstance(features[key], str):
            raise ExperimentConfigError(f"features.{key} must be a path string")

    model = cfg["model"]
    if model.get("name") != "mlp":
        raise ExperimentConfigError("only model.name=mlp is implemented")
    hidden = model.get("hidden_dims")
    if not isinstance(hidden, list) or not hidden:
        raise ExperimentConfigError("model.hidden_dims must be a non-empty list")
    for i, width in enumerate(hidden):
        _require_int(width, f"model.hidden_dims[{i}]", 1)
    _require_number(model.get("dropout"), "model.dropout", 0.0)
    if float(model["dropout"]) >= 1.0:
        raise ExperimentConfigError("model.dropout must be < 1.0")
    if model.get("activation") != "relu":
        raise ExperimentConfigError("only model.activation=relu is implemented")

    training = cfg["training"]
    _require_int(training.get("seed"), "training.seed", 0)
    _require_int(training.get("epochs"), "training.epochs", 1)
    _require_int(training.get("batch_size"), "training.batch_size", 1)
    _require_number(training.get("learning_rate"), "training.learning_rate", 1e-12)
    _require_number(training.get("weight_decay"), "training.weight_decay", 0.0)
    _require_int(training.get("early_stopping_patience"),
                 "training.early_stopping_patience", 1)
    if training.get("optimizer") not in ("adamw", "adam"):
        raise ExperimentConfigError("optimizer must be adamw or adam")
    if training.get("class_weighting") not in ("subject_balanced", "none"):
        raise ExperimentConfigError(
            "training.class_weighting must be subject_balanced or none")
    if training.get("device") not in ("cpu", "cuda"):
        raise ExperimentConfigError("training.device must be cpu or cuda")
    if training.get("selection_direction") != "max":
        raise ExperimentConfigError(
            "training.selection_direction=min is not implemented; use max")
    selection_metric = training.get("selection_metric")
    if selection_metric is not None and selection_metric not in SELECTION_METRICS:
        raise ExperimentConfigError(
            f"training.selection_metric must be one of {sorted(SELECTION_METRICS)}")

    evaluation = cfg["evaluation"]
    if evaluation.get("unit") not in (None, "subject", "recording"):
        raise ExperimentConfigError("evaluation.unit must be subject or recording")
    if evaluation.get("primary_metric") != "balanced_accuracy":
        raise ExperimentConfigError(
            "evaluation.primary_metric=balanced_accuracy is the only implemented metric")
    _require_number(evaluation.get("binary_threshold"),
                    "evaluation.binary_threshold", 0.0, 1.0)

    labels = cfg["labels"]
    order = labels.get("class_order")
    if not isinstance(order, list) or len(order) != 2 or len(set(order)) != 2 \
            or any(not isinstance(x, str) for x in order):
        raise ExperimentConfigError(
            "labels.class_order must contain exactly two distinct class names "
            "(multi-class is not implemented)")
    if labels.get("positive_class") not in order:
        raise ExperimentConfigError("labels.positive_class must be in labels.class_order")

    return cfg


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment(path: str | Path) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ExperimentConfigError(f"experiment config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, cfg)
    return validate_experiment(cfg)


def set_override(cfg: dict, assignment: str) -> dict:
    if "=" not in assignment:
        raise ExperimentConfigError(f"override must be key=value, got {assignment!r}")
    dotted, raw = assignment.split("=", 1)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ExperimentConfigError(f"bad override value {raw!r}: {e}") from e
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ExperimentConfigError(f"unknown config path: {dotted}")
        node = node[part]
    if parts[-1] not in node:
        raise ExperimentConfigError(f"unknown config key: {dotted}")
    node[parts[-1]] = value
    return cfg


def expand_grid(base_cfg: dict, grid: dict[str, list]) -> list[dict]:
    """Expand dot-key grid values into a list of independent configs."""
    configs = [copy.deepcopy(base_cfg)]
    for dotted, values in grid.items():
        next_configs = []
        for cfg in configs:
            for value in values:
                new = copy.deepcopy(cfg)
                set_override(new, f"{dotted}={json.dumps(value)}")
                next_configs.append(new)
        configs = next_configs
    return configs


def config_hash(cfg: dict) -> str:
    import hashlib
    payload = {
        "task_id": cfg.get("task_id"), "dataset": cfg.get("dataset"),
        "prepared_id": cfg.get("prepared_id"), "split_id": cfg.get("split_id"),
        "features": cfg.get("features"), "model": cfg.get("model"),
        "training": cfg.get("training"), "evaluation": cfg.get("evaluation"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
