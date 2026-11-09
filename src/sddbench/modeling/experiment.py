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
    },
    "evaluation": {"positive_class": "disorder"},
    "labels": {"class_order": ["control", "disorder"], "positive_class": "disorder"},
}


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
    for key in ("dataset", "prepared_id", "split_id"):
        if not cfg.get(key):
            raise ExperimentConfigError(f"experiment config needs {key!r}")
    if cfg["model"].get("name") != "mlp":
        raise ExperimentConfigError("only model.name=mlp is implemented")
    if cfg["training"].get("optimizer") not in ("adamw", "adam"):
        raise ExperimentConfigError("optimizer must be adamw or adam")
    return cfg


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
