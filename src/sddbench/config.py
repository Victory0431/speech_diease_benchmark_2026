"""Path binding and workspace-safety checks.

Only this module resolves user paths. All writes are confined to
workspace_root; dataset roots are used read-only.
"""
from __future__ import annotations

from pathlib import Path

import yaml


class ConfigError(RuntimeError):
    """Raised for missing/invalid configuration or unsafe paths."""


def load_paths(paths_file: str | Path) -> dict:
    p = Path(paths_file).expanduser().resolve()
    if not p.is_file():
        raise ConfigError(f"paths file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("workspace_root", "dataset_roots"):
        if key not in cfg:
            raise ConfigError(f"paths file missing required key: {key}")
    return {
        "workspace_root": Path(cfg["workspace_root"]).expanduser().resolve(),
        "dataset_roots": {
            k: Path(v).expanduser().resolve() for k, v in cfg["dataset_roots"].items()
        },
        "paths_file": p,
    }


def dataset_root(paths: dict, dataset_id: str) -> Path:
    """Return the explicitly bound read-only root for a dataset."""
    roots = paths["dataset_roots"]
    if dataset_id not in roots:
        raise ConfigError(
            f"dataset root not bound in paths file: {dataset_id!r}; "
            f"bound datasets: {sorted(roots)}"
        )
    root = roots[dataset_id]
    if not root.is_dir():
        raise ConfigError(f"dataset root does not exist or is not a directory: {root}")
    return root


def workspace_dir(paths: dict, *parts: str, create: bool = True) -> Path:
    """Resolve a path under workspace_root and refuse any escape from it."""
    base = Path(paths["workspace_root"])
    target = base.joinpath(*parts).resolve()
    if target != base and base not in target.parents:
        raise ConfigError(f"path escapes workspace root: {target}")
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_under(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root`, refusing escapes (input is read-only)."""
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ConfigError(f"path escapes dataset root: {relative!r}")
    return target
