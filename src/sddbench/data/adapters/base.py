"""Minimal adapter interface (macro doc §5.2, data-layer doc §2-§3).

An adapter ONLY explains its library: who owns which file, where candidate
labels live, which files are processed copies. It does NOT decode audio,
choose sample rates, crop, or compute features. `collect` receives the file
inventory produced by the generic scanner (data-layer doc step ②) and returns
enrichment fields + issues; the generic auditor does decoding/measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AdapterResult:
    """`enrichment`: relative_path -> dict of adapter columns to merge.
    `excluded`: rows the adapter proposes to keep out of later task scopes
    (path + reason; D1 keeps everything, decisions belong to prepare).
    `issues`: review items with severity, never silently dropped.
    """
    enrichment: dict[str, dict[str, Any]] = field(default_factory=dict)
    excluded: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)


class DatasetAdapter:
    """Base class; subclass and register a `name`."""
    name: str = ""

    def collect(self, root: Path, task_config: dict, inventory: list[dict]) -> AdapterResult:
        raise NotImplementedError
