"""Built-in adapter for the common case: one label per top-level folder.

Dataset root holds N class folders and each audio file inherits its folder's
label. Nothing else is required: no per-dataset Python class, no filename
pattern, no review CSV. `label_map` only translates folder names into the
label vocabulary used by the benchmark (e.g. Healthy/Control -> healthy).

This adapter records facts only. Mapping folder -> label happens in admission
(`review_from_label_folders`), and `label_source=directory` is written to the
manifest so the decision stays auditable.
"""
from __future__ import annotations

from pathlib import Path

from .base import AdapterResult, DatasetAdapter
from ..schema import (
    CATEGORY_AUDIO,
    COL_BLOCK_TOKEN,
    COL_CATEGORY,
    COL_DIRECTORY_GROUP,
    COL_LABEL_RAW,
    COL_MIC_TOKEN,
    COL_ORIGIN_ID,
    COL_PARSE_STATUS,
    COL_PROMPT_TOKEN,
    COL_RELATIVE_PATH,
    COL_REVIEW_STATUS,
    COL_SPEAKER_TOKEN,
)


class LabelFoldersAdapter(DatasetAdapter):
    name = "label_folders"

    def collect(self, root: Path, task_config: dict, inventory: list[dict]) -> AdapterResult:
        result = AdapterResult()
        seen_groups: set[str] = set()

        for row in inventory:
            if row.get(COL_CATEGORY) != CATEGORY_AUDIO:
                continue
            rel = row[COL_RELATIVE_PATH]
            parts = Path(rel).parts
            if not parts:
                continue
            folder = parts[0] if len(parts) > 1 else ""
            seen_groups.add(folder)
            if not folder:
                result.issues.append({
                    "issue_type": "file_outside_label_folder", "severity": "error",
                    "relative_path": rel,
                    "detail": "audio sits directly in the dataset root; put it under a label folder",
                })
            result.enrichment[rel] = {
                COL_DIRECTORY_GROUP: folder,
                COL_LABEL_RAW: folder,
                COL_SPEAKER_TOKEN: "",
                COL_BLOCK_TOKEN: "",
                COL_PROMPT_TOKEN: "",
                COL_MIC_TOKEN: "",
                COL_ORIGIN_ID: rel,
                COL_PARSE_STATUS: "ok" if folder else "parse_failed",
                COL_REVIEW_STATUS: "pending",
            }

        for folder in sorted(g for g in seen_groups if g):
            result.issues.append({
                "issue_type": "label_folder", "severity": "info", "relative_path": folder,
                "detail": f"folder {folder!r} is treated as one label class",
            })
        return result
