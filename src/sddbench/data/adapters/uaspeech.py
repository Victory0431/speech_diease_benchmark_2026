"""UASpeech adapter: explains naming and directory facts, nothing more.

Local arrangement (verified on disk, 2026-11-09, full scan):
  <root>/Disorder/*.wav   69,615 files, 13 speakers, all C-prefixed tokens
  <root>/Healthy/*.wav    73,675 files, 15 speakers, no C prefix
Filename: <speaker>_<block>_<prompt>_<mic>.wav, e.g. CF02_B1_C1_M2.wav.
Verified prompt token domain (candidates): C0..C12 digits, D0..D9,
UW<n> unusual words, CW<n> check words, LA..LZ long/sentence tokens.
255 distinct prompt tokens x 3 blocks x 7 mics = 5,355 files per full speaker;
several speakers have fewer files (missing recordings are reported, never
imputed).

Facts recorded as CANDIDATES (data-layer doc §5):
- directory_group/label_raw = "Disorder"/"Healthy": a directory fact, never an
  approved training label.
- speaker_token kept whole (CF02 stays CF02; the "C" is NOT stripped). In the
  official corpus convention C-prefix denotes control speakers [R3], yet here
  every C-prefixed speaker sits under Disorder/ -- a suspected inverted mapping
  that MUST be resolved via label_review before any supervised use.
- origin_id (candidate): same speaker+block+prompt across mics M2..M8 is the
  same utterance (multi-mic capture), not independent subjects/sources.
- mic coverage is measured by stats; mic selection is a human decision (D1
  report), never automatic.

label_review.csv (optional, under workspace): when present, confirmed rows
override review_status; when absent everything stays `pending`.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from .base import AdapterResult, DatasetAdapter
from ..schema import (
    COL_CATEGORY,
    COL_DIRECTORY_GROUP,
    COL_FILE_NAME,
    COL_LABEL_RAW,
    COL_MIC_TOKEN,
    COL_BLOCK_TOKEN,
    COL_ORIGIN_ID,
    COL_PARSE_STATUS,
    COL_PROMPT_TOKEN,
    COL_RELATIVE_PATH,
    COL_REVIEW_STATUS,
    COL_SPEAKER_TOKEN,
    CATEGORY_AUDIO,
    LABEL_REVIEW_COLUMNS,
)

STEM_PATTERN = re.compile(
    r"^(?P<speaker>[A-Za-z]+\d+)_(?P<block>B\d+)_(?P<prompt>[A-Z]+\d*)_(?P<mic>M\d+)$"
)


def parse_stem(stem: str) -> dict | None:
    m = STEM_PATTERN.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return {
        "speaker_token": d["speaker"],
        "block_token": d["block"],
        "prompt_token": d["prompt"],
        "mic_token": d["mic"],
    }


def load_label_review(path: Path) -> dict[tuple[str, str], dict]:
    """Load confirmed rows keyed by (speaker_token, directory_group)."""
    table: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return table
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("speaker_token", "").strip(),
                   row.get("directory_group", "").strip())
            if key == ("", ""):
                continue
            table[key] = {c: row.get(c, "").strip() for c in LABEL_REVIEW_COLUMNS}
    return table


class UASpeechAdapter(DatasetAdapter):
    name = "uaspeech"

    def _issue(self, issue_type: str, severity: str, relative_path: str, detail: str) -> dict:
        return {
            "issue_type": issue_type, "severity": severity,
            "relative_path": relative_path, "detail": detail,
        }

    def collect(self, root: Path, task_config: dict, inventory: list[dict]) -> AdapterResult:
        result = AdapterResult()
        prefix_by_dir: dict[str, Counter] = {}
        parse_failed: list[str] = []

        review_path_cfg = (task_config or {}).get("label_review") or {}
        workspace = (task_config or {}).get("_workspace_root")
        review_file = Path(review_path_cfg.get("file", "label_review/uaspeech_label_review.csv"))
        review_path = (Path(workspace) / review_file) if workspace else review_file
        review = load_label_review(review_path)
        if not review:
            result.issues.append(self._issue(
                "label_review_missing", "info", "",
                "label_review.csv not found; every speaker stays review_status=pending. "
                "No approved classification manifest can be produced until D4 review.",
            ))

        for row in inventory:
            if row.get(COL_CATEGORY) != CATEGORY_AUDIO:
                continue
            rel = row[COL_RELATIVE_PATH]
            parts = Path(rel).parts
            directory_group = parts[0] if parts else ""
            stem = Path(row[COL_FILE_NAME]).stem
            parsed = parse_stem(stem)
            fields = {
                COL_DIRECTORY_GROUP: directory_group,
                COL_LABEL_RAW: directory_group,  # directory fact only
                COL_SPEAKER_TOKEN: "",
                COL_BLOCK_TOKEN: "",
                COL_PROMPT_TOKEN: "",
                COL_MIC_TOKEN: "",
                COL_ORIGIN_ID: "",
                COL_PARSE_STATUS: "parse_failed",
                COL_REVIEW_STATUS: "pending",
            }
            if parsed is None:
                parse_failed.append(rel)
            else:
                fields.update(parsed)
                fields[COL_PARSE_STATUS] = "ok"
                fields[COL_ORIGIN_ID] = (
                    f"{parsed['speaker_token']}|{parsed['block_token']}|{parsed['prompt_token']}"
                )
                prefix_by_dir.setdefault(directory_group, Counter())[parsed["speaker_token"]] += 1
                key = (parsed["speaker_token"], directory_group)
                if key in review:
                    r = review[key]
                    if r.get("review_status") == "confirmed":
                        fields[COL_REVIEW_STATUS] = "confirmed"
            result.enrichment[rel] = fields

        if parse_failed:
            for rel in parse_failed:
                result.issues.append(self._issue(
                    "filename_parse_failed", "error", rel,
                    f"stem does not match {STEM_PATTERN.pattern}",
                ))

        # One consolidated naming-vs-directory observation (fact, not judgment).
        prefix_segregation = {
            d: {"c_prefixed_speakers": sorted({t for t in c if t.upper().startswith("C")}),
                "other_speakers": sorted({t for t in c if not t.upper().startswith("C")}),
                "n_files": sum(c.values())}
            for d, c in prefix_by_dir.items()
        }
        result.issues.append(self._issue(
            "prefix_directory_segregation_review", "warning", "",
            "speaker-token prefix classes are fully segregated by directory "
            f"({prefix_segregation}). Official UASpeech convention uses C-prefix for "
            "CONTROL speakers [R3]; if this local arrangement is inverted, directory "
            "names cannot be used as labels. Must be resolved in label_review (D4)."
        ))
        return result
