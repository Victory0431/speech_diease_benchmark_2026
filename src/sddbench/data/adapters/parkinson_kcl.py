"""Parkinson KCL 2017 adapter: naming facts only, never an approved label.

Local arrangement (verified on disk, 2026-11-09):
  <root>/Healthy/*.wav     42 files, 22 distinct ID tokens (44.1 kHz, mono, PCM_24)
  <root>/Parkinson/*.wav   31 files, 16 distinct ID tokens
Filename: ID<nn>[_](hc|pd)_<session tokens>[_<take>].wav, e.g.
  ID15_hc_0_0_0.wav      Healthy, "hc" = healthy-control candidate
  ID02_pd_2_0_0_1.wav    Parkinson, "pd" = Parkinson candidate
  ID22hc_0_0_0.wav       local anomaly: missing underscore before "hc"

Recorded as CANDIDATES:
- directory_group/label_raw = directory fact; "hc"/"pd" are filename tokens.
  Both must be confirmed in label_review.csv before an approved manifest.
- speaker_token = ID<n>; the token is kept even when the separator is missing.
- block_token = remaining session/take tokens (meaning unverified).
- origin_id = speaker|kind|session tokens (each file distinct); repeated `_1`
  suffixes may be second takes, so they are reported for review rather than
  silently merged or dropped.
- mic_token stays empty: this corpus has one channel and no microphone token.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .base import AdapterResult, DatasetAdapter
from ..schema import (
    CATEGORY_AUDIO,
    COL_BLOCK_TOKEN,
    COL_CATEGORY,
    COL_DIRECTORY_GROUP,
    COL_FILE_NAME,
    COL_LABEL_RAW,
    COL_MIC_TOKEN,
    COL_ORIGIN_ID,
    COL_PARSE_STATUS,
    COL_PROMPT_TOKEN,
    COL_RELATIVE_PATH,
    COL_REVIEW_STATUS,
    COL_SPEAKER_TOKEN,
)

STEM_PATTERN = re.compile(
    r"^(?P<speaker>ID\d+)_?(?P<kind>hc|pd)_(?P<rest>.+)$", re.IGNORECASE)


def parse_stem(stem: str) -> dict | None:
    m = STEM_PATTERN.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return {
        "speaker_token": d["speaker"].upper(),
        "kind": d["kind"].lower(),
        "rest": d["rest"],
        "missing_separator": not stem.startswith(f"{d['speaker']}_"),
    }


class ParkinsonKCLAdapter(DatasetAdapter):
    name = "parkinson_kcl"

    def _issue(self, issue_type: str, severity: str, relative_path: str, detail: str) -> dict:
        return {
            "issue_type": issue_type, "severity": severity,
            "relative_path": relative_path, "detail": detail,
        }

    def collect(self, root: Path, task_config: dict, inventory: list[dict]) -> AdapterResult:
        result = AdapterResult()
        kind_by_dir: dict[str, Counter] = {}
        parse_failed: list[str] = []
        take_variants: dict[str, list[str]] = {}

        for row in inventory:
            if row.get(COL_CATEGORY) != CATEGORY_AUDIO:
                continue
            rel = row[COL_RELATIVE_PATH]
            parts = Path(rel).parts
            directory_group = parts[0] if len(parts) > 1 else ""
            stem = Path(row[COL_FILE_NAME]).stem
            parsed = parse_stem(stem)
            fields = {
                COL_DIRECTORY_GROUP: directory_group,
                COL_LABEL_RAW: directory_group,
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
                fields.update({
                    COL_SPEAKER_TOKEN: parsed["speaker_token"],
                    COL_BLOCK_TOKEN: parsed["rest"],
                    COL_ORIGIN_ID: (f"{parsed['speaker_token']}|{parsed['kind']}|"
                                    f"{parsed['rest']}"),
                    COL_PARSE_STATUS: "ok",
                })
                kind_by_dir.setdefault(directory_group, Counter())[parsed["kind"]] += 1
                if parsed["missing_separator"]:
                    result.issues.append(self._issue(
                        "naming_missing_separator", "warning", rel,
                        "speaker token and hc/pd token are not separated by '_'; "
                        "parsed as candidate, verify against source metadata"))
                # `..._x_1` next to `..._x` is a candidate repeat take.
                base_rest = parsed["rest"][:-2] if parsed["rest"].endswith("_1") else parsed["rest"]
                take_variants.setdefault(
                    f"{parsed['speaker_token']}|{parsed['kind']}|{base_rest}", []).append(rel)
            result.enrichment[rel] = fields

        for rel in parse_failed:
            result.issues.append(self._issue(
                "filename_parse_failed", "error", rel,
                f"stem does not match {STEM_PATTERN.pattern}"))

        for key, rels in take_variants.items():
            if len(rels) > 1:
                result.issues.append(self._issue(
                    "repeat_take_review", "warning", "",
                    f"{key}: multiple files share the same session prefix but differ "
                    f"by a numeric suffix ({sorted(rels)}); confirm whether they are "
                    "repeat takes before treating them as independent recordings"))
        result.issues.append(self._issue(
            "hc_pd_naming_review", "info", "",
            "filename kind tokens by directory: "
            f"{ {d: dict(c) for d, c in kind_by_dir.items()} }; "
            "hc/pd are candidates until label_review.csv is confirmed"))
        return result
