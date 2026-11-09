"""D4 admission: turn a reviewed label table into a person-isolated manifest.

Nothing here guesses a label. A manifest is written only when every selected
speaker has a `review_status=confirmed` row in `label_review.csv`; otherwise
the job is blocked and only review-status/report files are produced.

Rules enforced before writing `manifest.csv`:
- every selected speaker is confirmed, with non-empty subject id and label;
- one confirmed subject never appears under two labels (person isolation);
- one origin (same utterance across microphones) never appears twice;
- an optional `selection.mic` keeps exactly one microphone per origin;
- `group_id` is derived from the confirmed subject, so any later split can be
  person-disjoint without re-reading this module.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .schema import (
    CATEGORY_AUDIO,
    COL_CATEGORY,
    COL_DECODE_STATUS,
    COL_DIRECTORY_GROUP,
    COL_DURATION_DECODED_SEC,
    COL_LABEL_RAW,
    COL_MIC_TOKEN,
    COL_ORIGIN_ID,
    COL_PARSE_STATUS,
    COL_RELATIVE_PATH,
    COL_REVIEW_STATUS,
    COL_SAMPLE_RATE,
    COL_CHANNELS,
    COL_SHA256,
    COL_SPEAKER_TOKEN,
    LABEL_REVIEW_COLUMNS,
    MANIFEST_COLUMNS,
)
from .prepare import eligible_rows

REVIEW_CONFIRMED = "confirmed"
REVIEW_PENDING = "pending"

STATUS_APPROVED = "approved"
STATUS_LABEL_REVIEW_MISSING = "blocked_label_review_missing"
STATUS_LABEL_REVIEW_INCOMPLETE = "blocked_label_review_incomplete"
STATUS_SUBJECT_CONFLICT = "blocked_subject_label_conflict"
STATUS_MIC_NOT_SELECTED = "blocked_mic_not_selected"
STATUS_ORIGIN_DUPLICATE = "blocked_multiple_rows_per_origin"
STATUS_ORIGIN_CONFLICT = "blocked_origin_conflict"
STATUS_NO_ROWS = "blocked_no_eligible_rows"


class AdmissionError(RuntimeError):
    """Raised for invalid review files or unsafe output paths."""


def _s(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN from pandas
        return ""
    return str(value).strip()


def _issue(issue_type: str, severity: str, relative_path: str, detail: str) -> dict:
    return {"issue_type": issue_type, "severity": severity,
            "relative_path": relative_path, "detail": detail}


# ---------------------------------------------------------------- review table

def load_label_review(path: Path) -> dict[tuple[str, str], dict]:
    """Load review rows keyed by (speaker_token, directory_group).

    The returned rows include `_line` (1-based data line) for traceability.
    """
    if not path.is_file():
        return {}
    table: dict[tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in LABEL_REVIEW_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise AdmissionError(
                f"label review {path} missing columns {missing}; "
                f"required: {LABEL_REVIEW_COLUMNS}")
        for line_no, raw in enumerate(reader, start=2):
            row = {c: _s(raw.get(c)) for c in LABEL_REVIEW_COLUMNS}
            key = (row["speaker_token"], row["directory_group"])
            if key == ("", ""):
                continue
            row["_line"] = line_no
            table[key] = row
    return table


def review_inventory(rows: list[dict]) -> list[dict]:
    """Speakers observed in eligible audio, with file/origin/mic coverage."""
    buckets: dict[tuple[str, str], dict] = {}
    for row in eligible_rows(rows):
        if _s(row.get(COL_PARSE_STATUS)) != "ok":
            continue
        speaker = _s(row.get(COL_SPEAKER_TOKEN))
        if not speaker:
            continue
        key = (speaker, _s(row.get(COL_DIRECTORY_GROUP)))
        item = buckets.setdefault(key, {
            "speaker_token": speaker,
            "directory_group": key[1],
            "n_files": 0,
            "origins": set(),
            "mics": set(),
        })
        item["n_files"] += 1
        item["origins"].add(_s(row.get(COL_ORIGIN_ID)))
        mic = _s(row.get(COL_MIC_TOKEN))
        if mic:
            item["mics"].add(mic)
    out = []
    for item in buckets.values():
        out.append({
            "speaker_token": item["speaker_token"],
            "directory_group": item["directory_group"],
            "n_files": item["n_files"],
            "n_origins": len(item["origins"]),
            "mic_tokens": ",".join(sorted(item["mics"])),
        })
    return sorted(out, key=lambda r: (r["directory_group"], r["speaker_token"]))


def review_status_rows(inventory: list[dict], review: dict) -> list[dict]:
    out = []
    for item in inventory:
        entry = review.get((item["speaker_token"], item["directory_group"]), {})
        status = entry.get("review_status", "") or REVIEW_PENDING
        out.append({
            **item,
            "review_status": status,
            "confirmed_subject_id": entry.get("confirmed_subject_id", ""),
            "confirmed_label": entry.get("confirmed_label", ""),
            "evidence": entry.get("evidence", ""),
        })
    return out


def validate_review(review: dict, inventory: list[dict]) -> dict:
    """Return coverage/conflict facts; `blocking` lists admission blockers."""
    blocking: list[dict] = []
    missing = [f"{i['speaker_token']}@{i['directory_group']}" for i in inventory
               if (i["speaker_token"], i["directory_group"]) not in review]
    unconfirmed, incomplete = [], []
    confirmed: dict[tuple[str, str], dict] = {}

    for item in inventory:
        key = (item["speaker_token"], item["directory_group"])
        entry = review.get(key)
        if entry is None:
            continue
        if entry.get("review_status", "").lower() != REVIEW_CONFIRMED:
            unconfirmed.append(f"{key[0]}@{key[1]}")
            continue
        if not entry.get("confirmed_subject_id") or not entry.get("confirmed_label"):
            incomplete.append(f"{key[0]}@{key[1]}")
            continue
        confirmed[key] = entry

    subject_labels: dict[str, set[str]] = defaultdict(set)
    subject_speakers: dict[str, set[str]] = defaultdict(set)
    for (speaker, _group), entry in confirmed.items():
        subject_labels[entry["confirmed_subject_id"]].add(entry["confirmed_label"])
        subject_speakers[entry["confirmed_subject_id"]].add(speaker)
    subject_conflicts = {s: sorted(v) for s, v in subject_labels.items() if len(v) > 1}
    subject_aliases = {s: sorted(v) for s, v in subject_speakers.items() if len(v) > 1}

    if missing:
        blocking.append(_issue(
            "label_review_missing_rows", "error", "",
            f"{len(missing)} speakers have no review row: {missing[:10]}"))
    if unconfirmed:
        blocking.append(_issue(
            "label_review_unconfirmed", "error", "",
            f"{len(unconfirmed)} speakers are not confirmed: {unconfirmed[:10]}"))
    if incomplete:
        blocking.append(_issue(
            "label_review_incomplete_confirmed_row", "error", "",
            f"confirmed rows missing subject/label: {incomplete[:10]}"))
    if subject_conflicts:
        blocking.append(_issue(
            "subject_label_conflict", "error", "",
            f"one subject with multiple labels: {dict(list(subject_conflicts.items())[:10])}"))
    if subject_aliases:
        blocking.append(_issue(
            "subject_speaker_alias_conflict", "error", "",
            f"one subject id mapped to multiple speaker tokens: "
            f"{dict(list(subject_aliases.items())[:10])}"))
    return {
        "missing": missing,
        "unconfirmed": unconfirmed,
        "incomplete": incomplete,
        "subject_conflicts": subject_conflicts,
        "subject_aliases": subject_aliases,
        "blocking": blocking,
        "confirmed": confirmed,
    }


def write_review_template(path: Path, inventory: list[dict], force: bool = False) -> bool:
    """Write a pending template for every observed speaker. Never overwrites."""
    if path.exists() and not force:
        raise AdmissionError(f"review file already exists (use --force to replace): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_REVIEW_COLUMNS)
        writer.writeheader()
        for item in inventory:
            writer.writerow({
                "speaker_token": item["speaker_token"],
                "directory_group": item["directory_group"],
                "confirmed_subject_id": "",
                "confirmed_label": "",
                "evidence": "",
                "review_status": REVIEW_PENDING,
            })
    return True


# ---------------------------------------------------------------- selection

def select_rows(rows: list[dict], selection: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Apply mic/origin rules before any label is attached.

    Returns (selected_rows, issues, block_statuses). A missing mic choice is a
    decision gap, not something this module may fill in silently.
    """
    issues: list[dict] = []
    blockers: list[str] = []
    candidates = [r for r in eligible_rows(rows)
                  if _s(r.get(COL_PARSE_STATUS)) == "ok"]
    if not candidates:
        return [], issues, [STATUS_NO_ROWS]

    has_mic = any(_s(r.get(COL_MIC_TOKEN)) for r in candidates)
    require_mic = bool(selection.get("require_mic", has_mic))
    mic = _s(selection.get("mic"))
    if require_mic and not mic:
        blockers.append(STATUS_MIC_NOT_SELECTED)
        issues.append(_issue(
            "mic_not_selected", "error", "",
            "this dataset has multiple microphones per utterance; set selection.mic "
            "in the task config (or --mic) before building an approved manifest"))
    if mic:
        before = len(candidates)
        candidates = [r for r in candidates if _s(r.get(COL_MIC_TOKEN)) == mic]
        if not candidates:
            blockers.append(STATUS_MIC_NOT_SELECTED)
            issues.append(_issue(
                "mic_filter_empty", "error", "",
                f"selection.mic={mic!r} matched 0 of {before} eligible rows"))

    if selection.get("one_per_origin", True):
        by_origin: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in candidates:
            by_origin[(_s(row.get(COL_SPEAKER_TOKEN)),
                       _s(row.get(COL_ORIGIN_ID)) or _s(row.get(COL_RELATIVE_PATH)))].append(row)
        duplicates = {k: len(v) for k, v in by_origin.items() if len(v) > 1}
        if duplicates:
            blockers.append(STATUS_ORIGIN_DUPLICATE)
            issues.append(_issue(
                "multiple_rows_per_origin", "error", "",
                f"{len(duplicates)} origins have >1 selected row after the mic rule; "
                f"examples: {list(duplicates.items())[:10]}"))
    return candidates, issues, blockers


# ---------------------------------------------------------------- manifest

def build_manifest(rows: list[dict], review: dict, dataset_id: str,
                   review_path: Path, selection: dict) -> tuple[list[dict], list[dict]]:
    """Attach confirmed labels and person-isolated group ids."""
    issues: list[dict] = []
    variant = _s(selection.get("variant")) or "original"
    elicitation = _s(selection.get("elicitation")) or "unknown"
    manifest: list[dict] = []
    origin_owner: dict[str, tuple[str, str]] = {}

    for row in sorted(rows, key=lambda r: _s(r.get(COL_RELATIVE_PATH))):
        key = (_s(row.get(COL_SPEAKER_TOKEN)), _s(row.get(COL_DIRECTORY_GROUP)))
        entry = review.get(key)
        if entry is None or entry.get("review_status", "").lower() != REVIEW_CONFIRMED:
            issues.append(_issue(
                "manifest_row_without_confirmed_label", "error",
                _s(row.get(COL_RELATIVE_PATH)),
                f"no confirmed review row for {key[0]}@{key[1]}"))
            continue
        subject = entry["confirmed_subject_id"]
        label = entry["confirmed_label"]
        origin = _s(row.get(COL_ORIGIN_ID)) or _s(row.get(COL_RELATIVE_PATH))
        previous = origin_owner.get(origin)
        if previous is not None and previous != (subject, label):
            issues.append(_issue(
                "origin_subject_conflict", "error", _s(row.get(COL_RELATIVE_PATH)),
                f"origin {origin!r} already owned by {previous}; refusing mixed ownership"))
            continue
        origin_owner[origin] = (subject, label)
        manifest.append({
            "sample_id": _s(row.get(COL_RELATIVE_PATH)),
            COL_RELATIVE_PATH: _s(row.get(COL_RELATIVE_PATH)),
            "subject_id": subject,
            "group_id": f"{dataset_id}:{subject}",
            "origin_id": f"{dataset_id}:{origin}",
            "label_raw": _s(row.get(COL_LABEL_RAW)) or _s(row.get(COL_DIRECTORY_GROUP)),
            "label": label,
            "variant": variant,
            "elicitation": elicitation,
            COL_SAMPLE_RATE: row.get(COL_SAMPLE_RATE),
            COL_DURATION_DECODED_SEC: row.get(COL_DURATION_DECODED_SEC),
            COL_CHANNELS: row.get(COL_CHANNELS),
            COL_SHA256: _s(row.get(COL_SHA256)),
            "official_split": "",
            "label_source": f"{review_path.name}#line{entry.get('_line', '')}",
        })

    # Final isolation check across the built manifest.
    seen: dict[str, set[str]] = defaultdict(set)
    for item in manifest:
        seen[item["subject_id"]].add(item["label"])
    for subject, labels in seen.items():
        if len(labels) > 1:
            issues.append(_issue(
                "subject_label_conflict", "error", "",
                f"subject {subject} appears with labels {sorted(labels)}"))
    return manifest, issues


def manifest_fingerprint(manifest: list[dict]) -> str:
    import hashlib
    h = hashlib.sha256()
    for row in sorted(manifest, key=lambda r: r["sample_id"]):
        h.update(f"{row['sample_id']}|{row.get(COL_SHA256)}|{row['label']}\n".encode())
    return h.hexdigest()
