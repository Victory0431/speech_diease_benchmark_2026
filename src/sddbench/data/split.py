"""D5: deterministic, label-stratified, group-disjoint splits.

The unit of assignment is `group_id`. With the default `grouping: subject`
that group is a confirmed subject, so every recording of a person lands in
exactly one split. Datasets without speaker metadata use `grouping: file`,
where each recording is its own group (leakage caveat documented in D10).
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from .schema import COL_GROUP_ID, COL_LABEL, COL_RELATIVE_PATH


class SplitError(RuntimeError):
    """Raised when a person-disjoint stratified split is impossible."""


def _digest(seed: int, subject: str) -> str:
    return hashlib.sha256(f"{seed}|{subject}".encode()).hexdigest()


def _allocate(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation with at least one subject per split."""
    if n < len(fractions):
        raise SplitError(
            f"class has {n} subjects but {len(fractions)} splits are requested; "
            "at least one subject per split is required")
    exact = {k: n * v for k, v in fractions.items()}
    counts = {k: int(v) for k, v in exact.items()}
    for key in fractions:  # minimum one subject per split
        if counts[key] == 0:
            counts[key] = 1
    while sum(counts.values()) > n:
        donor = max((k for k in counts if counts[k] > 1), key=lambda k: counts[k])
        counts[donor] -= 1
    remaining = n - sum(counts.values())
    order = sorted(fractions, key=lambda k: (-(exact[k] - int(exact[k])), k))
    idx = 0
    while remaining > 0:
        counts[order[idx % len(order)]] += 1
        remaining -= 1
        idx += 1
    return counts


def assign_splits(manifest: list[dict], seed: int,
                  fractions: dict[str, float]) -> tuple[list[dict], dict]:
    if set(fractions) != {"train", "val", "test"}:
        raise SplitError("fractions must define train/val/test")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"split fractions must sum to 1.0 (got {total})")

    subjects: dict[str, dict] = {}
    for row in manifest:
        group = str(row.get(COL_GROUP_ID) or "").strip()
        label = str(row.get(COL_LABEL) or "").strip()
        if not group or not label:
            raise SplitError(
                "manifest rows need group_id and label; datasets without speaker "
                "metadata must be prepared with grouping: file")
        previous = subjects.get(group)
        if previous is not None and previous["label"] != label:
            raise SplitError(f"subject {group} appears with labels "
                             f"{previous['label']!r} and {label!r}")
        subjects[group] = {"group_id": group, "label": label}

    by_label: dict[str, list[str]] = defaultdict(list)
    for group, item in subjects.items():
        by_label[item["label"]].append(group)

    assignment: dict[str, str] = {}
    allocation: dict[str, dict] = {}
    for label, groups in sorted(by_label.items()):
        counts = _allocate(len(groups), fractions)
        ordered = sorted(groups, key=lambda g: _digest(seed, g))
        cursor = 0
        for split in ("train", "val", "test"):
            selected = ordered[cursor:cursor + counts[split]]
            cursor += counts[split]
            for group in selected:
                assignment[group] = split
        allocation[label] = counts

    rows = []
    for row in manifest:
        group = str(row[COL_GROUP_ID]).strip()
        rows.append({
            "sample_id": row.get("sample_id") or row[COL_RELATIVE_PATH],
            COL_GROUP_ID: group,
            COL_LABEL: row[COL_LABEL],
            "split": assignment[group],
        })

    summary = {
        "seed": seed,
        "fractions": fractions,
        "n_samples": len(rows),
        "n_subjects": len(subjects),
        "n_groups": len(subjects),
        "allocation_by_label": allocation,
        "by_split": {},
        "subjects_by_split": {},
    }
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        groups_in_split = {r[COL_GROUP_ID] for r in part}
        summary["by_split"][split] = {
            "n_samples": len(part),
            "labels": dict(sorted(Counter(r[COL_LABEL] for r in part).items())),
        }
        summary["subjects_by_split"][split] = {
            "n_subjects": len(groups_in_split),
            "labels": dict(sorted(Counter(subjects[g]["label"] for g in groups_in_split).items())),
        }

    # Hard isolation checks.
    seen: dict[str, str] = {}
    for row in rows:
        group = row[COL_GROUP_ID]
        if group in seen and seen[group] != row["split"]:
            raise SplitError(f"person leakage: {group} in {seen[group]} and {row['split']}")
        seen[group] = row["split"]
    if len(rows) != len(manifest):
        raise SplitError("split row count does not match manifest")
    return rows, summary
