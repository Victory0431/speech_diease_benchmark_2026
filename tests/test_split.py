"""D5: person-disjoint stratified splits."""
from __future__ import annotations

import pytest

from sddbench.data.split import SplitError, assign_splits


def _manifest(n_per_label=10, rows_per_subject=3):
    rows = []
    for label, prefix in (("control", "C"), ("disorder", "D")):
        for i in range(n_per_label):
            subject = f"{prefix}{i:02d}"
            for j in range(rows_per_subject):
                rows.append({
                    "sample_id": f"{subject}_{j}.wav",
                    "group_id": f"demo:{subject}",
                    "label": label,
                })
    return rows


def test_split_is_person_disjoint_and_stratified():
    rows, summary = assign_splits(
        _manifest(), seed=7, fractions={"train": 0.6, "val": 0.2, "test": 0.2})
    assert len(rows) == 60
    by_split = {}
    for row in rows:
        by_split.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(v) == 1 for v in by_split.values())
    assert summary["subjects_by_split"]["train"]["n_subjects"] == 12
    assert summary["subjects_by_split"]["val"]["n_subjects"] == 4
    assert summary["subjects_by_split"]["test"]["n_subjects"] == 4
    for split, expected in (("train", 18), ("val", 6), ("test", 6)):
        assert summary["by_split"][split]["labels"] == {
            "control": expected, "disorder": expected}


def test_split_is_deterministic_and_seed_sensitive():
    a, _ = assign_splits(_manifest(), 1, {"train": 0.6, "val": 0.2, "test": 0.2})
    b, _ = assign_splits(_manifest(), 1, {"train": 0.6, "val": 0.2, "test": 0.2})
    c, _ = assign_splits(_manifest(), 2, {"train": 0.6, "val": 0.2, "test": 0.2})
    assert a == b
    assert a != c


def test_split_rejects_tiny_class_and_bad_fractions():
    tiny = _manifest(n_per_label=2, rows_per_subject=1)
    with pytest.raises(SplitError):
        assign_splits(tiny, 0, {"train": 0.6, "val": 0.2, "test": 0.2})
    with pytest.raises(SplitError):
        assign_splits(_manifest(), 0, {"train": 0.7, "val": 0.2, "test": 0.2})
    with pytest.raises(SplitError):
        assign_splits(_manifest(), 0, {"train": 0.5, "val": 0.5})


def test_split_rejects_subject_with_two_labels():
    rows = _manifest(n_per_label=4, rows_per_subject=1)
    rows[0]["group_id"] = rows[-1]["group_id"]
    with pytest.raises(SplitError):
        assign_splits(rows, 0, {"train": 0.6, "val": 0.2, "test": 0.2})
