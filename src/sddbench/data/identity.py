"""Content fingerprints for the prepared/split identity chain (audit F04).

These hashes are computed from the *actual parsed content*, not from a saved
string: callers recompute them on the files they are about to use and compare
with what admission/split recorded. Any change to a label, group, origin,
source hash, filter or task definition therefore invalidates old references
instead of silently reusing them.

`task_fingerprint` covers the whole resolved task YAML (labels, class order,
positive class, grouping, selection, layout, admission rules, version). Only
private keys starting with `_` are excluded, matching what `prepare` writes to
`task.yaml`.
"""
from __future__ import annotations

import hashlib
import json

from .schema import (
    COL_CHANNELS,
    COL_DURATION_DECODED_SEC,
    COL_ELICITATION,
    COL_GROUP_ID,
    COL_LABEL,
    COL_LABEL_RAW,
    COL_LABEL_SOURCE,
    COL_OFFICIAL_SPLIT,
    COL_ORIGIN_ID,
    COL_RELATIVE_PATH,
    COL_SAMPLE_RATE,
    COL_SHA256,
    COL_SUBJECT_ID,
    COL_VARIANT,
)

MANIFEST_FINGERPRINT_SCHEMA = "2"

# Everything that identifies *which recording* a row is and *which science* it
# belongs to. Two manifests that differ in any of these must not share a hash.
MANIFEST_IDENTITY_COLUMNS = [
    "sample_id", COL_RELATIVE_PATH, COL_SUBJECT_ID, COL_GROUP_ID, COL_ORIGIN_ID,
    COL_LABEL_RAW, COL_LABEL, COL_VARIANT, COL_ELICITATION,
    COL_SAMPLE_RATE, COL_DURATION_DECODED_SEC, COL_CHANNELS, COL_SHA256,
    COL_OFFICIAL_SPLIT, COL_LABEL_SOURCE,
]


def _s(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN from pandas
        return ""
    return str(value).strip()


def manifest_fingerprint(manifest: list[dict], task_hash: str = "") -> str:
    """Digest of the manifest content + the frozen task identity."""
    h = hashlib.sha256()
    h.update(f"manifest-schema={MANIFEST_FINGERPRINT_SCHEMA}\n".encode())
    h.update(f"task={task_hash}\n".encode())
    for row in sorted(manifest, key=lambda r: _s(r.get("sample_id"))):
        h.update("|".join(_s(row.get(col)) for col in MANIFEST_IDENTITY_COLUMNS).encode())
        h.update(b"\n")
    return h.hexdigest()


def task_fingerprint(task_cfg: dict) -> str:
    """Digest of the resolved task definition (labels/order/grouping/filters)."""
    payload = {k: v for k, v in (task_cfg or {}).items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def split_fingerprint(rows: list[dict]) -> str:
    """Digest of the sample -> (group, label, split) assignment table."""
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: _s(r.get("sample_id"))):
        h.update("|".join(_s(row.get(col)) for col in
                          ("sample_id", "group_id", "label", "split")).encode())
        h.update(b"\n")
    return h.hexdigest()
