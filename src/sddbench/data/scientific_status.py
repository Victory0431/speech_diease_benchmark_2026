"""Scientific eligibility gate for a dataset task.

Two different questions are kept apart on purpose:

1. **Label evidence** - where do the class labels come from?  A folder name is
   acceptable evidence for a technical run, but it cannot resolve a known
   conflict with the original corpus documentation.
2. **Identity evidence** - can we show that different files are different
   people?  A folder name never answers this.  Without subject ids, or a
   documented ``one recording per person`` source, a run is technically useful
   but must not join the formal leakage-free benchmark.

The verdict is computed from explicit human declarations in the task YAML plus
the facts admission already produced.  Nothing here guesses a label, promotes
folder names to person evidence, or flips labels globally.
"""
from __future__ import annotations

from ..config import ConfigError

LEVEL_BENCHMARK = "benchmark_eligible"
LEVEL_TECHNICAL = "technical_only"
LEVEL_BLOCKED = "blocked"
LEVELS = (LEVEL_BENCHMARK, LEVEL_TECHNICAL, LEVEL_BLOCKED)

IDENTITY_SUBJECT_IDS = "subject_ids"
IDENTITY_ONE_RECORDING = "one_recording_per_person"
IDENTITY_UNKNOWN = "unknown"
IDENTITY_KINDS = (IDENTITY_SUBJECT_IDS, IDENTITY_ONE_RECORDING, IDENTITY_UNKNOWN)


def _s(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if v not in (None, "")]
    return [value]


def parse_scientific_status(task_cfg: dict) -> dict:
    """Normalize the ``scientific_status`` block of a task config.

    A missing block is **not** treated as eligible: unverified data defaults to
    ``technical_only`` so that adding a dataset never silently enters the
    formal benchmark.
    """
    raw = task_cfg.get("scientific_status")
    if isinstance(raw, str):
        raw = {"level": raw}
    raw = raw or {}
    level = _s(raw.get("level")) or LEVEL_TECHNICAL
    if level not in LEVELS:
        raise ConfigError(
            f"scientific_status.level must be one of {LEVELS}, got {level!r}")
    return {
        "level": level,
        "reason": _s(raw.get("reason")),
        "detail": _s(raw.get("detail")),
        "evidence": _list(raw.get("evidence")),
        "missing_evidence": _list(raw.get("missing_evidence")),
    }


def _identity_evidence(task_cfg: dict, grouping: str, n_confirmed_groups: int) -> dict:
    raw = task_cfg.get("identity")
    if isinstance(raw, str):
        raw = {"evidence": raw}
    raw = raw or {}
    kind = _s(raw.get("evidence"))
    reference = _s(raw.get("reference"))

    if grouping == "subject":
        # Confirmed subject ids are the normal evidence for person isolation.
        effective = IDENTITY_SUBJECT_IDS
    elif grouping == "file":
        effective = kind if kind in (IDENTITY_ONE_RECORDING,) else IDENTITY_UNKNOWN
    else:
        effective = IDENTITY_UNKNOWN

    if effective == IDENTITY_ONE_RECORDING and not reference:
        effective = IDENTITY_UNKNOWN

    status = "declared" if effective != IDENTITY_UNKNOWN else "missing"
    return {
        "kind": effective,
        "declared": kind or "",
        "reference": reference,
        "status": status,
        "n_confirmed_groups": int(n_confirmed_groups),
    }


def assess_scientific_status(task_cfg: dict, *, grouping: str,
                             n_confirmed_groups: int,
                             leakage_issues: list[dict] | None = None) -> dict:
    """Return the effective scientific verdict for a prepared task.

    ``leakage_issues`` are error-severity identity problems (duplicate content
    across groups/labels, origin conflicts).  Any of them makes the task
    ineligible even if the YAML claims otherwise.
    """
    declared = parse_scientific_status(task_cfg)
    identity = _identity_evidence(task_cfg, grouping, n_confirmed_groups)
    layout = _s(task_cfg.get("layout")) or "rich"
    label_evidence = {
        "source": "directory" if layout == "label_folders" else "label_review_csv",
        "status": "weak" if layout == "label_folders" else "reviewed",
    }
    blocking: list[dict] = []
    for issue in leakage_issues or []:
        if str(issue.get("severity")) == "error":
            blocking.append(issue)

    effective = declared["level"]
    reason = declared["reason"]
    if effective == LEVEL_BENCHMARK and identity["kind"] == IDENTITY_UNKNOWN:
        effective = LEVEL_TECHNICAL
        reason = reason or "identity_unverified"
    if blocking:
        effective = LEVEL_BLOCKED
        reason = "leakage_detected"
    if effective == LEVEL_BLOCKED and not reason:
        reason = "declared_blocked"

    return {
        "level": effective,
        "declared_level": declared["level"],
        "reason": reason,
        "detail": declared["detail"],
        "evidence": declared["evidence"],
        "missing_evidence": declared["missing_evidence"],
        "label_evidence": label_evidence,
        "identity_evidence": identity,
        "benchmark_eligible": effective == LEVEL_BENCHMARK,
        "blocking": blocking,
    }
