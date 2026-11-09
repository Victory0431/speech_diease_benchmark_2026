"""Task preparation helpers shared by technical sampling and (later) D4 admission.

This stage only provides deterministic sample selection so a limited technical
run is not "the first 12 sorted files" from one speaker. Approved manifests and
label checks arrive with D4; nothing here turns a directory name into a label.
"""
from __future__ import annotations

import math
from collections import OrderedDict, defaultdict, deque

from .schema import CATEGORY_AUDIO, COL_DECODE_STATUS, COL_DIRECTORY_GROUP, \
    COL_RELATIVE_PATH, COL_SPEAKER_TOKEN, COL_CATEGORY


def _key(value) -> str:
    """Group key: None/NaN/empty collapse to '' (pandas gives NaN for blanks)."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def eligible_rows(rows: list[dict]) -> list[dict]:
    """Rows that can enter a view/feature job: decoded audio candidates."""
    return [r for r in rows
            if r.get(COL_CATEGORY) == CATEGORY_AUDIO and r.get(COL_DECODE_STATUS) == "ok"]


def stratified_sample(rows: list[dict], limit: int | None) -> tuple[list[dict], dict]:
    """Deterministic round-robin sample across directory groups and speakers.

    Order: groups sorted, speakers sorted, files sorted by relative path; one
    file per speaker per pass, cycling groups. With `limit=None` every row is
    returned in sorted order. The same inputs always yield the same selection,
    independent of dictionary/file ordering.
    """
    eligible = sorted(eligible_rows(rows), key=lambda r: r[COL_RELATIVE_PATH])
    summary = {
        "strategy": "all" if limit is None or limit >= len(eligible) else "stratified_round_robin",
        "n_eligible": len(eligible),
        "limit": limit,
    }
    if limit is None or limit >= len(eligible):
        chosen = eligible
    else:
        buckets: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        for r in eligible:
            buckets[_key(r.get(COL_DIRECTORY_GROUP))][_key(r.get(COL_SPEAKER_TOKEN))].append(r)
        group_names = sorted(buckets)
        speaker_order = OrderedDict(
            (g, sorted(buckets[g])) for g in group_names)
        max_speakers = max(len(s) for s in speaker_order.values())
        chosen = []
        while len(chosen) < limit:
            progressed = False
            # One file per group per speaker slot keeps groups interleaved
            # instead of exhausting the alphabetically first group.
            for slot in range(max_speakers):
                for g in group_names:
                    speakers = speaker_order[g]
                    if slot >= len(speakers):
                        continue
                    queue = buckets[g][speakers[slot]]
                    if queue:
                        chosen.append(queue.popleft())
                        progressed = True
                        if len(chosen) >= limit:
                            break
                if len(chosen) >= limit:
                    break
            if not progressed:
                break

    summary["n_selected"] = len(chosen)
    summary["selected_by_group"] = dict(sorted(
        _count(chosen, COL_DIRECTORY_GROUP).items()))
    summary["selected_by_speaker"] = dict(sorted(
        _count(chosen, COL_SPEAKER_TOKEN).items()))
    summary["sample_ids"] = [r[COL_RELATIVE_PATH] for r in chosen]
    return chosen, summary


def _count(rows: list[dict], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        key = _key(r.get(column)) or "(none)"
        out[key] = out.get(key, 0) + 1
    return out
