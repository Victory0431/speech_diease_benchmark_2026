"""Aggregate per-file audit facts into dataset statistics (data-layer doc §4).

Consumes enriched rows (scan + adapter + audit); decodes nothing. Duration
quantiles use NumPy `method="linear"` over successfully decoded durations,
equal-weight per recording. quick-mode header estimates are kept in separate
keys and never mixed with full-mode measurements.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from .schema import (
    CATEGORY_ARCHIVE,
    CATEGORY_AUDIO,
    CATEGORY_SIDECAR,
    CATEGORY_UNKNOWN,
    COL_ALL_ZERO,
    COL_CATEGORY,
    COL_DECODE_STATUS,
    COL_DIRECTORY_GROUP,
    COL_DURATION_DECODED_SEC,
    COL_DURATION_HEADER_SEC,
    COL_ENCODING,
    COL_CONTAINER,
    COL_EXTENSION,
    COL_EXTENSION_MISMATCH,
    COL_MIC_TOKEN,
    COL_N_SAMPLES,
    COL_PARSE_STATUS,
    COL_PROBE_STATUS,
    COL_PROMPT_TOKEN,
    COL_QUALITY_NEAR_FULL_RATIO,
    COL_QUALITY_NONFINITE,
    COL_QUALITY_PEAK,
    COL_QUALITY_RMS,
    COL_RELATIVE_PATH,
    COL_CHANNELS,
    COL_SAMPLE_RATE,
    COL_SPEAKER_TOKEN,
)

QUANTILES = [0.10, 0.25, 0.50, 0.60, 0.80, 0.90, 0.95, 0.99]
DEFAULT_LENGTH_CANDIDATES = [1.0, 2.0, 4.0, 6.0, 10.0]
DURATION_QUANTILE_METHOD = "linear"  # fixed; documented in report


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    out = {
        "n": int(arr.size),
        "min": float(arr.min()), "max": float(arr.max()),
        "mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
        "total": float(arr.sum()),
    }
    for q in QUANTILES:
        out[f"p{int(round(q * 100))}"] = float(np.quantile(arr, q, method=DURATION_QUANTILE_METHOD))
    return out


def _count_field(rows: list[dict], key: str) -> dict:
    c = Counter(str(r[key]) for r in rows if r.get(key) is not None)
    total = sum(c.values())
    return {k: {"count": v, "ratio": (v / total if total else None)} for k, v in sorted(c.items())}


def _length_candidates(durations: list[float], lengths: list[float]) -> list[dict]:
    """Retention/padding cost per candidate L (data-layer doc §4.4 formulas)."""
    rows = []
    if not durations:
        return rows
    d = np.asarray(durations, dtype=np.float64)
    n = d.size
    for L in lengths:
        valid = np.minimum(d, L)
        pad = np.maximum(0.0, L - d)
        cut = np.maximum(0.0, d - L)
        rows.append({
            "L_sec": L,
            "n_recordings": int(n),
            "n_shorter_than_L": int(np.count_nonzero(d < L)),
            "ratio_shorter_than_L": float(np.count_nonzero(d < L) / n),
            "n_longer_than_L": int(np.count_nonzero(d > L)),
            "ratio_longer_than_L": float(np.count_nonzero(d > L)),
            "valid_seconds_total": float(valid.sum()),
            "padded_seconds_total": float(pad.sum()),
            "truncated_seconds_total": float(cut.sum()),
            "total_padding_ratio": float(pad.sum() / (n * L)),
            "mean_per_recording_padding_ratio": float(np.mean(pad / L)),
            "retention_ratio": float(valid.sum() / d.sum()),
        })
    return rows


def _length_candidates_grouped(durations_by_group: dict[str, list[float]],
                               lengths: list[float]) -> list[dict]:
    out = []
    for group, durs in sorted(durations_by_group.items()):
        for row in _length_candidates(durs, lengths):
            out.append({"group": group, **row})
    return out


def collect_statistics(rows: list[dict], stats_config: dict | None = None) -> dict:
    cfg = stats_config or {}
    lengths = [float(x) for x in cfg.get("length_candidates", DEFAULT_LENGTH_CANDIDATES)]

    audio = [r for r in rows if r.get(COL_CATEGORY) == CATEGORY_AUDIO]
    decoded = [r for r in audio if r.get(COL_DECODE_STATUS) == "ok"]
    decode_failed = [r for r in audio if r.get(COL_DECODE_STATUS) == "failed"]
    probed = [r for r in audio if str(r.get(COL_PROBE_STATUS, "")).startswith("ok")]
    probe_failed = [r for r in audio if r.get(COL_PROBE_STATUS) == "failed"]

    counts = {
        "total_discovered": len(rows),
        "audio_candidates": len(audio),
        "probe_ok": len(probed),
        "probe_failed": len(probe_failed),
        "decoded_ok": len(decoded),
        "decode_failed": len(decode_failed),
        "archives": sum(1 for r in rows if r.get(COL_CATEGORY) == CATEGORY_ARCHIVE),
        "sidecar": sum(1 for r in rows if r.get(COL_CATEGORY) == CATEGORY_SIDECAR),
        "unknown": sum(1 for r in rows if r.get(COL_CATEGORY) == CATEGORY_UNKNOWN),
        "extension_mismatch": sum(1 for r in audio if r.get(COL_EXTENSION_MISMATCH) is True),
        "parse_failed": sum(1 for r in audio if r.get(COL_PARSE_STATUS) == "parse_failed"),
    }

    formats = {
        "extensions": _count_field(rows, COL_EXTENSION),
        "containers": _count_field(audio, COL_CONTAINER),
        "encodings": _count_field(audio, COL_ENCODING),
    }

    sample_rates = {
        "decoded": _count_field(decoded, COL_SAMPLE_RATE),
        "header": _count_field([r for r in audio if r.get(COL_SAMPLE_RATE)], COL_SAMPLE_RATE),
    }
    channels = {"decoded": _count_field(decoded, COL_CHANNELS)}

    durations_decoded = [r[COL_DURATION_DECODED_SEC] for r in decoded
                         if r.get(COL_DURATION_DECODED_SEC) is not None]
    durations_header = [r[COL_DURATION_HEADER_SEC] for r in probed
                        if r.get(COL_DURATION_HEADER_SEC) is not None]

    # per-speaker median durations (equal weight per speaker)
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for r in decoded:
        if r.get(COL_SPEAKER_TOKEN) and r.get(COL_DURATION_DECODED_SEC) is not None:
            by_speaker[r[COL_SPEAKER_TOKEN]].append(r[COL_DURATION_DECODED_SEC])
    speaker_medians = [float(np.median(v)) for v in by_speaker.values() if v]

    # groups
    by_directory: dict[str, dict] = {}
    for group in sorted({r.get(COL_DIRECTORY_GROUP) or "" for r in audio}):
        subset = [r for r in audio if (r.get(COL_DIRECTORY_GROUP) or "") == group]
        dec = [r for r in subset if r.get(COL_DECODE_STATUS) == "ok"]
        durs = [r[COL_DURATION_DECODED_SEC] for r in dec if r.get(COL_DURATION_DECODED_SEC) is not None]
        by_directory[group or "(none)"] = {
            "n_files": len(subset),
            "n_decoded_ok": len(dec),
            "n_decode_failed": len(subset) - len(dec),
            "n_speakers": len({r.get(COL_SPEAKER_TOKEN) for r in subset if r.get(COL_SPEAKER_TOKEN)}),
            "duration_sec": _percentiles(durs),
        }

    mic_table: dict[str, dict] = {}
    for group in sorted({(r.get(COL_DIRECTORY_GROUP) or "", r.get(COL_MIC_TOKEN) or "") for r in audio}):
        g, mic = group
        subset = [r for r in audio
                  if (r.get(COL_DIRECTORY_GROUP) or "") == g and (r.get(COL_MIC_TOKEN) or "") == mic]
        mic_table[f"{g}|{mic}"] = {
            "n_files": len(subset),
            "n_probe_ok": sum(1 for r in subset if str(r.get(COL_PROBE_STATUS, "")).startswith("ok")),
        }

    speakers_table = {
        spk: {
            "directory_group": next((r.get(COL_DIRECTORY_GROUP) for r in audio
                                     if r.get(COL_SPEAKER_TOKEN) == spk), ""),
            "n_files": len(v),
            "duration_sec_median": float(np.median(v)) if v else None,
        }
        for spk, v in sorted(by_speaker.items())
    }

    # duration summaries by prompt class and by mic (doc §4.2 分组摘要)
    def _grouped_duration(key_fn) -> dict:
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in decoded:
            if r.get(COL_DURATION_DECODED_SEC) is None:
                continue
            k = key_fn(r)
            if k:
                buckets[k].append(r[COL_DURATION_DECODED_SEC])
        return {k: _percentiles(v) for k, v in sorted(buckets.items())}

    def prompt_class_of(token: str | None) -> str:
        tok = token or ""
        if tok.startswith("CW"):
            return "check_word_CW"
        if tok.startswith("UW"):
            return "unusual_word_UW"
        if tok.startswith("L"):
            return "long_token_L"      # LA..LZ, semantics to verify
        if tok.startswith("D"):
            return "digit_D"
        if tok.startswith("C"):
            return "digit_C"
        return "other" if tok else ""

    duration_by_prompt_class = _grouped_duration(
        lambda r: prompt_class_of(r.get(COL_PROMPT_TOKEN)))
    duration_by_mic = _grouped_duration(lambda r: r.get(COL_MIC_TOKEN) or "")

    # quality summaries (technical facts only)
    quality = {
        "peak": _percentiles([r[COL_QUALITY_PEAK] for r in decoded if r.get(COL_QUALITY_PEAK) is not None]),
        "rms": _percentiles([r[COL_QUALITY_RMS] for r in decoded if r.get(COL_QUALITY_RMS) is not None]),
        "near_full_ratio": _percentiles([r[COL_QUALITY_NEAR_FULL_RATIO] for r in decoded
                                         if r.get(COL_QUALITY_NEAR_FULL_RATIO) is not None]),
        "n_all_zero": sum(1 for r in decoded if r.get(COL_ALL_ZERO) is True),
        "n_nonfinite_samples_files": sum(1 for r in decoded if (r.get(COL_QUALITY_NONFINITE) or 0) > 0),
        "n_extremely_short": sum(1 for r in decoded
                                 if (r.get(COL_DURATION_DECODED_SEC) or 0) < cfg.get("short_duration_sec", 0.1)),
    }

    anomalies = {
        "probe_failed_examples": [r[COL_RELATIVE_PATH] for r in probe_failed[:20]],
        "decode_failed_reasons": _count_field(decode_failed, "decode_error"),
        "probe_warnings": _count_field([r for r in audio if r.get("probe_warnings")], "probe_warnings"),
    }

    length_candidates = _length_candidates(durations_decoded, lengths)
    durations_by_group: dict[str, list[float]] = defaultdict(list)
    for r in decoded:
        if r.get(COL_DURATION_DECODED_SEC) is not None:
            durations_by_group[r.get(COL_DIRECTORY_GROUP) or "(none)"].append(r[COL_DURATION_DECODED_SEC])
    length_candidates_grouped = _length_candidates_grouped(durations_by_group, lengths)

    # prompt token classes (naming facts; semantics pending review)
    prompt_classes = Counter()
    for r in audio:
        cls = prompt_class_of(r.get(COL_PROMPT_TOKEN))
        if cls:
            prompt_classes[cls] += 1

    return {
        "counts": counts,
        "formats": formats,
        "sample_rates": sample_rates,
        "channels": channels,
        "duration_seconds": {
            "decoded_full": _percentiles(durations_decoded),
            "header_estimate_quick": _percentiles(durations_header),
            "quantile_method": DURATION_QUANTILE_METHOD,
            "weighting": "equal per recording",
            "per_speaker_median": _percentiles(speaker_medians),
        },
        "groups": {
            "by_directory_group": by_directory,
            "by_directory_and_mic": mic_table,
            "speakers": speakers_table,
            "prompt_classes": dict(prompt_classes),
            "duration_by_prompt_class": duration_by_prompt_class,
            "duration_by_mic": duration_by_mic,
        },
        "quality": quality,
        "anomalies": anomalies,
        "length_candidates": length_candidates,
        "length_candidates_by_group": length_candidates_grouped,
        "length_candidate_lengths_sec": lengths,
    }
