"""Window/view index: deterministic crop coordinates, padding, and a
self-describing window table (data-layer doc §8).

Definitions (all in ORIGINAL sample coordinates):
  N = decoded samples, s = original sample rate
  K = round_half_up(window_seconds * s)          (target window in samples)
  D = max(N - K, 0)
  head: start=0 | center: start=floor(D/2) | tail: start=D
  fixed_random: start drawn once from integers [0, D] with a PCG64 RNG whose
  64-bit seed is derived from SHA256(algorithm_version|global_seed|sample_id).
  end = min(start + K, N); when N < K every position keeps the whole recording
  and right zero pads; padding never pretends to be recorded silence.

Views are built from a saved D1 scan (files.csv): the source hash is re-checked
so a changed original cannot silently enter a feature job.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import (
    FrontendGeometry,
    RESAMPLE_IMPLEMENTATION,
    RESAMPLE_METHOD,
    RESAMPLE_PARAMS,
    RESAMPLE_VERSION,
    ViewError,
    crop,
    frontend_geometry,
    pad_right,
    resample,
    sample_count,
    select_channel,
)
from .decoders import DecodeError, decode, file_sha256
from .schema import (
    CATEGORY_AUDIO,
    COL_CATEGORY,
    COL_DECODE_STATUS,
    COL_DIRECTORY_GROUP,
    COL_DURATION_DECODED_SEC,
    COL_MIC_TOKEN,
    COL_N_SAMPLES,
    COL_PARSE_STATUS,
    COL_PROMPT_TOKEN,
    COL_RELATIVE_PATH,
    COL_REVIEW_STATUS,
    COL_SAMPLE_RATE,
    COL_SHA256,
    COL_SPEAKER_TOKEN,
)

CROP_ALGORITHM_VERSION = "1.0"
POSITIONS = ("head", "center", "tail", "fixed_random")
DEFAULT_FRAME_MS = 25.0
DEFAULT_HOP_MS = 10.0

WINDOW_COLUMNS = [
    "sample_id", "window_id", "relative_path", "source_scan_id",
    "directory_group", "speaker_token", "prompt_token", "mic_token",
    "review_status", "parse_status",
    "source_sha256", "decoder_backend",
    "original_sample_rate", "original_n_samples",
    "original_start_sample", "original_end_sample",
    "original_start_sec", "original_end_sec", "source_duration_sec",
    "processed_sample_rate", "valid_num_samples", "target_num_samples",
    "pad_num_samples", "padding_ratio", "truncated_num_samples",
    "position", "seed", "rng_seed_used", "crop_algorithm_version",
    "channel_policy", "channel_index", "n_channels",
    "resample_applied", "resample_info",
    "win_length", "hop_length", "n_fft", "frame_ms", "hop_ms",
    "view_hash", "preview_path",
]

PREPROCESS_DEFAULTS = {
    "schema_version": "1.0",
    "profile": "technical_only",
    "sample_rate_policy": "native",
    "allow_mixed_native": False,
    "target_sample_rate": None,
    "channel_policy": "select",
    "channel_index": 0,
    "view_mode": "single_segment",
    "observe_seconds": 2.0,
    "window_seconds": None,   # defaults to observe_seconds
    "position": "head",
    "crop_seed": 0,
    "short_policy": "right_zero_pad",
    "amplitude_normalization": "none",
    "frame_ms": DEFAULT_FRAME_MS,
    "hop_ms": DEFAULT_HOP_MS,
    "crop_algorithm_version": CROP_ALGORITHM_VERSION,
}


class PreprocessConfigError(ValueError):
    """Raised when a preprocessing config cannot be applied as written."""


def normalize_preprocess(raw: dict) -> dict:
    """Validate + fill defaults; refuse semantics this stage does not implement."""
    cfg = {**PREPROCESS_DEFAULTS, **dict(raw or {})}
    if cfg["window_seconds"] is None:
        cfg["window_seconds"] = cfg["observe_seconds"]

    policy = cfg["sample_rate_policy"]
    if policy not in ("native", "resample"):
        raise PreprocessConfigError(f"sample_rate_policy must be native|resample, got {policy!r}")
    if policy == "resample":
        tsr = cfg["target_sample_rate"]
        if not isinstance(tsr, int) or tsr <= 0:
            raise PreprocessConfigError(
                f"sample_rate_policy=resample requires a positive integer "
                f"target_sample_rate, got {tsr!r}")
    else:
        cfg["target_sample_rate"] = None
        if cfg["allow_mixed_native"] not in (True, False):
            raise PreprocessConfigError("allow_mixed_native must be true or false")

    if cfg["channel_policy"] not in ("select", "require_mono"):
        raise PreprocessConfigError(
            f"channel_policy must be select|require_mono, got {cfg['channel_policy']!r}")
    if cfg["channel_policy"] == "select" and (
            not isinstance(cfg["channel_index"], int) or cfg["channel_index"] < 0):
        raise PreprocessConfigError(f"invalid channel_index: {cfg['channel_index']!r}")

    if cfg["view_mode"] != "single_segment":
        raise PreprocessConfigError(
            f"only view_mode=single_segment is implemented, got {cfg['view_mode']!r}")
    for key in ("observe_seconds", "window_seconds"):
        v = cfg[key]
        if not isinstance(v, (int, float)) or v <= 0:
            raise PreprocessConfigError(f"{key} must be a positive number, got {v!r}")
    if not math.isclose(float(cfg["observe_seconds"]), float(cfg["window_seconds"]),
                        rel_tol=0.0, abs_tol=1e-9):
        raise PreprocessConfigError(
            "single_segment requires observe_seconds == window_seconds "
            f"({cfg['observe_seconds']} != {cfg['window_seconds']})")
    if cfg["position"] not in POSITIONS:
        raise PreprocessConfigError(f"position must be one of {POSITIONS}, got {cfg['position']!r}")
    if cfg["short_policy"] != "right_zero_pad":
        raise PreprocessConfigError(
            f"only short_policy=right_zero_pad is implemented, got {cfg['short_policy']!r}")
    if cfg["amplitude_normalization"] != "none":
        raise PreprocessConfigError(
            "amplitude_normalization must be none in this stage, got "
            f"{cfg['amplitude_normalization']!r}")
    if not isinstance(cfg["crop_seed"], int):
        raise PreprocessConfigError(f"crop_seed must be an integer, got {cfg['crop_seed']!r}")
    for key in ("frame_ms", "hop_ms"):
        v = cfg[key]
        if not isinstance(v, (int, float)) or v <= 0:
            raise PreprocessConfigError(f"{key} must be a positive number, got {v!r}")
    cfg["profile"] = str(cfg["profile"])
    cfg["crop_algorithm_version"] = str(cfg["crop_algorithm_version"])
    return cfg


def preprocess_fingerprint(cfg: dict) -> str:
    """Stable hash of the preprocessing semantics (not the display name)."""
    payload = {k: v for k, v in cfg.items() if k not in ("name", "description")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def derive_crop_seed(global_seed: int, sample_id: str,
                     algorithm_version: str = CROP_ALGORITHM_VERSION) -> int:
    """64-bit RNG seed from SHA256(version|global_seed|sample_id).

    Independent of enumeration order and of Python's randomized hash().
    """
    material = f"{algorithm_version}|{int(global_seed)}|{sample_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed)))


def window_geometry(n_samples: int, original_sr: int, cfg: dict,
                    sample_id: str) -> dict:
    """Resolve start/end/target in original samples for one recording."""
    k = sample_count(float(cfg["window_seconds"]), original_sr)
    n = int(n_samples)
    d = max(n - k, 0)
    position = cfg["position"]
    rng_seed = None
    if position == "head":
        start = 0
    elif position == "center":
        start = d // 2
    elif position == "tail":
        start = d
    elif position == "fixed_random":
        rng_seed = derive_crop_seed(int(cfg["crop_seed"]), sample_id,
                                    str(cfg["crop_algorithm_version"]))
        start = int(rng_for(rng_seed).integers(0, d + 1)) if d > 0 else 0
    else:  # normalize_preprocess already rejects this
        raise PreprocessConfigError(f"unknown position: {position!r}")
    end = min(start + k, n)
    return {
        "target_num_samples_original": k,
        "window_d_num_samples": d,
        "original_start_sample": int(start),
        "original_end_sample": int(end),
        "original_start_sec": float(start / original_sr),
        "original_end_sec": float(end / original_sr),
        "rng_seed_used": rng_seed,
    }


def compute_view_hash(cfg: dict, sample_id: str, source_sha256: str | None,
                      resolved: dict) -> str:
    payload = {
        "preprocess": {k: v for k, v in cfg.items() if k not in ("name", "description")},
        "sample_id": sample_id,
        "source_sha256": source_sha256,
        "resolved": resolved,
        "resampler": {
            "implementation": RESAMPLE_IMPLEMENTATION,
            "method": RESAMPLE_METHOD,
            "params": RESAMPLE_PARAMS,
            "version": RESAMPLE_VERSION,
        },
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class ViewTable:
    rows: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    sample_rate_inventory: dict = field(default_factory=dict)
    sample_rate_selected: dict = field(default_factory=dict)
    frontend_by_sample_rate: dict = field(default_factory=dict)


def _issue(issue_type: str, severity: str, relative_path: str, detail: str) -> dict:
    return {"issue_type": issue_type, "severity": severity,
            "relative_path": relative_path, "detail": detail}


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _s(value) -> str:
    return "" if _missing(value) else str(value)


def _preview_name(index: int, rel: str) -> str:
    stem = Path(rel).stem.replace("/", "__").replace(" ", "_")
    return f"{index:03d}_{stem}.wav"


def build_view_index(root: Path, records: list[dict], cfg: dict, *,
                     source_scan_id: str,
                     preview_dir: Path | None = None,
                     preview_limit: int = 0,
                     verify_source_hash: bool = True,
                     on_view=None) -> ViewTable:
    """Decode each selected record once and build one window row per sample.

    Only successfully decoded audio candidates produce windows. Files whose
    content hash differs from the D1 scan are refused (not silently reused),
    which couples every view to the statistics it was derived from.

    `on_view(row, valid_waveform, processed_sample_rate)` runs synchronously for
    each built window (used by MFCC extraction) so full runs never keep all
    waveforms in memory and never decode twice.
    """
    table = ViewTable()
    counts = Counter({
        "requested": len(records),
        "built": 0,
        "skipped_not_eligible": 0,
        "decode_failed": 0,
        "hash_mismatch": 0,
        "channel_error": 0,
        "view_error": 0,
        "too_short_for_frontend": 0,
        "previews_written": 0,
    })
    sr_inventory: Counter = Counter()
    sr_selected: Counter = Counter()
    geometry_cache: dict[int, FrontendGeometry] = {}

    for rec in records:
        rel = rec.get(COL_RELATIVE_PATH, "")
        sr_inventory[_s(rec.get(COL_SAMPLE_RATE)) or "missing"] += 1
        if rec.get(COL_CATEGORY) != CATEGORY_AUDIO or rec.get(COL_DECODE_STATUS) != "ok":
            counts["skipped_not_eligible"] += 1
            table.issues.append(_issue(
                "source_not_eligible", "warning", rel,
                "source row is not a successfully decoded audio candidate"))
            continue

        path = root / rel
        try:
            decoded = decode(path)
        except DecodeError as e:
            counts["decode_failed"] += 1
            table.issues.append(_issue("decode_failed", "error", rel, str(e)[:500]))
            continue

        recorded_sha = _s(rec.get(COL_SHA256))
        if verify_source_hash:
            try:
                digest = file_sha256(path)
            except OSError as e:
                counts["decode_failed"] += 1
                table.issues.append(_issue("hash_read_failed", "error", rel, str(e)[:300]))
                continue
            if recorded_sha and digest != recorded_sha:
                counts["hash_mismatch"] += 1
                table.issues.append(_issue(
                    "source_hash_mismatch", "error", rel,
                    f"file content changed since scan {source_scan_id}; refusing this view"))
                continue
            if not recorded_sha:
                table.issues.append(_issue(
                    "source_sha256_missing", "warning", rel,
                    "scan row has no sha256; using the freshly computed hash"))
        else:
            digest = recorded_sha
            table.issues.append(_issue(
                "source_hash_not_verified", "warning", rel,
                "verify_source_hash disabled for this job"))

        original_sr = int(decoded.sample_rate)
        sr_selected[original_sr] += 1
        try:
            wave, ch_info = select_channel(decoded.waveform,
                                           cfg["channel_policy"], cfg["channel_index"])
        except ViewError as e:
            counts["channel_error"] += 1
            table.issues.append(_issue("channel_error", "error", rel, str(e)))
            continue

        n_samples = int(wave.shape[0])
        recorded_n = rec.get(COL_N_SAMPLES)
        if not _missing(recorded_n) and int(recorded_n) != n_samples:
            table.issues.append(_issue(
                "sample_count_mismatch", "warning", rel,
                f"decoded {n_samples} samples but scan recorded {recorded_n}"))

        geom = window_geometry(n_samples, original_sr, cfg, rel)
        start, end = geom["original_start_sample"], geom["original_end_sample"]
        try:
            valid = crop(wave, start, end)
        except ViewError as e:
            counts["view_error"] += 1
            table.issues.append(_issue("crop_error", "error", rel, str(e)))
            continue

        target_sr = (int(cfg["target_sample_rate"])
                     if cfg["sample_rate_policy"] == "resample" else original_sr)
        try:
            valid, resample_info = resample(valid, original_sr, target_sr)
        except ViewError as e:
            counts["view_error"] += 1
            table.issues.append(_issue("resample_error", "error", rel, str(e)))
            continue

        k_target = sample_count(float(cfg["window_seconds"]), target_sr)
        valid_n = int(valid.shape[0])
        if valid_n > k_target:
            # only possible via resampling rounding; make it explicit
            table.issues.append(_issue(
                "valid_longer_than_target", "warning", rel,
                f"valid {valid_n} > target {k_target}; trimmed after resampling"))
            valid = valid[:k_target]
            valid_n = k_target
        pad_n = k_target - valid_n

        geom_fe = geometry_cache.get(target_sr)
        if geom_fe is None:
            geom_fe = frontend_geometry(target_sr, float(cfg["frame_ms"]), float(cfg["hop_ms"]))
            geometry_cache[target_sr] = geom_fe

        if valid_n < geom_fe.min_valid_samples:
            counts["too_short_for_frontend"] += 1
            table.issues.append(_issue(
                "too_short_for_frontend", "warning", rel,
                f"valid {valid_n} samples < min {geom_fe.min_valid_samples} "
                f"(n_fft at {target_sr} Hz)"))
            continue

        sample_id = rel
        resolved = {
            **geom,
            "processed_sample_rate": target_sr,
            "valid_num_samples": valid_n,
            "target_num_samples": k_target,
            "pad_num_samples": pad_n,
            "channel": ch_info,
            "resample": resample_info,
            "frontend": geom_fe.as_dict(),
        }
        view_hash = compute_view_hash(cfg, sample_id, digest, resolved)

        preview_rel = None
        if preview_dir is not None and counts["previews_written"] < preview_limit:
            preview_dir.mkdir(parents=True, exist_ok=True)
            name = _preview_name(counts["previews_written"], rel)
            sf.write(str(preview_dir / name), valid, target_sr, subtype="PCM_16")
            preview_rel = str(Path(preview_dir.name) / name)
            counts["previews_written"] += 1

        row = {
            "sample_id": sample_id,
            # include scan + view so a different crop can never hit a stale cache row
            "window_id": f"{source_scan_id}|{sample_id}|w0|{view_hash[:12]}",
            "relative_path": rel,
            "source_scan_id": source_scan_id,
            "directory_group": _s(rec.get(COL_DIRECTORY_GROUP)),
            "speaker_token": _s(rec.get(COL_SPEAKER_TOKEN)),
            "prompt_token": _s(rec.get(COL_PROMPT_TOKEN)),
            "mic_token": _s(rec.get(COL_MIC_TOKEN)),
            "review_status": _s(rec.get(COL_REVIEW_STATUS)) or "pending",
            "parse_status": _s(rec.get(COL_PARSE_STATUS)),
            "source_sha256": digest,
            "decoder_backend": decoded.backend,
            "original_sample_rate": original_sr,
            "original_n_samples": n_samples,
            **{k: geom[k] for k in (
                "original_start_sample", "original_end_sample",
                "original_start_sec", "original_end_sec", "rng_seed_used")},
            "source_duration_sec": (float(rec.get(COL_DURATION_DECODED_SEC))
                                    if not _missing(rec.get(COL_DURATION_DECODED_SEC))
                                    else float(n_samples / original_sr)),
            "processed_sample_rate": target_sr,
            "valid_num_samples": valid_n,
            "target_num_samples": k_target,
            "pad_num_samples": pad_n,
            "padding_ratio": float(pad_n / k_target),
            "truncated_num_samples": max(n_samples - end, 0),
            "position": cfg["position"],
            "seed": int(cfg["crop_seed"]),
            "crop_algorithm_version": str(cfg["crop_algorithm_version"]),
            "channel_policy": ch_info["channel_policy"],
            "channel_index": ch_info["channel_index"],
            "n_channels": ch_info["n_channels"],
            "resample_applied": bool(resample_info["applied"]),
            "resample_info": json.dumps(resample_info, sort_keys=True),
            "win_length": geom_fe.win_length,
            "hop_length": geom_fe.hop_length,
            "n_fft": geom_fe.n_fft,
            "frame_ms": geom_fe.frame_ms,
            "hop_ms": geom_fe.hop_ms,
            "view_hash": view_hash,
            "preview_path": preview_rel,
        }
        table.rows.append({c: row.get(c) for c in WINDOW_COLUMNS})
        counts["built"] += 1
        if on_view is not None:
            on_view(row, valid, target_sr)

    table.counts = dict(counts)
    table.sample_rate_inventory = {str(k): v for k, v in sorted(sr_inventory.items(),
                                                                key=lambda kv: str(kv[0]))}
    table.sample_rate_selected = {str(k): v for k, v in sorted(sr_selected.items(),
                                                               key=lambda kv: str(kv[0]))}
    table.frontend_by_sample_rate = {str(sr): g.as_dict() for sr, g in sorted(geometry_cache.items())}
    return table
