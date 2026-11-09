"""Per-file audit: probe + (full mode) decode, hash, quality summary.

Runs after scan + adapter enrichment. Workers are stateless (path -> facts);
the orchestrator appends results to progress.jsonl so an interrupted scan can
resume by (relative_path, size_bytes, mtime_ns). Quick mode probes only; full
mode decodes every candidate and computes real sample counts.

Quality metrics are TECHNICAL facts (peak/RMS/DC/near-full-scale/non-finite);
they never judge disease, quality scores, or automatically exclude samples
(data-layer doc §4.3).
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .decoders import DecodeError, decode, extension_matches, probe
from .schema import (
    CATEGORY_AUDIO,
    COL_ALL_ZERO,
    COL_CATEGORY,
    COL_DECODE_BACKEND,
    COL_DECODE_STATUS,
    COL_DURATION_DECODED_SEC,
    COL_DURATION_HEADER_SEC,
    COL_EXTENSION_MISMATCH,
    COL_FRAMES,
    COL_MTIME_NS,
    COL_N_SAMPLES,
    COL_PROBE_STATUS,
    COL_QUALITY_DC,
    COL_QUALITY_NEAR_FULL_RATIO,
    COL_QUALITY_NONFINITE,
    COL_QUALITY_PEAK,
    COL_QUALITY_PER_CHANNEL,
    COL_QUALITY_RMS,
    COL_RELATIVE_PATH,
    COL_SAMPLE_RATE,
    COL_CHANNELS,
    COL_SHA256,
    COL_SIZE_BYTES,
    COL_CONTAINER,
    COL_ENCODING,
    COL_AUDIO_CODEC,
)


def audit_config_defaults() -> dict:
    return {
        "near_full_threshold": 0.99,   # |x| > t counted; float overshoot != ADC clipping
        "short_duration_sec": 0.1,     # reported as extremely short, never auto-excluded
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_one(root_str: str, rel: str, mode: str, quality_cfg: dict) -> dict:
    """Worker: probe + optional decode for one file. Returns audit columns."""
    root = Path(root_str)
    path = root / rel
    out: dict = {
        COL_PROBE_STATUS: "failed", COL_CONTAINER: None, COL_ENCODING: None,
        COL_AUDIO_CODEC: None, COL_SAMPLE_RATE: None, COL_CHANNELS: None,
        COL_FRAMES: None, COL_DURATION_HEADER_SEC: None,
        COL_EXTENSION_MISMATCH: None,
        COL_DECODE_STATUS: "not_attempted", COL_DECODE_BACKEND: None,
        COL_N_SAMPLES: None, COL_DURATION_DECODED_SEC: None, COL_SHA256: None,
        COL_QUALITY_PEAK: None, COL_QUALITY_RMS: None, COL_QUALITY_DC: None,
        COL_QUALITY_NEAR_FULL_RATIO: None, COL_QUALITY_NONFINITE: None,
        COL_ALL_ZERO: None, COL_QUALITY_PER_CHANNEL: None,
    }
    info = probe(path)
    out[COL_PROBE_STATUS] = info.status
    out[COL_CONTAINER] = info.container
    out[COL_ENCODING] = info.encoding
    out[COL_AUDIO_CODEC] = info.audio_codec
    out[COL_SAMPLE_RATE] = info.sample_rate
    out[COL_CHANNELS] = info.channels
    out[COL_FRAMES] = info.frames
    out[COL_DURATION_HEADER_SEC] = info.duration_sec
    ext = os.path.splitext(rel)[1].lower()
    out[COL_EXTENSION_MISMATCH] = (
        (extension_matches(ext, info.container) is False) if info.container else None
    )
    if info.warnings:
        out["probe_warnings"] = "; ".join(info.warnings)[:500]
    if mode == "quick" or info.status != "ok_soundfile":
        # full mode still decodes ffprobe-only files? No: decode needs a
        # libsndfile-compatible stream; ffprobe-ok but sf-failed files are
        # reported as decode_failed candidates for the fallback decoder.
        if mode == "full" and info.status == "failed":
            out[COL_DECODE_STATUS] = "failed"
            out["decode_error"] = "probe failed; no decoder output"
        return out

    if mode == "full":
        try:
            dec = decode(path)
        except DecodeError as e:
            out[COL_DECODE_STATUS] = "failed"
            out["decode_error"] = str(e)[:500]
            return out
        wave = dec.waveform  # [C, N] float32
        n = wave.shape[1]
        out[COL_DECODE_STATUS] = "ok"
        out[COL_DECODE_BACKEND] = dec.backend
        out[COL_N_SAMPLES] = int(n)
        out[COL_SAMPLE_RATE] = int(dec.sample_rate)
        out[COL_DURATION_DECODED_SEC] = float(n / dec.sample_rate) if dec.sample_rate else None
        out[COL_SHA256] = _sha256(path)
        finite = np.isfinite(wave)
        nonfinite = int(np.count_nonzero(~finite))
        safe = np.where(finite, wave, 0.0)
        all_zero = bool(np.all(safe == 0.0)) if n > 0 else None
        out[COL_ALL_ZERO] = all_zero
        out[COL_QUALITY_NONFINITE] = nonfinite
        if n > 0:
            peak_all = float(np.max(np.abs(safe)))
            rms_all = float(np.sqrt(np.mean(np.square(safe, dtype=np.float64))))
            dc_all = float(np.mean(safe, dtype=np.float64))
            near_all = float(np.count_nonzero(np.abs(safe) > quality_cfg["near_full_threshold"]) / safe.size)
            out[COL_QUALITY_PEAK] = peak_all
            out[COL_QUALITY_RMS] = rms_all
            out[COL_QUALITY_DC] = dc_all
            out[COL_QUALITY_NEAR_FULL_RATIO] = near_all
            per_ch = []
            for c in range(wave.shape[0]):
                ch = safe[c]
                per_ch.append({
                    "channel": c,
                    "peak": float(np.max(np.abs(ch))),
                    "rms": float(np.sqrt(np.mean(np.square(ch, dtype=np.float64)))),
                    "dc": float(np.mean(ch, dtype=np.float64)),
                    "near_full_ratio": float(np.count_nonzero(np.abs(ch) > quality_cfg["near_full_threshold"]) / ch.size),
                })
            out[COL_QUALITY_PER_CHANNEL] = json.dumps(per_ch)
        else:
            out[COL_ALL_ZERO] = None
    return out


def resume_key(row: dict) -> tuple:
    return (row[COL_RELATIVE_PATH], row.get(COL_SIZE_BYTES), row.get(COL_MTIME_NS))


def run_audit(root: Path, rows: list[dict], mode: str, scan_dir: Path,
              workers: int, quality_cfg: dict | None = None) -> tuple[list[dict], bool]:
    """Enrich audio-candidate rows with audit facts.

    Returns (enriched_rows, completed_flag). progress.jsonl keeps raw results
    for resumability; an interrupted run leaves status partial.
    """
    quality_cfg = {**audit_config_defaults(), **(quality_cfg or {})}
    progress_path = scan_dir / "progress.jsonl"
    done: dict[tuple, dict] = {}
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[(rec[COL_RELATIVE_PATH], rec.get("size_bytes"), rec.get("mtime_ns"))] = rec

    candidates = [r for r in rows if r[COL_CATEGORY] == CATEGORY_AUDIO]
    todo = [r for r in candidates if resume_key(r) not in done]
    new_results: list[dict] = []
    interrupted = False

    if todo:
        root_str = str(root)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(audit_one, root_str, r[COL_RELATIVE_PATH], mode, quality_cfg): r
                for r in todo
            }
            try:
                with progress_path.open("a", encoding="utf-8") as pf:
                    for fut in as_completed(futures):
                        base = futures[fut]
                        try:
                            res = fut.result()
                        except Exception as e:  # worker crash: record, keep going
                            res = {COL_DECODE_STATUS: "failed",
                                   COL_PROBE_STATUS: "failed",
                                   "decode_error": f"worker exception: {type(e).__name__}: {e}"}
                        rec = {COL_RELATIVE_PATH: base[COL_RELATIVE_PATH],
                               "size_bytes": base.get(COL_SIZE_BYTES),
                               "mtime_ns": base.get(COL_MTIME_NS), **res}
                        done[resume_key(base)] = rec
                        new_results.append(rec)
                        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except KeyboardInterrupt:
                interrupted = True
                pool.shutdown(wait=False, cancel_futures=True)
                # fall through: partial results remain in progress.jsonl

    # merge audit facts into full row copies (preserving scan+adapter columns)
    by_key = {resume_key(r): r for r in candidates}
    enriched = []
    for r in rows:
        out_row = dict(r)
        if r[COL_CATEGORY] == CATEGORY_AUDIO:
            rec = done.get(resume_key(r))
            if rec:
                out_row.update({k: v for k, v in rec.items()
                                if k not in (COL_RELATIVE_PATH, "size_bytes", "mtime_ns")})
            else:
                out_row[COL_PROBE_STATUS] = "not_attempted"
                out_row[COL_DECODE_STATUS] = "not_attempted"
        enriched.append(out_row)
    return enriched, not interrupted
