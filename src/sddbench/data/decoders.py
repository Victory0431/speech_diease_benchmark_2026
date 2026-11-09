"""Audio decoding: probe (header facts) and decode (float32 samples).

Backend policy (data-layer doc §6): soundfile/libsndfile first; ffprobe/ffmpeg
fallback with explicit argument arrays (no shell string拼接), timeout, local
files only. `probe` never changes audio; `decode` returns the REAL sample rate
(never an assumed 16000), no resampling, no channel folding, no normalization.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

FFPROBE_TIMEOUT_SEC = 30


class DecodeError(RuntimeError):
    """Raised when a file cannot be decoded into samples."""


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file (read-only, streaming)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class AudioInfo:
    status: str  # ok_soundfile | ok_ffprobe | failed
    container: str | None = None      # actual container (sf format / ffprobe format_name)
    encoding: str | None = None       # sample encoding/subtype (e.g. PCM_16); null for lossy
    audio_codec: str | None = None    # codec name (ffprobe); null when unknown
    sample_rate: int | None = None
    channels: int | None = None
    frames: int | None = None
    duration_sec: float | None = None
    backend: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class DecodedAudio:
    waveform: np.ndarray  # float32 [channels, samples]
    sample_rate: int      # the file's REAL sample rate
    backend: str
    warnings: list[str] = field(default_factory=list)


def tool_versions() -> dict:
    ffprobe_version = None
    try:
        out = subprocess.run(
            ["ffprobe", "-version"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            ffprobe_version = out.stdout.splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "ffprobe": ffprobe_version,
    }


def _probe_ffprobe(path: Path) -> AudioInfo:
    """ffprobe fallback: machine-readable stream/format info, no decoding."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration,nb_frames:format=format_name",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SEC)
    except FileNotFoundError:
        return AudioInfo(status="failed", backend="ffprobe",
                         warnings=["ffprobe executable not found"])
    except subprocess.TimeoutExpired:
        return AudioInfo(status="failed", backend="ffprobe",
                         warnings=[f"ffprobe timeout after {FFPROBE_TIMEOUT_SEC}s"])
    if proc.returncode != 0:
        return AudioInfo(status="failed", backend="ffprobe",
                         warnings=[f"ffprobe rc={proc.returncode}: {proc.stderr.strip()[:300]}"])
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return AudioInfo(status="failed", backend="ffprobe", warnings=["ffprobe output not JSON"])
    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}
    warnings = []
    if not streams:
        warnings.append("no audio stream found")
        return AudioInfo(status="failed", backend="ffprobe", container=fmt.get("format_name"),
                         warnings=warnings)
    s = streams[0]
    sr = int(s["sample_rate"]) if s.get("sample_rate") else None
    frames = int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None
    duration = None
    if s.get("duration") is not None:
        try:
            duration = float(s["duration"])
        except ValueError:
            pass
    if duration is None and fmt.get("duration") is not None:
        try:
            duration = float(fmt["duration"])
            warnings.append("duration from container (stream duration missing)")
        except ValueError:
            pass
    return AudioInfo(
        status="ok_ffprobe",
        container=fmt.get("format_name"),
        encoding=None,  # lossy/compressed: no meaningful PCM bit depth; do not invent one
        audio_codec=s.get("codec_name"),
        sample_rate=sr,
        channels=int(s["channels"]) if s.get("channels") else None,
        frames=frames,
        duration_sec=duration,
        backend="ffprobe",
        warnings=warnings,
    )


def probe(path: Path) -> AudioInfo:
    """Header-level probe without decoding samples."""
    try:
        info = sf.info(str(path))
    except Exception as e:  # soundfile raises RuntimeError for unknown/unsupported
        fb = _probe_ffprobe(path)
        if fb.status != "ok_ffprobe":
            fb.warnings.append(f"soundfile: {type(e).__name__}: {e}")
            fb.status = "failed"
        return fb
    return AudioInfo(
        status="ok_soundfile",
        container=info.format,
        encoding=info.subtype,
        audio_codec=None,
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        duration_sec=(info.frames / info.samplerate) if info.samplerate else None,
        backend=f"soundfile({sf.__libsndfile_version__})",
    )


def decode(path: Path) -> DecodedAudio:
    """Decode to float32 [channels, samples] at the file's REAL sample rate."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        raise DecodeError(f"{type(e).__name__}: {e}") from e
    if sr is None or sr <= 0:
        raise DecodeError(f"invalid sample rate from decoder: {sr!r}")
    wave = np.ascontiguousarray(data.T)  # [C, N]
    return DecodedAudio(waveform=wave, sample_rate=int(sr),
                        backend=f"soundfile({sf.__libsndfile_version__})")


def extension_matches(ext: str, container: str | None) -> bool | None:
    """True/False when comparable, None when no declared container is known."""
    if container is None:
        return None
    c = container.upper()
    table = {
        ".wav": ("WAV", "WAVEX", "RF64", "W64"),
        ".wave": ("WAV", "WAVEX"),
        ".flac": ("FLAC",),
        ".aiff": ("AIFF",), ".aif": ("AIFF",), ".aifc": ("AIFF",),
        ".ogg": ("OGG",), ".oga": ("OGG",),
        ".au": ("AU",), ".snd": ("AU",),
    }
    expected = table.get(ext.lower())
    if expected is None:
        return None  # extension not in our static table; no claim
    return c in expected
