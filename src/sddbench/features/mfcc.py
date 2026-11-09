"""Fixed MFCC recipe and 40-dim (mean,+std) statistics.

One recipe, no dataset branches. The implementation is torchaudio's MFCC with
`log_mels=True` so the matrix is log(mel_power + log_offset) and the default
dB scaling (which depends on the input maximum) is avoided (data-layer doc §9).
All resolved values are recorded; the feature hash covers the formula, the
implementation versions and the actual sample rate geometry.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from ..data.audio import frontend_geometry

FEATURE_IMPLEMENTATION = "torchaudio.transforms.MFCC"
FEATURE_IMPLEMENTATION_VERSION = "1.0"
LOG_OFFSET = 1e-6  # checked against torchaudio.transforms.MFCC.forward
LOG_APPLIED_TO = "mel_power"

FEATURE_DEFAULTS = {
    "schema_version": "1.0",
    "name": "mfcc_stats",
    "implementation": "torchaudio",
    "n_mfcc": 20,
    "n_mels": 64,
    "frame_ms": 25.0,
    "hop_ms": 10.0,
    "n_fft_policy": "next_power_of_two_window",
    "window": "hann_periodic",
    "center": False,
    "power": 2.0,
    "f_min_hz": 0.0,
    "f_max_policy": "nyquist",
    "f_max_hz": None,
    "mel_scale": "htk",
    "mel_norm": None,
    "log_mels": True,
    "log_implementation": "torchaudio_native_mels",
    "log_epsilon": LOG_OFFSET,
    "dct_type": 2,
    "dct_norm": "ortho",
    "include_c0": True,
    "stats": ["mean", "std"],
    "std_ddof": 0,
    "save_frames": False,
}

_TRANSFORM_CACHE: dict[tuple, object] = {}


class FeatureConfigError(ValueError):
    """Raised when a feature config cannot be applied exactly as written."""


class FeatureError(RuntimeError):
    """Raised when MFCC cannot be computed for a valid view."""


@dataclass
class MfccResult:
    vector: np.ndarray          # float32 [2*n_mfcc]
    n_frames: int
    n_mfcc: int
    sample_rate: int
    frame_geometry: dict
    frames: np.ndarray | None = None  # present only when save_frames=True


def _close(a, b, tol=1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def normalize_feature_config(raw: dict) -> dict:
    cfg = {**FEATURE_DEFAULTS, **dict(raw or {})}
    if cfg["implementation"] != "torchaudio":
        raise FeatureConfigError(
            f"implementation must be torchaudio, got {cfg['implementation']!r}")
    if cfg["n_fft_policy"] != "next_power_of_two_window":
        raise FeatureConfigError(
            f"n_fft_policy must be next_power_of_two_window, got {cfg['n_fft_policy']!r}")
    if cfg["window"] != "hann_periodic":
        raise FeatureConfigError(f"window must be hann_periodic, got {cfg['window']!r}")
    if cfg["center"] is not False:
        raise FeatureConfigError("center must be false (valid audio only, no fake edges)")
    if not isinstance(cfg["n_mfcc"], int) or cfg["n_mfcc"] <= 0:
        raise FeatureConfigError(f"n_mfcc must be a positive integer, got {cfg['n_mfcc']!r}")
    if not isinstance(cfg["n_mels"], int) or cfg["n_mels"] <= 0:
        raise FeatureConfigError(f"n_mels must be a positive integer, got {cfg['n_mels']!r}")
    if cfg["n_mfcc"] > cfg["n_mels"]:
        raise FeatureConfigError(
            f"n_mfcc ({cfg['n_mfcc']}) cannot exceed n_mels ({cfg['n_mels']})")
    for key in ("frame_ms", "hop_ms"):
        v = cfg[key]
        if not isinstance(v, (int, float)) or v <= 0:
            raise FeatureConfigError(f"{key} must be a positive number, got {v!r}")
    if not isinstance(cfg["power"], (int, float)) or cfg["power"] <= 0:
        raise FeatureConfigError(f"power must be positive, got {cfg['power']!r}")
    if not isinstance(cfg["f_min_hz"], (int, float)) or cfg["f_min_hz"] < 0:
        raise FeatureConfigError(f"f_min_hz must be non-negative, got {cfg['f_min_hz']!r}")
    if cfg["f_max_policy"] not in ("nyquist", "fixed"):
        raise FeatureConfigError(
            f"f_max_policy must be nyquist|fixed, got {cfg['f_max_policy']!r}")
    if cfg["f_max_policy"] == "fixed":
        if not isinstance(cfg["f_max_hz"], (int, float)) or cfg["f_max_hz"] <= 0:
            raise FeatureConfigError("f_max_policy=fixed requires positive f_max_hz")
    else:
        cfg["f_max_hz"] = None
    if cfg["mel_scale"] != "htk":
        raise FeatureConfigError(f"mel_scale must be htk, got {cfg['mel_scale']!r}")
    if cfg["mel_norm"] is not None:
        raise FeatureConfigError("mel_norm must be null in this recipe")
    if cfg["log_mels"] is not True:
        raise FeatureConfigError("log_mels must be true to avoid input-dependent dB scaling")
    if cfg["log_implementation"] != "torchaudio_native_mels":
        raise FeatureConfigError(
            f"log_implementation must be torchaudio_native_mels, got {cfg['log_implementation']!r}")
    if not _close(cfg["log_epsilon"], LOG_OFFSET):
        raise FeatureConfigError(
            f"log_epsilon must equal the locked implementation value {LOG_OFFSET}, "
            f"got {cfg['log_epsilon']!r}")
    if cfg["dct_type"] != 2 or cfg["dct_norm"] != "ortho":
        raise FeatureConfigError("dct_type=2 and dct_norm=ortho are required")
    if cfg["include_c0"] is not True:
        raise FeatureConfigError("include_c0 must be true (torchaudio keeps c0)")
    if list(cfg["stats"]) != ["mean", "std"]:
        raise FeatureConfigError("stats must be exactly [mean, std]")
    if not isinstance(cfg["std_ddof"], int) or cfg["std_ddof"] < 0:
        raise FeatureConfigError(f"std_ddof must be a non-negative integer, got {cfg['std_ddof']!r}")
    if cfg["save_frames"] not in (True, False):
        raise FeatureConfigError("save_frames must be true or false")
    if cfg["save_frames"]:
        raise FeatureConfigError(
            "save_frames=true is not implemented in this stage; refusing to silently drop frames")
    cfg["schema_version"] = str(cfg["schema_version"])
    return cfg


def implementation_info() -> dict:
    """Versions and the exact log/DCT semantics this recipe relies on."""
    try:
        import torch
        import torchaudio
        import torchaudio.functional as _f  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on environment
        raise FeatureConfigError(
            "torch/torchaudio are required for MFCC; install with `pip install -e .[features]`. "
            f"Import error: {e}") from e
    return {
        "name": FEATURE_IMPLEMENTATION,
        "version": FEATURE_IMPLEMENTATION_VERSION,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "log_mels": True,
        "log_offset": LOG_OFFSET,
        "log_applied_to": LOG_APPLIED_TO,
        "dct": "dct_type=2, norm=ortho, include_c0=true",
        "window_fn": "torch.hann_window(periodic=True)",
        "mel_basis_normalized": False,
    }


def resolve_f_max(cfg: dict, sample_rate: int) -> float:
    nyquist = sample_rate / 2.0
    if cfg["f_max_policy"] == "nyquist":
        return float(nyquist)
    f_max = float(cfg["f_max_hz"])
    if f_max > nyquist:
        raise FeatureConfigError(
            f"fixed f_max_hz={f_max} exceeds Nyquist={nyquist} at {sample_rate} Hz")
    return f_max


def feature_hash(cfg: dict, sample_rate: int, impl: dict | None = None) -> str:
    """Hash of formula + implementation + actual-Hz geometry (not the source file)."""
    impl = impl or implementation_info()
    geometry = frontend_geometry(sample_rate, float(cfg["frame_ms"]), float(cfg["hop_ms"]))
    payload = {
        "config": {k: v for k, v in cfg.items() if k not in ("name", "description")},
        "implementation": impl,
        "sample_rate": int(sample_rate),
        "geometry": geometry.as_dict(),
        "f_max_hz_resolved": resolve_f_max(cfg, sample_rate),
        "center": False,
        "pad": 0,
        "stats": list(cfg["stats"]),
        "std_ddof": int(cfg["std_ddof"]),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def feature_key(cfg: dict, sample_rate: int) -> str:
    return f"{cfg['name']}_{feature_hash(cfg, sample_rate)[:16]}"


def _build_transform(sample_rate: int, cfg: dict, f_hash: str):
    key = (int(sample_rate), f_hash)
    cached = _TRANSFORM_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import torch
        import torchaudio.transforms as transforms
    except ImportError as e:  # pragma: no cover
        raise FeatureConfigError(
            "torch/torchaudio are required for MFCC; install with `pip install -e .[features]`"
        ) from e
    geometry = frontend_geometry(sample_rate, float(cfg["frame_ms"]), float(cfg["hop_ms"]))
    melkwargs = {
        "n_fft": geometry.n_fft,
        "win_length": geometry.win_length,
        "hop_length": geometry.hop_length,
        "f_min": float(cfg["f_min_hz"]),
        "f_max": resolve_f_max(cfg, sample_rate),
        "pad": 0,
        "n_mels": int(cfg["n_mels"]),
        "window_fn": torch.hann_window,
        "power": float(cfg["power"]),
        "normalized": False,
        "center": False,
        "pad_mode": "reflect",
        "norm": None,
        "mel_scale": "htk",
    }
    transform = transforms.MFCC(
        sample_rate=int(sample_rate),
        n_mfcc=int(cfg["n_mfcc"]),
        dct_type=2,
        norm="ortho",
        log_mels=True,
        melkwargs=melkwargs,
    )
    _TRANSFORM_CACHE[key] = transform
    return transform


def extract_mfcc(waveform: np.ndarray, sample_rate: int, cfg: dict,
                 impl: dict | None = None) -> MfccResult:
    """Compute MFCC stats on VALID audio only (never on padding)."""
    import torch

    impl = impl or implementation_info()
    wave = np.ascontiguousarray(waveform, dtype=np.float32).reshape(-1)
    geometry = frontend_geometry(sample_rate, float(cfg["frame_ms"]), float(cfg["hop_ms"]))
    if wave.size < geometry.min_valid_samples:
        raise FeatureError(
            f"too_short_for_frontend: {wave.size} samples < n_fft={geometry.n_fft} "
            f"at {sample_rate} Hz")
    if not np.isfinite(wave).all():
        raise FeatureError("valid audio contains non-finite samples")

    f_hash = feature_hash(cfg, sample_rate, impl)
    transform = _build_transform(sample_rate, cfg, f_hash)
    with torch.no_grad():
        mfcc = transform(torch.from_numpy(wave).unsqueeze(0))
    values = mfcc.squeeze(0).numpy()
    n_mfcc, n_frames = int(values.shape[0]), int(values.shape[1])
    expected_frames = 1 + (wave.size - geometry.n_fft) // geometry.hop_length
    if n_frames != expected_frames:
        raise FeatureError(
            f"frame count mismatch: implementation gave {n_frames}, "
            f"locked formula expects {expected_frames}")
    if n_frames < 1:
        raise FeatureError("MFCC produced no frames")
    if not np.isfinite(values).all():
        raise FeatureError("MFCC contains non-finite values")

    mean = values.mean(axis=1, dtype=np.float64)
    std = values.std(axis=1, ddof=int(cfg["std_ddof"]), dtype=np.float64)
    vector = np.concatenate([mean, std]).astype(np.float32)
    if vector.shape[0] != 2 * int(cfg["n_mfcc"]) or not np.isfinite(vector).all():
        raise FeatureError(f"invalid MFCC statistics vector: {vector.shape}")
    return MfccResult(
        vector=vector, n_frames=n_frames, n_mfcc=n_mfcc,
        sample_rate=int(sample_rate), frame_geometry=geometry.as_dict(),
    )
