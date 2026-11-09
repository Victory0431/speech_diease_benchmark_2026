"""Generic audio view operations: channel selection, segmentation, resampling.

This module works on decoded float32 arrays with the REAL sample rate and never
writes to the dataset root. It does not know about dataset naming, labels or
MFCC. Every transform returns the resolved parameters that must be stored with
the view (data-layer doc §7-§8):

- channel selection never averages channels (unknown channel semantics is an
  error, not a silent mono mix);
- resampling is a single locked implementation (torchaudio
  `sinc_interp_hann`) whose parameters are recorded and whose output length is
  verified against round_half_up(N * target_sr / source_sr);
- padding is a tensor-container policy (right zero pad) and is recorded
  separately from the valid audio.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Locked resampler identity. Changing any of these changes `view_hash` and
# invalidates derived features.
RESAMPLE_IMPLEMENTATION = "torchaudio.functional.resample"
RESAMPLE_METHOD = "sinc_interp_hann"
RESAMPLE_VERSION = "1.0"
RESAMPLE_PARAMS = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9,
    "beta": None,
}


class ViewError(RuntimeError):
    """Raised when a channel/segment/resample step cannot be applied as asked."""


def round_half_up(x: float) -> int:
    """Round a non-negative value half up (never banker's rounding)."""
    if x < 0:
        raise ViewError(f"round_half_up expects a non-negative value, got {x!r}")
    return int(math.floor(x + 0.5))


def sample_count(seconds: float, sample_rate: int) -> int:
    """Samples represented by `seconds` at `sample_rate` (locked rounding)."""
    if seconds <= 0:
        raise ViewError(f"duration must be positive, got {seconds!r}")
    if sample_rate <= 0:
        raise ViewError(f"sample rate must be positive, got {sample_rate!r}")
    return round_half_up(float(seconds) * int(sample_rate))


def select_channel(waveform: np.ndarray, policy: str = "select",
                   index: int = 0) -> tuple[np.ndarray, dict]:
    """Pick one channel by explicit policy; never fold channels silently.

    Returns (mono float32 array, resolved channel info).
    """
    if waveform.ndim != 2:
        raise ViewError(f"expected waveform [channels, samples], got shape {waveform.shape}")
    n_channels = int(waveform.shape[0])
    if n_channels < 1:
        raise ViewError("decoded waveform has no channels")
    if policy == "select":
        if not isinstance(index, int) or index < 0 or index >= n_channels:
            raise ViewError(
                f"channel_index {index!r} out of range for {n_channels} channel(s)")
        out = np.ascontiguousarray(waveform[index], dtype=np.float32)
    elif policy == "require_mono":
        if n_channels != 1:
            raise ViewError(f"channel_policy=require_mono but file has {n_channels} channels")
        out = np.ascontiguousarray(waveform[0], dtype=np.float32)
    else:
        raise ViewError(f"unknown channel_policy: {policy!r}")
    info = {
        "channel_policy": policy,
        "channel_index": 0 if policy == "require_mono" else int(index),
        "n_channels": n_channels,
    }
    return out, info


def crop(waveform: np.ndarray, start: int, end: int) -> np.ndarray:
    """Take [start, end) from a 1-D waveform, validating the coordinates."""
    n = int(waveform.shape[0])
    if start < 0 or end < start or end > n:
        raise ViewError(f"invalid crop [{start}, {end}) for {n} samples")
    return np.ascontiguousarray(waveform[start:end], dtype=np.float32)


def resample(waveform: np.ndarray, source_sr: int, target_sr: int) -> tuple[np.ndarray, dict]:
    """Resample to `target_sr` with the locked implementation.

    The result length is checked against the deterministic expectation; a
    difference of one sample is aligned explicitly and recorded. Larger
    differences raise instead of silently producing a different duration.
    """
    info = {
        "implementation": RESAMPLE_IMPLEMENTATION,
        "method": RESAMPLE_METHOD,
        "params": dict(RESAMPLE_PARAMS),
        "version": RESAMPLE_VERSION,
        "source_sample_rate": int(source_sr),
        "target_sample_rate": int(target_sr),
        "applied": int(source_sr) != int(target_sr),
    }
    if int(source_sr) == int(target_sr):
        info.update({"expected_num_samples": int(waveform.shape[0]),
                     "actual_num_samples": int(waveform.shape[0]),
                     "length_adjustment_samples": 0})
        return np.ascontiguousarray(waveform, dtype=np.float32), info

    try:
        import torch
        import torchaudio.functional as ta_functional
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ViewError(
            "resampling requires a locked torchaudio install "
            f"({RESAMPLE_IMPLEMENTATION}); import failed: {e}") from e

    tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)
    out = ta_functional.resample(
        tensor,
        orig_freq=int(source_sr),
        new_freq=int(target_sr),
        lowpass_filter_width=RESAMPLE_PARAMS["lowpass_filter_width"],
        rolloff=RESAMPLE_PARAMS["rolloff"],
        resampling_method=RESAMPLE_METHOD,
        beta=RESAMPLE_PARAMS["beta"],
    ).squeeze(0).numpy().astype(np.float32, copy=False)

    expected = round_half_up(waveform.shape[0] * int(target_sr) / int(source_sr))
    actual = int(out.size)
    if abs(actual - expected) > 1:
        raise ViewError(
            f"resample length mismatch: expected {expected}, implementation gave {actual} "
            f"({waveform.shape[0]} @ {source_sr} -> {target_sr})")
    adjustment = expected - actual
    if adjustment > 0:
        out = np.pad(out, (0, adjustment))
    elif adjustment < 0:
        out = out[:expected]
    info.update({
        "expected_num_samples": expected,
        "actual_num_samples": int(out.size),
        "length_adjustment_samples": int(adjustment),
    })
    return out, info


def pad_right(waveform: np.ndarray, target_num_samples: int) -> tuple[np.ndarray, int]:
    """Right zero pad to `target_num_samples`; returns (padded, pad_count)."""
    n = int(waveform.shape[0])
    if target_num_samples < n:
        raise ViewError(f"cannot pad {n} samples down to {target_num_samples}")
    pad = target_num_samples - n
    if pad == 0:
        return np.ascontiguousarray(waveform, dtype=np.float32), 0
    return np.pad(waveform, (0, pad)), pad


@dataclass
class FrontendGeometry:
    """Millisecond-defined analysis geometry for one actual sample rate."""
    sample_rate: int
    frame_ms: float
    hop_ms: float
    win_length: int
    hop_length: int
    n_fft: int

    @property
    def min_valid_samples(self) -> int:
        # center=false MFCC needs at least one full analysis window.
        return self.n_fft

    def as_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "frame_ms": self.frame_ms,
            "hop_ms": self.hop_ms,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "min_valid_samples": self.min_valid_samples,
        }


def next_power_of_two(n: int) -> int:
    if n < 1:
        raise ViewError(f"next_power_of_two expects a positive integer, got {n!r}")
    return 1 << (int(n) - 1).bit_length()


def frontend_geometry(sample_rate: int, frame_ms: float = 25.0,
                      hop_ms: float = 10.0) -> FrontendGeometry:
    """Derive win/hop/n_fft from milliseconds at the actual sample rate.

    Never reuse 16 kHz numbers for a 44.1 kHz waveform (data-layer doc §7.2).
    """
    win = max(1, round_half_up(frame_ms / 1000.0 * sample_rate))
    hop = max(1, round_half_up(hop_ms / 1000.0 * sample_rate))
    return FrontendGeometry(
        sample_rate=int(sample_rate),
        frame_ms=float(frame_ms),
        hop_ms=float(hop_ms),
        win_length=win,
        hop_length=hop,
        n_fft=next_power_of_two(win),
    )
