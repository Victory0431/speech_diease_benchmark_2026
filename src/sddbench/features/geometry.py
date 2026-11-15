"""Frame geometry for feature recipes only.

Preprocessing produces a waveform; it never decides how that waveform is framed.
Keeping this math inside `features/` means the same audio view can feed MFCC,
log-mel, or any later recipe without the view layer knowing `frame_ms/hop_ms`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


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
        # center=false needs at least one full analysis window.
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


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError(f"next_power_of_two expects a positive integer, got {n!r}")
    return 1 << (int(n) - 1).bit_length()


def frontend_geometry(sample_rate: int, frame_ms: float,
                      hop_ms: float) -> FrontendGeometry:
    """Derive win/hop/n_fft from milliseconds at the actual sample rate."""
    if frame_ms <= 0 or hop_ms <= 0:
        raise ValueError(f"frame_ms/hop_ms must be positive, got {frame_ms!r}/{hop_ms!r}")
    win = max(1, _round_half_up(frame_ms / 1000.0 * sample_rate))
    hop = max(1, _round_half_up(hop_ms / 1000.0 * sample_rate))
    return FrontendGeometry(
        sample_rate=int(sample_rate),
        frame_ms=float(frame_ms),
        hop_ms=float(hop_ms),
        win_length=win,
        hop_length=hop,
        n_fft=next_power_of_two(win),
    )
