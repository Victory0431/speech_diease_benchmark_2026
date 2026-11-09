"""Shared synthetic fixtures. NO real patient audio is committed to Git."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


def _write_wav(path, duration_s=1.0, sr=16000, channels=1, freq=440.0, amplitude=0.5):
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    sig = amplitude * np.sin(2 * np.pi * freq * t)
    if channels == 2:
        sig = np.stack([sig, -sig], axis=1)
    sf.write(str(path), sig, sr, subtype="PCM_16")


@pytest.fixture
def synth_root(tmp_path):
    """A tiny fake dataset with: valid wavs (1ch/2ch), short wav, silent wav,
    corrupt .wav (text bytes), .wav containing FLAC content (mismatch),
    a zip archive, a sidecar file, an unparsable UAspeech-style name."""
    root = tmp_path / "fake_UAspeech"
    d1 = root / "Disorder"; d1.mkdir(parents=True)
    d2 = root / "Healthy"; d2.mkdir(parents=True)

    _write_wav(d1 / "CF02_B1_C1_M2.wav", 2.0)          # 2 s mono
    _write_wav(d1 / "CF02_B1_CW84_M3.wav", 1.5)        # prompt token CW
    _write_wav(d1 / "CM13_B2_UW8_M4.wav", 0.9)         # prompt token UW, <1 s
    _write_wav(d1 / "CF05_B3_C10_M8.wav", 1.0, channels=2)
    _write_wav(d1 / "CF02_B1_C1_M3.wav", 2.0)          # same utterance, other mic
    _write_wav(d2 / "F02_B1_C5_M2.wav", 0.02)          # extremely short
    _write_wav(d2 / "M01_B2_CW11_M5.wav", 1.2)
    # silent all-zero wav
    sf.write(str(d2 / "M04_B1_C2_M6.wav"), np.zeros((16000,), dtype="float32"), 16000)
    # corrupt: text bytes with .wav extension
    (d2 / "BAD_B1_C1_M2.wav").write_bytes(b"this is not audio at all" * 10)
    # mismatch: FLAC container stored with .wav extension
    _write_wav(tmp_path / "_tmp.flac", 1.0)
    (tmp_path / "_tmp.flac").rename(d2 / "F03_B1_C3_M2.wav")
    # archive + sidecar
    import zipfile
    with zipfile.ZipFile(root / "notes.zip", "w") as z:
        z.writestr("readme.txt", "hi")
    (d1 / "._CF02_B1_C1_M2.wav").write_bytes(b"\x00\x01resourcefork")
    # unparsable stem
    _write_wav(d2 / "weirdname.wav", 0.5)
    return root
