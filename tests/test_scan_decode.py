"""scan + decoder behaviour on synthetic files (macro doc stage-2 style checks)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sddbench.data.scan import scan_files
from sddbench.data.schema import (
    CATEGORY_ARCHIVE, CATEGORY_AUDIO, CATEGORY_SIDECAR, CATEGORY_UNKNOWN,
    COL_CATEGORY, COL_RELATIVE_PATH,
)
from sddbench.data.decoders import DecodeError, decode, extension_matches, probe


def test_scan_classification(synth_root):
    rows, issues = scan_files(synth_root)
    by_path = {r[COL_RELATIVE_PATH]: r for r in rows}
    assert len(rows) == 13  # 11 audio-ish (incl. bad/flac-mismatch) + zip + sidecar
    assert by_path["Disorder/CF02_B1_C1_M2.wav"][COL_CATEGORY] == CATEGORY_AUDIO
    assert by_path["notes.zip"][COL_CATEGORY] == CATEGORY_ARCHIVE
    assert by_path["Disorder/._CF02_B1_C1_M2.wav"][COL_CATEGORY] == CATEGORY_SIDECAR
    # sorted, deterministic order
    assert rows == sorted(rows, key=lambda r: r[COL_RELATIVE_PATH])


def test_scan_does_not_follow_dir_symlinks(synth_root):
    link = synth_root / "elsewhere"
    real = synth_root.parent / "real_audio"
    real.mkdir(exist_ok=True)
    link.symlink_to(real, target_is_directory=True)
    rows, issues = scan_files(synth_root)
    assert all(not r[COL_RELATIVE_PATH].startswith("elsewhere/") for r in rows)
    assert any(i["issue_type"] == "symlink_dir_not_followed" for i in issues)


def test_probe_real_wav(synth_root):
    info = probe(synth_root / "Disorder" / "CF02_B1_C1_M2.wav")
    assert info.status == "ok_soundfile"
    assert info.container == "WAV"
    assert info.encoding == "PCM_16"
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert abs(info.duration_sec - 2.0) < 0.01


def test_probe_corrupt_file_reports_failed_not_empty(synth_root):
    info = probe(synth_root / "Healthy" / "BAD_B1_C1_M2.wav")
    assert info.status == "failed"  # never "empty library"


def test_decode_returns_real_sr_float32(synth_root):
    dec = decode(synth_root / "Disorder" / "CF02_B1_C1_M2.wav")
    assert dec.sample_rate == 16000  # the file's real rate, not an assumed constant
    assert dec.waveform.dtype.name == "float32"
    assert dec.waveform.shape[0] == 1  # [C, N]
    assert abs(dec.waveform.shape[1] / 16000 - 2.0) < 0.01


def test_decode_failure_raises_decodeerror(synth_root):
    with pytest.raises(DecodeError):
        decode(synth_root / "Healthy" / "BAD_B1_C1_M2.wav")


def test_extension_mismatch_flac_in_wav(synth_root):
    info = probe(synth_root / "Healthy" / "F03_B1_C3_M2.wav")
    assert info.container == "FLAC"
    assert extension_matches(".wav", info.container) is False


def test_extension_match_table():
    assert extension_matches(".wav", "WAV") is True
    assert extension_matches(".flac", "FLAC") is True
    assert extension_matches(".mp3", None) is None  # no claim without container info
