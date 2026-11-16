"""F11: probe and decode are separate capabilities; ffmpeg really decodes.

All samples are synthetic; no patient audio is used.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from sddbench.data.audit import audit_config_defaults, audit_one
from sddbench.data.decoders import DecodeError, decode, probe, tool_versions
from sddbench.data.schema import (
    CATEGORY_AUDIO,
    COL_CATEGORY,
    COL_DECODE_STATUS,
    COL_RELATIVE_PATH,
)
from sddbench.data.stats import collect_statistics


def _usable(exe: str) -> bool:
    path = shutil.which(exe)
    if not path:
        return False
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    try:
        return subprocess.run([path, "-version"], capture_output=True,
                              timeout=10, env=env).returncode == 0
    except OSError:
        return False


HAS_FFPROBE = _usable("ffprobe")
HAS_FFMPEG = _usable("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not usable here")
needs_ffprobe = pytest.mark.skipif(not HAS_FFPROBE, reason="ffprobe not usable here")


def _wav(root, name="ok.wav") -> None:
    t = np.arange(16000) / 16000.0
    sf.write(root / name, (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"),
             16000, subtype="PCM_16")


def _m4a(root, source: str = "ok.wav", name: str = "only_ffmpeg.m4a") -> bool:
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(root / source),
         "-c:a", "aac", "-b:a", "64k", str(root / name)],
        capture_output=True, env=env)
    return proc.returncode == 0


@needs_ffprobe
def test_probe_can_succeed_where_decoding_needs_a_fallback(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    _wav(root)
    if not HAS_FFMPEG or not _m4a(root):
        pytest.skip("ffmpeg/AAC encoder unavailable for the synthetic container")
    info = probe(root / "only_ffmpeg.m4a")
    assert info.status == "ok_ffprobe"  # header was readable...
    assert info.audio_codec == "aac"
    assert info.sample_rate == 16000


@needs_ffmpeg
def test_ffmpeg_fallback_decodes_compressed_container_at_native_rate(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    _wav(root)
    if not _m4a(root):
        pytest.skip("ffmpeg build lacks an AAC encoder")
    dec = decode(root / "only_ffmpeg.m4a")
    assert dec.backend.startswith("ffmpeg(")
    assert dec.sample_rate == 16000
    assert dec.waveform.dtype.name == "float32"
    assert dec.waveform.shape[0] == 1
    assert abs(dec.waveform.shape[1] / 16000 - 1.0) < 0.05
    assert np.isfinite(dec.waveform).all()
    assert tool_versions()["ffmpeg"]


@needs_ffmpeg
def test_audit_full_really_decodes_probe_only_files(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    _wav(root)
    if not _m4a(root):
        pytest.skip("ffmpeg build lacks an AAC encoder")
    rec = audit_one(str(root), "only_ffmpeg.m4a", "full", audit_config_defaults())
    assert rec[COL_DECODE_STATUS] == "ok"
    assert str(rec["decode_backend"]).startswith("ffmpeg(")
    assert rec["n_samples"] and abs(rec["duration_decoded_sec"] - 1.0) < 0.05


def test_decode_failure_is_explicit_when_no_backend_can_read_the_file(tmp_path):
    bad = tmp_path / "not_audio.wav"
    bad.write_text("this is not audio", encoding="utf-8")
    with pytest.raises(DecodeError):
        decode(bad)


def test_audit_status_counts_close_over_candidates(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    _wav(root)
    (root / "bad.wav").write_text("broken", encoding="utf-8")
    has_m4a = HAS_FFMPEG and _m4a(root)
    names = ["ok.wav", "bad.wav"] + (["only_ffmpeg.m4a"] if has_m4a else [])
    rows = []
    for rel in names:
        rec = audit_one(str(root), rel, "full", audit_config_defaults())
        rows.append({COL_RELATIVE_PATH: rel, COL_CATEGORY: CATEGORY_AUDIO, **rec})
    stats = collect_statistics(rows)
    counts = stats["counts"]
    assert counts["audio_candidates"] == len(names)
    assert counts["decode_unclassified"] == 0
    assert (counts["decoded_ok"] + counts["decode_failed"]
            + counts["decode_not_attempted"]) == counts["audio_candidates"]
    # a probe failure has no decoder input, so it must be "not attempted"
    bad = next(r for r in rows if r[COL_RELATIVE_PATH] == "bad.wav")
    assert bad[COL_DECODE_STATUS] == "not_attempted"
    assert bad["decode_error"].startswith("probe failed")
    if has_m4a:
        assert counts["decoded_ok"] == 2
    group = stats["groups"]["by_directory_group"]["(none)"]
    assert (group["n_decoded_ok"] + group["n_decode_failed"]
            + group["n_decode_not_attempted"]) == group["n_files"]
