"""Statistics aggregation: quantiles, grouping, length candidates (doc §4)."""
from __future__ import annotations

import numpy as np

from sddbench.data.audit import audit_one
from sddbench.data.schema import COL_DECODE_STATUS, COL_DURATION_DECODED_SEC
from sddbench.data.stats import _length_candidates, collect_statistics


def test_percentiles_known_values():
    # build enriched rows synthetically: durations 1..5
    rows = [{"category": "audio_candidate", "decode_status": "ok",
             "duration_decoded_sec": float(d), "directory_group": "G",
             "speaker_token": f"S{d}", "mic_token": "M2", "prompt_token": "C1",
             "sample_rate": 16000, "channels": 1, "container": "WAV",
             "probe_status": "ok_soundfield_fake", "parse_status": "ok",
             "extension_mismatch": False, "quality_peak": 0.5,
             "quality_rms": 0.2, "quality_near_full_ratio": 0.0,
             "quality_nonfinite": 0, "all_zero": False,
             "relative_path": f"G/S{d}.wav"}
            for d in range(1, 6)]
    stats = collect_statistics(rows, {"length_candidates": [1.0, 4.0]})
    dur = stats["duration_seconds"]["decoded_full"]
    assert dur["n"] == 5
    assert dur["p50"] == 3.0
    assert np.isclose(dur["p60"], 3.4)  # linear: idx 0.6*(5-1)=2.4 -> 3 + 0.4
    assert np.isclose(dur["p90"], 4.6)  # linear: idx 3.6 -> 4 + 0.6
    assert np.isclose(dur["total"], 15.0)


def test_length_candidates_formulas():
    d = [2.0, 5.0, 12.0]  # L=10: valid = 2+5+10=17, pad = 8+5+0=13, cut = 0+0+2
    out = _length_candidates(d, [10.0])
    r = out[0]
    assert r["n_shorter_than_L"] == 2
    assert r["n_longer_than_L"] == 1
    assert np.isclose(r["valid_seconds_total"], 17.0)
    assert np.isclose(r["padded_seconds_total"], 13.0)
    assert np.isclose(r["truncated_seconds_total"], 2.0)
    assert np.isclose(r["total_padding_ratio"], 13.0 / 30.0)  # pad/(n*L)
    assert np.isclose(r["mean_per_recording_padding_ratio"], np.mean([8/10, 5/10, 0/10]))
    assert np.isclose(r["retention_ratio"], 17.0 / 19.0)


def test_quick_and_full_durations_not_mixed():
    rows = [{"category": "audio_candidate", "decode_status": "ok",
             "duration_decoded_sec": 3.0, "duration_header_sec": 2.9,
             "directory_group": "G", "speaker_token": "S1", "mic_token": "M2",
             "prompt_token": "C1", "sample_rate": 16000, "channels": 1,
             "container": "WAV", "probe_status": "ok_soundfile",
             "parse_status": "ok", "extension_mismatch": False,
             "quality_peak": None, "quality_rms": None,
             "quality_near_full_ratio": None, "quality_nonfinite": 0,
             "all_zero": False, "relative_path": "G/a.wav"}]
    stats = collect_statistics(rows)
    assert stats["duration_seconds"]["decoded_full"]["n"] == 1
    # header estimates live in their own key, never merged
    assert stats["duration_seconds"]["header_estimate_quick"]["n"] == 1


def test_audit_one_full_mode_measures_real_samples(synth_root):
    import json
    cfg = {"near_full_threshold": 0.99, "short_duration_sec": 0.1}
    res = audit_one(str(synth_root), "Disorder/CF02_B1_C1_M2.wav", "full", cfg)
    assert res[COL_DECODE_STATUS] == "ok"
    assert res["sample_rate"] == 16000
    assert res["duration_decoded_sec"] > 1.99
    assert res["sha256"] and len(res["sha256"]) == 64
    assert res["quality_peak"] <= 1.0 + 1e-6
    per_ch = json.loads(res["quality_per_channel"])
    assert per_ch[0]["channel"] == 0


def test_audit_one_corrupt_file_fails_cleanly(synth_root):
    cfg = {"near_full_threshold": 0.99, "short_duration_sec": 0.1}
    res = audit_one(str(synth_root), "Healthy/BAD_B1_C1_M2.wav", "full", cfg)
    assert res["probe_status"] == "failed"
    assert res[COL_DECODE_STATUS] == "failed"
