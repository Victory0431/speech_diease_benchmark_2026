"""D2: exact crop/pad coordinates, determinism, channel and resample rules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from sddbench.cli import main
from sddbench.data.audio import ViewError, pad_right, resample, select_channel
from sddbench.data.audit import audit_config_defaults, run_audit
from sddbench.data.adapters.registry import get_adapter
from sddbench.data.prepare import stratified_sample
from sddbench.data.scan import scan_files
from sddbench.data.schema import (
    COL_CATEGORY,
    COL_DECODE_STATUS,
    COL_DIRECTORY_GROUP,
    COL_RELATIVE_PATH,
    COL_SPEAKER_TOKEN,
    FILES_COLUMNS,
)
from sddbench.data.windows import (
    WINDOW_COLUMNS,
    build_view_index,
    derive_crop_seed,
    normalize_preprocess,
    window_geometry,
)


def _cfg(**over):
    base = {
        "schema_version": "1.0",
        "name": "test",
        "profile": "technical_only",
        "sample_rate_policy": "native",
        "allow_mixed_native": False,
        "target_sample_rate": None,
        "channel_policy": "select",
        "channel_index": 0,
        "view_mode": "single_segment",
        "observe_seconds": 10.0,
        "window_seconds": 10.0,
        "position": "head",
        "crop_seed": 0,
        "short_policy": "right_zero_pad",
        "amplitude_normalization": "none",
        "frame_ms": 25.0,
        "hop_ms": 10.0,
    }
    base.update(over)
    return normalize_preprocess(base)


def test_20s_take_10s_positions():
    n, sr = 20 * 16000, 16000
    expected = {"head": 0, "center": 80000, "tail": 160000}
    for pos, start in expected.items():
        g = window_geometry(n, sr, _cfg(position=pos), "sampleA")
        assert g["original_start_sample"] == start
        assert g["original_end_sample"] == start + 160000
        assert g["original_start_sec"] == start / sr


def test_20s_fixed_random_within_legal_range_and_deterministic():
    n, sr = 20 * 16000, 16000
    cfg = _cfg(position="fixed_random", crop_seed=7)
    g1 = window_geometry(n, sr, cfg, "sampleA")
    g2 = window_geometry(n, sr, cfg, "sampleA")
    assert 0 <= g1["original_start_sample"] <= 160000
    assert g1 == g2  # same seed, same sample -> same coordinate
    # enumeration order does not matter: geometry only depends on the seed material
    assert derive_crop_seed(7, "sampleA") == derive_crop_seed(7, "sampleA")
    assert derive_crop_seed(7, "sampleA") != derive_crop_seed(8, "sampleA")


def test_3s_take_10s_keeps_all_audio_and_pads_7s():
    n, sr = 3 * 16000, 16000
    wave = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    for pos in ("head", "center", "tail", "fixed_random"):
        g = window_geometry(n, sr, _cfg(position=pos), "short")
        assert g["original_start_sample"] == 0
        assert g["original_end_sample"] == n
        padded, pad = pad_right(wave, 10 * 16000)
        assert pad == 7 * 16000
        assert padded.shape[0] == 10 * 16000
        assert np.array_equal(padded[:n], wave)       # no sound lost/reordered
        assert np.all(padded[n:] == 0.0)              # padding is zeros, not looped


def test_pad_right_refuses_shrinking():
    with pytest.raises(ViewError):
        pad_right(np.zeros(100, dtype=np.float32), 50)


def test_channel_selection_never_averages(synth_root):
    dec = None
    from sddbench.data.decoders import decode
    dec = decode(synth_root / "Disorder" / "CF05_B3_C10_M8.wav")  # stereo: [+x, -x]
    ch0, info0 = select_channel(dec.waveform, "select", 0)
    ch1, info1 = select_channel(dec.waveform, "select", 1)
    assert info0["n_channels"] == 2 and info1["channel_index"] == 1
    # PCM_16 quantization is not perfectly symmetric, so compare within 1 LSB-ish
    assert np.allclose(ch0, -ch1, atol=1e-3)
    with pytest.raises(ViewError):
        select_channel(dec.waveform, "select", 2)
    with pytest.raises(ViewError):
        select_channel(dec.waveform, "require_mono", 0)


def test_resample_length_rule_and_native_shortcut():
    pytest.importorskip("torchaudio")
    sr = 16000
    wave = np.sin(2 * np.pi * 220 * np.arange(sr, dtype=np.float32) / sr)
    same, info = resample(wave, sr, sr)
    assert info["applied"] is False and same.shape == wave.shape
    out, info = resample(wave, sr, 8000)
    assert info["applied"] is True
    assert info["expected_num_samples"] == 8000
    assert out.shape[0] == 8000
    assert abs(info["length_adjustment_samples"]) <= 1
    out, info = resample(wave, 44100, 16000)  # non-integer ratio still exact length
    assert out.shape[0] == info["expected_num_samples"]


def test_config_validation_rejects_unimplemented_semantics():
    with pytest.raises(ValueError):
        _cfg(position="middle")
    with pytest.raises(ValueError):
        _cfg(channel_policy="average")
    with pytest.raises(ValueError):
        _cfg(short_policy="loop")
    with pytest.raises(ValueError):
        _cfg(sample_rate_policy="resample", target_sample_rate=None)
    with pytest.raises(ValueError):
        _cfg(view_mode="sliding")


def _make_scan(tmp_path: Path, synth_root: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "ws"
    scan_dir = workspace / "inventory" / "uaspeech" / "scan1"
    scan_dir.mkdir(parents=True)
    rows, _ = scan_files(synth_root)
    adapter = get_adapter("uaspeech")
    collected = adapter.collect(synth_root, {}, rows)
    merged = [dict(r, **collected.enrichment.get(r[COL_RELATIVE_PATH], {})) for r in rows]
    enriched, completed = run_audit(synth_root, merged, "full", scan_dir, workers=2,
                                    quality_cfg=audit_config_defaults())
    assert completed
    pd.DataFrame(enriched, columns=FILES_COLUMNS).to_csv(scan_dir / "files.csv", index=False)
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_id": "scan1", "mode": "full", "status": "completed",
        "task_config": {"admission": "pending_review"},
    }), encoding="utf-8")
    paths_file = workspace / "paths.local.yaml"
    paths_file.write_text(yaml.safe_dump({
        "workspace_root": str(workspace),
        "dataset_roots": {"uaspeech": str(synth_root)},
    }, allow_unicode=True), encoding="utf-8")
    return workspace, paths_file


def test_extract_features_end_to_end_keeps_originals(tmp_path, synth_root, monkeypatch):
    workspace, paths_file = _make_scan(tmp_path, synth_root)
    preprocess = Path(__file__).resolve().parents[1] / "configs" / "preprocessing" / \
        "uaspeech_native_smoke.yaml"
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in synth_root.rglob("*.wav")}
    monkeypatch.chdir(tmp_path)
    code = main([
        "extract-features", "--dataset", "uaspeech", "--source", "scan1",
        "--preprocess", str(preprocess), "--limit", "5", "--previews", "2",
        "--dry-run", "--paths", str(paths_file),
    ])
    assert code == 0
    job_dirs = list((workspace / "feature_jobs").iterdir())
    assert len(job_dirs) == 1
    job_dir = job_dirs[0]
    for name in ("job.json", "config.yaml", "windows.csv", "issues.csv", "inspection.md"):
        assert (job_dir / name).is_file()
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert job["status"] == "planned" and job["technical_only"] is True
    assert job["feature"] is None
    windows = pd.read_csv(job_dir / "windows.csv")
    assert list(windows.columns) == WINDOW_COLUMNS
    assert len(windows) <= 5 and len(windows) > 0
    assert (windows["processed_sample_rate"] == 16000).all()
    assert (windows["valid_num_samples"] + windows["pad_num_samples"]
            == windows["target_num_samples"]).all()
    # determinism: same config + same scan -> same view_hash for the same sample
    again = build_view_index(
        synth_root,
        [r for r in pd.read_csv(workspace / "inventory" / "uaspeech" / "scan1" / "files.csv")
         .to_dict("records") if r["relative_path"] == windows.iloc[0]["sample_id"]],
        normalize_preprocess(yaml.safe_load(preprocess.read_text(encoding="utf-8"))),
        source_scan_id="scan1", verify_source_hash=True)
    assert again.rows[0]["view_hash"] == windows.iloc[0]["view_hash"]
    # originals untouched
    for p, digest in originals.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest


def test_extract_features_rejects_missing_feature_config(tmp_path, synth_root, monkeypatch):
    _, paths_file = _make_scan(tmp_path, synth_root)
    monkeypatch.chdir(tmp_path)
    code = main([
        "extract-features", "--dataset", "uaspeech", "--source", "scan1",
        "--preprocess", "configs/preprocessing/uaspeech_native_smoke.yaml",
        "--features", "configs/features/mfcc20_stats.yaml", "--paths", str(paths_file),
    ])
    assert code == 2  # config does not exist here -> explicit error, not silent success


def test_stratified_sample_is_deterministic_and_cross_speaker():
    rows = []
    for group in ("Disorder", "Healthy"):
        for spk in ("S1", "S2", "S3"):
            for i in range(4):
                rel = f"{group}/{spk}_{i}.wav"
                rows.append({
                    COL_RELATIVE_PATH: rel, COL_CATEGORY: "audio_candidate",
                    COL_DECODE_STATUS: "ok", COL_DIRECTORY_GROUP: group,
                    COL_SPEAKER_TOKEN: spk,
                })
    chosen, summary = stratified_sample(rows, 6)
    assert [r[COL_RELATIVE_PATH] for r in chosen] == [
        r[COL_RELATIVE_PATH] for r in stratified_sample(list(reversed(rows)), 6)[0]]
    assert {r[COL_SPEAKER_TOKEN] for r in chosen} == {"S1", "S2", "S3"}
    assert {r[COL_DIRECTORY_GROUP] for r in chosen} == {"Disorder", "Healthy"}
    assert summary["strategy"] == "stratified_round_robin"
