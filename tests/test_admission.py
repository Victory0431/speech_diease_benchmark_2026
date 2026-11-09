"""D4 admission: label review, mic/origin rules, person-isolated manifest."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

from sddbench.cli import main
from sddbench.data.adapters.parkinson_kcl import ParkinsonKCLAdapter, parse_stem
from sddbench.data.admission import (
    build_manifest,
    review_inventory,
    select_rows,
    validate_review,
)
from sddbench.data.scan import scan_files
from sddbench.data.schema import (
    COL_CHANNELS,
    COL_DECODE_STATUS,
    COL_DURATION_DECODED_SEC,
    COL_LABEL_RAW,
    COL_SAMPLE_RATE,
    COL_SHA256,
)

REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "configs" / "datasets" / "uaspeech_v1.yaml"


def _write_review(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "speaker_token", "directory_group", "confirmed_subject_id",
            "confirmed_label", "evidence", "review_status"])
        writer.writeheader()
        writer.writerows(rows)


def _run_prepare(ws, paths_file, *extra):
    return main(["prepare", "--dataset", "uaspeech", "--source", "scan1",
                 "--task", str(TASK), "--paths", str(paths_file), *extra])


def _latest_prepared(ws) -> Path:
    dirs = sorted((ws / "prepared" / "uaspeech_v1").iterdir())
    return dirs[-1]


def test_kcl_naming_parser():
    p = parse_stem("ID15_hc_0_0_0")
    assert p == {"speaker_token": "ID15", "kind": "hc", "rest": "0_0_0",
                 "missing_separator": False}
    p = parse_stem("ID22hc_0_0_0")
    assert p["speaker_token"] == "ID22" and p["missing_separator"] is True
    p = parse_stem("ID02_pd_2_0_0_1")
    assert p["kind"] == "pd" and p["rest"] == "2_0_0_1"
    assert parse_stem("junk_name") is None


def test_missing_review_blocks(scan_workspace):
    ws, paths_file, _ = scan_workspace
    assert _run_prepare(ws, paths_file) == 3
    out = _latest_prepared(ws)
    job = json.loads((out / "admission.json").read_text(encoding="utf-8"))
    assert job["status"] == "blocked_label_review_missing"
    assert not (out / "manifest.csv").exists()
    assert "label_review_missing" in {b["issue_type"] for b in job["blockers"]}


def test_pending_template_blocks(scan_workspace):
    ws, paths_file, _ = scan_workspace
    assert _run_prepare(ws, paths_file, "--write-review-template") == 3
    review = ws / "label_review" / "uaspeech_label_review.csv"
    assert review.is_file()
    rows = pd.read_csv(review)
    assert set(rows["review_status"]) == {"pending"}
    out = _latest_prepared(ws)
    job = json.loads((out / "admission.json").read_text(encoding="utf-8"))
    assert job["status"] == "blocked_label_review_incomplete"
    assert not (out / "manifest.csv").exists()


def test_confirmed_review_writes_manifest(scan_workspace):
    ws, paths_file, _ = scan_workspace
    _write_review(ws / "label_review" / "uaspeech_label_review.csv", [
        {"speaker_token": "CF02", "directory_group": "Disorder",
         "confirmed_subject_id": "S001", "confirmed_label": "disorder",
         "evidence": "unit-test fixture", "review_status": "confirmed"},
        {"speaker_token": "F02", "directory_group": "Healthy",
         "confirmed_subject_id": "S002", "confirmed_label": "control",
         "evidence": "unit-test fixture", "review_status": "confirmed"},
        {"speaker_token": "F03", "directory_group": "Healthy",
         "confirmed_subject_id": "S003", "confirmed_label": "control",
         "evidence": "unit-test fixture", "review_status": "confirmed"},
    ])
    assert _run_prepare(ws, paths_file, "--mic", "M2") == 0
    out = _latest_prepared(ws)
    manifest = pd.read_csv(out / "manifest.csv")
    assert len(manifest) == 3
    assert set(manifest["label"]) == {"disorder", "control"}
    assert set(manifest["group_id"]) == {
        "uaspeech:S001", "uaspeech:S002", "uaspeech:S003"}
    assert manifest["origin_id"].is_unique
    assert set(manifest["sample_id"]) == {
        "Disorder/CF02_B1_C1_M2.wav", "Healthy/F02_B1_C5_M2.wav",
        "Healthy/F03_B1_C3_M2.wav"}
    assert (manifest["label_source"].str.startswith("uaspeech_label_review.csv#line")).all()
    job = json.loads((out / "admission.json").read_text(encoding="utf-8"))
    assert job["status"] == "approved"
    assert job["counts"]["n_subjects"] == 3
    assert job["manifest_fingerprint"]


def test_mic_required_blocks_even_with_confirmed_review(scan_workspace):
    ws, paths_file, _ = scan_workspace
    _write_review(ws / "label_review" / "uaspeech_label_review.csv", [
        {"speaker_token": "CF02", "directory_group": "Disorder",
         "confirmed_subject_id": "S001", "confirmed_label": "disorder",
         "evidence": "unit-test", "review_status": "confirmed"},
        {"speaker_token": "CM13", "directory_group": "Disorder",
         "confirmed_subject_id": "S004", "confirmed_label": "disorder",
         "evidence": "unit-test", "review_status": "confirmed"},
        {"speaker_token": "CF05", "directory_group": "Disorder",
         "confirmed_subject_id": "S005", "confirmed_label": "disorder",
         "evidence": "unit-test", "review_status": "confirmed"},
        {"speaker_token": "F02", "directory_group": "Healthy",
         "confirmed_subject_id": "S002", "confirmed_label": "control",
         "evidence": "unit-test", "review_status": "confirmed"},
        {"speaker_token": "M01", "directory_group": "Healthy",
         "confirmed_subject_id": "S006", "confirmed_label": "control",
         "evidence": "unit-test", "review_status": "confirmed"},
        {"speaker_token": "M04", "directory_group": "Healthy",
         "confirmed_subject_id": "S003", "confirmed_label": "control",
         "evidence": "unit-test", "review_status": "confirmed"},
        {"speaker_token": "F03", "directory_group": "Healthy",
         "confirmed_subject_id": "S007", "confirmed_label": "control",
         "evidence": "unit-test", "review_status": "confirmed"},
    ])
    assert _run_prepare(ws, paths_file) == 3
    out = _latest_prepared(ws)
    job = json.loads((out / "admission.json").read_text(encoding="utf-8"))
    assert job["status"] == "blocked_mic_not_selected"
    assert not (out / "manifest.csv").exists()


def test_subject_conflict_blocks(scan_workspace):
    ws, paths_file, _ = scan_workspace
    _write_review(ws / "label_review" / "uaspeech_label_review.csv", [
        {"speaker_token": "CF02", "directory_group": "Disorder",
         "confirmed_subject_id": "S001", "confirmed_label": "disorder",
         "evidence": "unit-test", "review_status": "confirmed"},
        {"speaker_token": "F02", "directory_group": "Healthy",
         "confirmed_subject_id": "S001", "confirmed_label": "control",
         "evidence": "unit-test", "review_status": "confirmed"},
        {"speaker_token": "F03", "directory_group": "Healthy",
         "confirmed_subject_id": "S003", "confirmed_label": "control",
         "evidence": "unit-test", "review_status": "confirmed"},
    ])
    assert _run_prepare(ws, paths_file, "--mic", "M2") == 3
    out = _latest_prepared(ws)
    job = json.loads((out / "admission.json").read_text(encoding="utf-8"))
    assert job["status"] == "blocked_subject_label_conflict"
    assert not (out / "manifest.csv").exists()


def test_file_grouping_needs_no_speaker_metadata():
    from sddbench.data.schema import (
        COL_CATEGORY,
        COL_DIRECTORY_GROUP,
        COL_ORIGIN_ID,
        COL_PARSE_STATUS,
        COL_RELATIVE_PATH,
        COL_SPEAKER_TOKEN,
        CATEGORY_AUDIO,
    )

    rows = []
    for rel in ("Control/a.wav", "Control/b.wav", "Disease/c.wav"):
        rows.append({
            COL_RELATIVE_PATH: rel,
            COL_CATEGORY: CATEGORY_AUDIO,
            COL_DECODE_STATUS: "ok",
            COL_PARSE_STATUS: "ok",
            COL_SPEAKER_TOKEN: "",
            COL_DIRECTORY_GROUP: rel.split("/")[0],
            COL_ORIGIN_ID: rel,
        })

    inventory = review_inventory(rows, grouping="file")
    assert {(r["speaker_token"], r["directory_group"]) for r in inventory} == {
        ("", "Control"), ("", "Disease")}

    review = {
        ("", "Control"): {"confirmed_subject_id": "", "confirmed_label": "control",
                          "review_status": "confirmed"},
        ("", "Disease"): {"confirmed_subject_id": "", "confirmed_label": "disorder",
                          "review_status": "confirmed"},
    }
    check = validate_review(review, inventory, grouping="file")
    assert not check["blocking"]

    manifest, issues = build_manifest(
        rows, check["confirmed"], "demo", Path("demo_review.csv"),
        {"require_mic": False}, grouping="file")
    assert not [i for i in issues if i["severity"] == "error"]
    assert all(m["subject_id"] == "" for m in manifest)
    assert {m["group_id"] for m in manifest} == {
        "demo:Control/a.wav", "demo:Control/b.wav", "demo:Disease/c.wav"}
    assert {m["label"] for m in manifest} == {"control", "disorder"}


def test_kcl_adapter_reuses_admission(tmp_path):
    root = tmp_path / "kcl"
    (root / "Healthy").mkdir(parents=True)
    (root / "Parkinson").mkdir()
    for rel in ["Healthy/ID00_hc_0_0_0.wav", "Healthy/ID00_hc_0_0_0_1.wav",
                "Parkinson/ID01_pd_2_0_0.wav"]:
        sf.write(str(root / rel), [0.0] * 800, 44100, subtype="PCM_24")

    rows, _ = scan_files(root)
    adapter = ParkinsonKCLAdapter()
    result = adapter.collect(root, {}, rows)
    all_rows = []
    for row in rows:
        enriched = dict(row, **result.enrichment.get(row["relative_path"], {}))
        enriched.update({
            COL_DECODE_STATUS: "ok", COL_SAMPLE_RATE: 44100, COL_CHANNELS: 1,
            COL_DURATION_DECODED_SEC: 0.2, COL_SHA256: "fixture",
        })
        all_rows.append(enriched)

    inventory = review_inventory(all_rows)
    assert {r["speaker_token"] for r in inventory} == {"ID00", "ID01"}
    assert any(i["issue_type"] == "repeat_take_review" for i in result.issues)

    review = {
        ("ID00", "Healthy"): {"confirmed_subject_id": "K100",
                              "confirmed_label": "control",
                              "review_status": "confirmed"},
        ("ID01", "Parkinson"): {"confirmed_subject_id": "K101",
                                "confirmed_label": "parkinson",
                                "review_status": "confirmed"},
    }
    check = validate_review(review, inventory)
    assert not check["blocking"]
    selected, issues, blockers = select_rows(
        all_rows, {"require_mic": False, "one_per_origin": True})
    assert not blockers and selected
    manifest, manifest_issues = build_manifest(
        selected, check["confirmed"], "parkinson_kcl",
        Path("parkinson_kcl_label_review.csv"), {"variant": "original"})
    assert not [i for i in manifest_issues if i["severity"] == "error"]
    assert len(manifest) == 3
    assert {m["subject_id"] for m in manifest} == {"K100", "K101"}
    assert all(m["group_id"].startswith("parkinson_kcl:") for m in manifest)
    assert {m["label"] for m in manifest} == {"control", "parkinson"}
