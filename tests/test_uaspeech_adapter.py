"""UAspeech adapter: token parsing and directory facts (data-layer doc §5)."""
from __future__ import annotations

from sddbench.data.adapters.uaspeech import UASpeechAdapter, parse_stem
from sddbench.data.schema import (
    CATEGORY_AUDIO, COL_CATEGORY, COL_DIRECTORY_GROUP, COL_MIC_TOKEN,
    COL_ORIGIN_ID, COL_PARSE_STATUS, COL_PROMPT_TOKEN, COL_RELATIVE_PATH,
    COL_REVIEW_STATUS, COL_SPEAKER_TOKEN,
)


def test_parse_cf02_example():
    # the user-provided example file; tokens stay CANDIDATES
    p = parse_stem("CF02_B1_C1_M2")
    assert p == {"speaker_token": "CF02", "block_token": "B1",
                 "prompt_token": "C1", "mic_token": "M2"}


def test_parse_cw_uw_prompt_tokens():
    assert parse_stem("CM13_B1_CW84_M8")["prompt_token"] == "CW84"
    assert parse_stem("CM10_B2_UW8_M5")["prompt_token"] == "UW8"


def test_parse_letter_only_and_second_digit_prompts():
    # verified on real data: LA..LZ long tokens and D0..D9 second digit set
    assert parse_stem("CF02_B1_LA_M2")["prompt_token"] == "LA"
    assert parse_stem("CF02_B1_D0_M2")["prompt_token"] == "D0"
    assert parse_stem("M05_B3_LZ_M7")["prompt_token"] == "LZ"


def test_speaker_token_keeps_c_prefix():
    # CF02 must NOT become F02
    assert parse_stem("CF02_B1_C1_M2")["speaker_token"] == "CF02"


def test_parse_failure():
    assert parse_stem("weirdname") is None
    assert parse_stem("CF02_B1_M2") is None  # missing segment
    assert parse_stem("CF02_X1_C1_M2") is None  # bad block token


def _make_inventory(root):
    from sddbench.data.scan import scan_files
    rows, _ = scan_files(root)
    return rows


def test_adapter_collect_directory_facts(synth_root):
    rows = _make_inventory(synth_root)
    result = UASpeechAdapter().collect(synth_root, {}, rows)
    enr = result.enrichment
    row = enr["Disorder/CF02_B1_C1_M2.wav"]
    assert row[COL_DIRECTORY_GROUP] == "Disorder"
    assert row[COL_SPEAKER_TOKEN] == "CF02"
    assert row[COL_PARSE_STATUS] == "ok"
    assert row[COL_REVIEW_STATUS] == "pending"  # no label_review.csv yet
    # same utterance across mics shares candidate origin
    assert enr["Disorder/CF02_B1_C1_M2.wav"][COL_ORIGIN_ID] == "CF02|B1|C1"
    assert enr["Disorder/CF02_B1_C1_M3.wav"][COL_ORIGIN_ID] == "CF02|B1|C1"
    assert enr["Disorder/CF02_B1_C1_M3.wav"][COL_MIC_TOKEN] == "M3"


def test_adapter_flags_parse_failures(synth_root):
    rows = _make_inventory(synth_root)
    result = UASpeechAdapter().collect(synth_root, {}, rows)
    parse_issues = [i for i in result.issues if i["issue_type"] == "filename_parse_failed"]
    assert any("weirdname.wav" in i["relative_path"] for i in parse_issues)
    assert all(i["severity"] == "error" for i in parse_issues)


def test_adapter_flags_prefix_segregation_for_review(synth_root):
    rows = _make_inventory(synth_root)
    result = UASpeechAdapter().collect(synth_root, {}, rows)
    seg = [i for i in result.issues if i["issue_type"] == "prefix_directory_segregation_review"]
    assert len(seg) == 1 and seg[0]["severity"] == "warning"
    # disorder group holds only C-prefixed speakers in the fixture
    assert "CF02" in seg[0]["detail"]


def test_label_review_confirmed_rows_override_pending(synth_root, tmp_path):
    import csv
    review = tmp_path / "uaspeech_label_review.csv"
    with review.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_token", "directory_group", "confirmed_subject_id",
                    "confirmed_label", "evidence", "review_status"])
        w.writerow(["CF02", "Disorder", "CF02", "control_pending_verify",
                    "official corpus naming convention", "confirmed"])
    rows = _make_inventory(synth_root)
    cfg = {"label_review": {"file": str(review)}}
    result = UASpeechAdapter().collect(synth_root, cfg, rows)
    row = result.enrichment["Disorder/CF02_B1_C1_M2.wav"]
    assert row[COL_REVIEW_STATUS] == "confirmed"
