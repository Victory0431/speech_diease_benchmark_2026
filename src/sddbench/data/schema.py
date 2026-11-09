"""Column schema shared by scan / adapter / audit / stats (D1 scope).

One row of `files.csv` = one discovered file, enriched in stages:
scan (base facts) -> adapter (naming/identity candidates) -> audit (probe/decode facts).

Column names are the single source of truth; modules must reference these
constants, never positional tuples. Training manifest columns (macro doc §5.1)
arrive with prepare (D4+) and are deliberately out of scope here.
"""
from __future__ import annotations

# --- discovery facts (scan.py) ---
COL_RELATIVE_PATH = "relative_path"
COL_FILE_NAME = "file_name"
COL_EXTENSION = "extension"
COL_SIZE_BYTES = "size_bytes"
COL_MTIME_NS = "mtime_ns"
COL_CATEGORY = "category"  # audio_candidate | archive | sidecar | unknown | symlink_file

# --- naming/identity candidates (adapters) ---
COL_SPEAKER_TOKEN = "speaker_token"
COL_BLOCK_TOKEN = "block_token"
COL_PROMPT_TOKEN = "prompt_token"
COL_MIC_TOKEN = "mic_token"
COL_DIRECTORY_GROUP = "directory_group"
COL_LABEL_RAW = "label_raw"
COL_PARSE_STATUS = "parse_status"  # ok | parse_failed
COL_ORIGIN_ID = "origin_id"  # candidate same-utterance group, pending review
COL_REVIEW_STATUS = "review_status"  # pending until label_review.csv confirms

# --- probe facts (decoders.probe) ---
COL_PROBE_STATUS = "probe_status"  # ok_soundfile | ok_ffprobe | failed | not_attempted
COL_CONTAINER = "container"  # actual container per decoder (e.g. WAV, FLAC)
COL_ENCODING = "encoding"  # sample encoding/subtype (e.g. PCM_16); null when N/A
COL_AUDIO_CODEC = "audio_codec"  # codec name when probed via ffprobe
COL_SAMPLE_RATE = "sample_rate"
COL_CHANNELS = "channels"
COL_FRAMES = "frames"  # header/probe frame count
COL_DURATION_HEADER_SEC = "duration_header_sec"  # estimate from header/probe
COL_EXTENSION_MISMATCH = "extension_mismatch"  # declared ext vs actual container

# --- decode facts (full mode, decoders.decode + audit) ---
COL_DECODE_STATUS = "decode_status"  # ok | failed | not_attempted
COL_DECODE_BACKEND = "decode_backend"
COL_N_SAMPLES = "n_samples"  # samples per channel, actually decoded
COL_DURATION_DECODED_SEC = "duration_decoded_sec"
COL_SHA256 = "sha256"
COL_QUALITY_PEAK = "quality_peak"
COL_QUALITY_RMS = "quality_rms"
COL_QUALITY_DC = "quality_dc"
COL_QUALITY_NEAR_FULL_RATIO = "quality_near_full_ratio"
COL_QUALITY_NONFINITE = "quality_nonfinite"
COL_ALL_ZERO = "all_zero"
COL_QUALITY_PER_CHANNEL = "quality_per_channel"  # JSON list, per-channel peak/rms/dc/near_full

# canonical column order for files.csv
FILES_COLUMNS = [
    COL_RELATIVE_PATH, COL_FILE_NAME, COL_EXTENSION, COL_SIZE_BYTES, COL_MTIME_NS,
    COL_CATEGORY,
    COL_DIRECTORY_GROUP, COL_SPEAKER_TOKEN, COL_BLOCK_TOKEN, COL_PROMPT_TOKEN,
    COL_MIC_TOKEN, COL_LABEL_RAW, COL_ORIGIN_ID, COL_PARSE_STATUS, COL_REVIEW_STATUS,
    COL_PROBE_STATUS, COL_CONTAINER, COL_ENCODING, COL_AUDIO_CODEC,
    COL_SAMPLE_RATE, COL_CHANNELS, COL_FRAMES, COL_DURATION_HEADER_SEC,
    COL_EXTENSION_MISMATCH,
    COL_DECODE_STATUS, COL_DECODE_BACKEND, COL_N_SAMPLES, COL_DURATION_DECODED_SEC,
    COL_SHA256,
    COL_QUALITY_PEAK, COL_QUALITY_RMS, COL_QUALITY_DC, COL_QUALITY_NEAR_FULL_RATIO,
    COL_QUALITY_NONFINITE, COL_ALL_ZERO, COL_QUALITY_PER_CHANNEL,
]

# --- D4 admission: label review + approved manifest (schema source of truth) ---
LABEL_REVIEW_COLUMNS = [
    "speaker_token", "directory_group", "confirmed_subject_id",
    "confirmed_label", "evidence", "review_status",
]
COL_SUBJECT_ID = "subject_id"
COL_GROUP_ID = "group_id"
COL_LABEL = "label"
COL_VARIANT = "variant"
COL_ELICITATION = "elicitation"
COL_OFFICIAL_SPLIT = "official_split"
COL_LABEL_SOURCE = "label_source"
MANIFEST_COLUMNS = [
    "sample_id", COL_RELATIVE_PATH, COL_SUBJECT_ID, COL_GROUP_ID, COL_ORIGIN_ID,
    COL_LABEL_RAW, COL_LABEL, COL_VARIANT, COL_ELICITATION,
    COL_SAMPLE_RATE, COL_DURATION_DECODED_SEC, COL_CHANNELS, COL_SHA256,
    COL_OFFICIAL_SPLIT, COL_LABEL_SOURCE,
]

CATEGORY_AUDIO = "audio_candidate"
CATEGORY_ARCHIVE = "archive"
CATEGORY_SIDECAR = "sidecar"
CATEGORY_UNKNOWN = "unknown"
CATEGORY_SYMLINK_FILE = "symlink_file"

# review/issue rows (issues.csv)
ISSUE_COLUMNS = ["issue_type", "severity", "relative_path", "detail"]
