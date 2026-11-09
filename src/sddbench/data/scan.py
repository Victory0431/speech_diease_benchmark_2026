"""File discovery: recursive enumeration of the dataset root.

Read-only. Never follows directory symlinks. Deterministic ordering
(sorted). Classifies files as audio_candidate / archive / sidecar / unknown.
Per data-layer doc §3: format handling lives here and in decoders, never in
dataset adapters.
"""
from __future__ import annotations

import os
from pathlib import Path

from .schema import (
    CATEGORY_ARCHIVE,
    CATEGORY_AUDIO,
    CATEGORY_SIDECAR,
    CATEGORY_SYMLINK_FILE,
    CATEGORY_UNKNOWN,
    COL_CATEGORY,
    COL_EXTENSION,
    COL_FILE_NAME,
    COL_MTIME_NS,
    COL_RELATIVE_PATH,
    COL_SIZE_BYTES,
)

AUDIO_EXTS = {
    ".wav", ".wave", ".flac", ".mp3", ".ogg", ".oga", ".m4a", ".aac",
    ".wma", ".aiff", ".aif", ".aifc", ".opus", ".webm", ".au", ".snd",
}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".rar", ".7z"}
SIDECAR_NAMES = {".ds_store", "thumbs.db"}


def classify(name: str) -> str:
    lower = name.lower()
    stem = os.path.basename(name)
    if stem.startswith("._") or lower in SIDECAR_NAMES:
        return CATEGORY_SIDECAR
    ext = os.path.splitext(lower)[1]
    if ext in AUDIO_EXTS:
        return CATEGORY_AUDIO
    if ext in ARCHIVE_EXTS:
        return CATEGORY_ARCHIVE
    return CATEGORY_UNKNOWN


def scan_files(root: Path) -> tuple[list[dict], list[dict]]:
    """Enumerate all files under `root` (read-only).

    Returns (rows, issues). Rows contain the base discovery columns only;
    adapters and audit enrich them later. Directory symlinks are recorded as
    issues and not followed.
    """
    rows: list[dict] = []
    issues: list[dict] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirpath_p = Path(dirpath)
        # skip symlinked directories, deterministically
        keep = []
        for d in sorted(dirnames):
            full = dirpath_p / d
            if full.is_symlink():
                issues.append({
                    "issue_type": "symlink_dir_not_followed",
                    "severity": "warning",
                    "relative_path": str(full.relative_to(root)),
                    "detail": "directory symlink not followed (scan policy)",
                })
            else:
                keep.append(d)
        dirnames[:] = keep
        for fname in sorted(filenames):
            full = dirpath_p / fname
            rel = str(full.relative_to(root))
            try:
                st = full.lstat()
            except OSError as e:
                issues.append({
                    "issue_type": "stat_failed",
                    "severity": "warning",
                    "relative_path": rel,
                    "detail": f"{type(e).__name__}: {e}",
                })
                continue
            category = CATEGORY_SYMLINK_FILE if full.is_symlink() else classify(fname)
            rows.append({
                COL_RELATIVE_PATH: rel,
                COL_FILE_NAME: fname,
                COL_EXTENSION: os.path.splitext(fname)[1].lower(),
                COL_SIZE_BYTES: st.st_size,
                COL_MTIME_NS: st.st_mtime_ns,
                COL_CATEGORY: category,
            })
    rows.sort(key=lambda r: r[COL_RELATIVE_PATH])
    return rows, issues
