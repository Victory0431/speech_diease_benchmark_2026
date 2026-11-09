"""sddbench command-line interface.

Stage-0 minimal foundation + D1 scope: inspect-file / stats / dataset-report.
Unimplemented later-stage commands (prepare, split, run, sweep, cache, train,
dashboard) are listed in --help as "not implemented yet" and refuse to run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from . import __version__
from .config import ConfigError, dataset_root, load_paths, resolve_under, workspace_dir
from .data.adapters.registry import get_adapter
from .data.adapters.uaspeech import parse_stem  # noqa: F401  (used via adapter)
from .data.audit import audit_config_defaults, run_audit
from .data.decoders import DecodeError, decode, probe, tool_versions
from .data.scan import scan_files
from .data.schema import (
    CATEGORY_AUDIO,
    COL_BLOCK_TOKEN,
    COL_CATEGORY,
    COL_DECODE_STATUS,
    COL_DIRECTORY_GROUP,
    COL_DURATION_DECODED_SEC,
    COL_FILE_NAME,
    COL_MIC_TOKEN,
    COL_N_SAMPLES,
    COL_PARSE_STATUS,
    COL_PROBE_STATUS,
    COL_PROMPT_TOKEN,
    COL_RELATIVE_PATH,
    COL_REVIEW_STATUS,
    COL_SAMPLE_RATE,
    COL_SHA256,
    COL_SPEAKER_TOKEN,
    FILES_COLUMNS,
    ISSUE_COLUMNS,
)
from .data.stats import collect_statistics
from .data.prepare import stratified_sample
from .data.windows import (
    WINDOW_COLUMNS,
    PreprocessConfigError,
    build_view_index,
    normalize_preprocess,
    preprocess_fingerprint,
)
from .reporting.dataset_report import build_report
from .reporting.job_report import write_inspection

NOT_IMPLEMENTED = ("prepare", "split", "run", "sweep", "cache", "train", "summarize", "dashboard")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def _int_or_none(value):
    try:
        if value is None:
            return None
        v = float(value)
        return None if v != v else int(v)  # NaN check without importing math
    except (TypeError, ValueError):
        return None


def _load_yaml(path: str | Path) -> dict:
    import yaml
    p = Path(path).expanduser()
    if not p.is_file():
        raise ConfigError(f"yaml config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_task_config(repo_root: Path, dataset_id: str, task_path: str | None) -> dict:
    import yaml
    if task_path:
        cfg_file = Path(task_path)
    else:
        cfg_file = repo_root / "configs" / "datasets" / f"{dataset_id}_v1.yaml"
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {"task_id": f"{dataset_id}_v1", "dataset_id": dataset_id,
               "adapter": dataset_id, "admission": "pending_review"}
    cfg["_workspace_root"] = None  # filled by caller after paths are known
    return cfg


def _dir_fingerprint(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x[COL_RELATIVE_PATH]):
        h.update(f"{r[COL_RELATIVE_PATH]}|{r.get('size_bytes')}|{r.get('mtime_ns')}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------- commands

def cmd_inspect_file(args) -> int:
    paths = load_paths(args.paths)
    root = dataset_root(paths, args.dataset)
    target = resolve_under(root, args.relative_path)
    if not target.is_file():
        raise ConfigError(f"file not found under dataset root: {args.relative_path}")

    info = probe(target)
    print(f"# inspect-file: {args.dataset} :: {args.relative_path}")
    print(f"absolute path (read-only): {target}")
    summary = {
        "dataset": args.dataset,
        "relative_path": args.relative_path,
        "size_bytes": target.stat().st_size,
        "probe": {
            "status": info.status, "backend": info.backend,
            "container": info.container, "encoding": info.encoding,
            "audio_codec": info.audio_codec,
            "sample_rate": info.sample_rate, "channels": info.channels,
            "frames": info.frames, "duration_header_sec": info.duration_sec,
            "warnings": info.warnings,
        },
    }
    if info.status.startswith("ok"):
        try:
            dec = decode(target)
            wave = dec.waveform
            import numpy as np
            summary["decode"] = {
                "backend": dec.backend, "sample_rate": dec.sample_rate,
                "n_samples_per_channel": int(wave.shape[1]),
                "duration_decoded_sec": float(wave.shape[1] / dec.sample_rate),
                "peak": float(np.max(np.abs(wave))) if wave.size else None,
                "rms": float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))) if wave.size else None,
                "shape_channels": int(wave.shape[0]),
            }
        except DecodeError as e:
            summary["decode"] = {"status": "failed", "error": str(e)}
    stem = Path(args.relative_path).stem
    summary["naming_parse"] = parse_stem(stem) or {"status": "parse_failed", "stem": stem}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("\n注意：naming_parse 中的 token 为候选解析事实，不构成标签核验。")
    return 0


def cmd_stats(args) -> int:
    t0 = time.time()
    paths = load_paths(args.paths)
    root = dataset_root(paths, args.dataset)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    task_cfg = _load_task_config(repo_root, args.dataset, args.task)
    task_cfg["_workspace_root"] = str(paths["workspace_root"])

    workers = args.workers or min(16, os.cpu_count() or 4)
    quality_cfg = audit_config_defaults()

    print(f"[stats] dataset={args.dataset} mode={args.mode} root={root}")
    rows, scan_issues = scan_files(root)
    print(f"[stats] discovered {len(rows)} files ({time.time() - t0:.1f}s)")

    adapter = get_adapter(task_cfg.get("adapter", args.dataset))
    adapter_result = adapter.collect(root, task_cfg, rows)
    index = {r[COL_RELATIVE_PATH]: r for r in rows}
    for rel, fields in adapter_result.enrichment.items():
        if rel in index:
            index[rel].update(fields)
    issues = [dict({"issue_type": i.get("issue_type"), "severity": i.get("severity"),
                    "relative_path": i.get("relative_path"), "detail": i.get("detail")})
              for i in (scan_issues + adapter_result.issues)]

    fingerprint = _dir_fingerprint(rows)
    if getattr(args, "scan_id", None):
        # explicit resume: reuse the existing scan directory and its progress.jsonl
        scan_id = args.scan_id
        scan_dir = workspace_dir(paths, "inventory", args.dataset, scan_id, create=False)
        if not (scan_dir / "meta.json").exists():
            raise ConfigError(f"--scan-id not found or incomplete: {scan_dir}")
        old_meta = json.loads((scan_dir / "meta.json").read_text(encoding="utf-8"))
        if old_meta.get("mode") != args.mode:
            raise ConfigError(f"mode mismatch for scan {scan_id}: "
                              f"saved={old_meta.get('mode')} requested={args.mode}")
        if old_meta.get("directory_fingerprint") != fingerprint:
            raise ConfigError(f"dataset content changed since scan {scan_id} "
                              f"(directory fingerprint mismatch); start a new scan")
        print(f"[stats] resuming scan_id={scan_id}")
    else:
        scan_id = f"{_stamp()}_{args.dataset}_{args.mode}_{fingerprint[:8]}"
        scan_dir = workspace_dir(paths, "inventory", args.dataset, scan_id)
        print(f"[stats] scan_id={scan_id}")
    print(f"[stats] output dir: {scan_dir}")

    meta = {
        "scan_id": scan_id, "dataset_id": args.dataset, "mode": args.mode,
        "dataset_root": str(root), "status": "running",
        "started_at": _now_iso(), "finished_at": None,
        "tools": tool_versions(), "quality_cfg": quality_cfg,
        "directory_fingerprint": fingerprint, "workers": workers,
        "task_config": {k: v for k, v in task_cfg.items() if not k.startswith("_")},
    }
    (scan_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    enriched, completed = run_audit(root, rows, args.mode, scan_dir, workers, quality_cfg)
    if not completed:
        meta["status"] = "partial"
        meta["interrupted_at"] = _now_iso()
        (scan_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        print("[stats] interrupted; partial results saved. Re-run the same command to resume.")
        return 130

    statistics = collect_statistics(enriched)
    statistics["schema_version"] = "1.0"
    statistics["scan_id"] = scan_id
    statistics["scope"] = {"mode": args.mode, "dataset_id": args.dataset}

    df = pd.DataFrame(enriched, columns=FILES_COLUMNS)
    df.to_csv(scan_dir / "files.csv", index=False)
    (scan_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    issues_df = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    issues_df.to_csv(scan_dir / "issues.csv", index=False)

    grp_rows = []
    for g, v in statistics["groups"]["by_directory_group"].items():
        for k2, v2 in v["duration_sec"].items():
            grp_rows.append({"group": g, "metric": f"duration_{k2}", "value": v2})
    for key, v in statistics["groups"]["by_directory_and_mic"].items():
        g, mic = key.split("|", 1)  # keys are "directory_group|mic_token"
        grp_rows.append({"group": g, "metric": f"mic_{mic}_n_files", "value": v["n_files"]})
    pd.DataFrame(grp_rows).to_csv(scan_dir / "group_statistics.csv", index=False)

    lc = pd.DataFrame(statistics["length_candidates"])
    lc.to_csv(scan_dir / "length_candidates.csv", index=False)
    if statistics["length_candidates_by_group"]:
        pd.DataFrame(statistics["length_candidates_by_group"]).to_csv(
            scan_dir / "length_candidates_by_group.csv", index=False)

    meta["status"] = "completed"
    meta["finished_at"] = _now_iso()
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    (scan_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    report_path = build_report(scan_dir, args.dataset)
    c = statistics["counts"]
    print(f"[stats] done in {time.time() - t0:.1f}s: decoded {c['decoded_ok']}/{c['audio_candidates']}, "
          f"failed {c['decode_failed']}, parse_failed {c['parse_failed']}")
    print(f"[stats] report: {report_path}")
    print(f"[stats] next: sddbench dataset-report --dataset {args.dataset} --scan {scan_id} "
          f"--paths {args.paths}")
    return 0


def cmd_dataset_report(args) -> int:
    paths = load_paths(args.paths)
    scan_dir = workspace_dir(paths, "inventory", args.dataset, args.scan, create=False)
    if not (scan_dir / "statistics.json").exists():
        raise ConfigError(f"scan not found or incomplete: {scan_dir}")
    report_path = build_report(scan_dir, args.dataset)
    print(f"[dataset-report] rebuilt: {report_path}")
    return 0


def cmd_extract_features(args) -> int:
    """Build a deterministic view/window index from a saved D1 scan.

    D2 scope: channel selection, crop, optional resample, padding and
    inspection previews. No MFCC yet -- asking for `--features` refuses instead
    of pretending features were computed.
    """
    if args.features:
        print("CONFIG ERROR: --features (MFCC) is not implemented in this stage "
              "(D2 builds the view/window index only). Refusing to fake feature output.",
              file=sys.stderr)
        return 2

    t0 = time.time()
    paths = load_paths(args.paths)
    root = dataset_root(paths, args.dataset)

    pre_raw = _load_yaml(args.preprocess)
    try:
        cfg = normalize_preprocess(pre_raw)
    except PreprocessConfigError as e:
        raise ConfigError(f"invalid preprocessing config {args.preprocess}: {e}") from e
    cfg["name"] = str(pre_raw.get("name") or Path(args.preprocess).stem)
    cfg["description"] = str(pre_raw.get("description") or "")

    scan_dir = workspace_dir(paths, "inventory", args.dataset, args.source, create=False)
    meta_file = scan_dir / "meta.json"
    if not meta_file.is_file():
        raise ConfigError(f"--source scan not found: {scan_dir}")
    scan_meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if scan_meta.get("status") != "completed":
        raise ConfigError(f"source scan {args.source} is {scan_meta.get('status')!r}, "
                          f"not completed; finish or resume it first")
    if scan_meta.get("mode") != "full":
        raise ConfigError(f"source scan {args.source} is mode={scan_meta.get('mode')!r}; "
                          f"view building needs full-mode decode facts")
    files_csv = scan_dir / "files.csv"
    if not files_csv.is_file():
        raise ConfigError(f"source scan has no files.csv (regenerate it): {files_csv}")

    cols = [COL_RELATIVE_PATH, COL_FILE_NAME, COL_CATEGORY, COL_DIRECTORY_GROUP,
            COL_SPEAKER_TOKEN, COL_BLOCK_TOKEN, COL_PROMPT_TOKEN, COL_MIC_TOKEN,
            COL_REVIEW_STATUS, COL_PARSE_STATUS, COL_DECODE_STATUS, COL_SHA256,
            COL_SAMPLE_RATE, COL_N_SAMPLES, COL_DURATION_DECODED_SEC]
    missing = [c for c in cols if c not in pd.read_csv(files_csv, nrows=0).columns]
    if missing:
        raise ConfigError(f"files.csv missing columns {missing}; regenerate the scan")
    df = pd.read_csv(files_csv, usecols=cols)
    all_rows = df.to_dict("records")
    print(f"[extract-features] source={args.source} rows={len(all_rows)}")

    chosen, selection = stratified_sample(all_rows, args.limit)

    sr_inventory = Counter(
        sr for sr in (_int_or_none(r.get(COL_SAMPLE_RATE)) for r in all_rows
                      if r.get(COL_DECODE_STATUS) == "ok") if sr is not None)
    sr_selected = Counter(
        sr for sr in (_int_or_none(r.get(COL_SAMPLE_RATE)) for r in chosen) if sr is not None)
    mixed = len(sr_selected) > 1
    blocked = (cfg["sample_rate_policy"] == "native"
               and not cfg["allow_mixed_native"] and mixed)

    cfg_fp = preprocess_fingerprint(cfg)
    job_id = (f"{_stamp()}_{args.dataset}_views_{cfg['position']}_"
              f"{float(cfg['observe_seconds']):g}s_{cfg_fp[:8]}")
    job_dir = workspace_dir(paths, "feature_jobs", job_id)
    print(f"[extract-features] job_id={job_id}")
    print(f"[extract-features] output dir: {job_dir}")

    preview_limit = max(0, int(args.previews or 0))
    preview_dir = (job_dir / "previews") if preview_limit and not blocked else None

    table = None
    if blocked:
        print(f"[extract-features] BLOCKED: sample_rate_policy=native with mixed selected Hz "
              f"{dict(sr_selected)} and allow_mixed_native=false. "
              f"Choose resample(target_sample_rate=...) or enable mixed-native explicitly.")
        view_rows, issues, counts = [], [{
            "issue_type": "blocked_mixed_native_sample_rate", "severity": "error",
            "relative_path": "",
            "detail": f"selected sample rates {dict(sr_selected)}; policy=native, "
                      f"allow_mixed_native=false",
        }], {"requested": len(chosen), "built": 0, "blocked": True}
        frontend = {}
    else:
        table = build_view_index(
            root, chosen, cfg,
            source_scan_id=args.source,
            preview_dir=preview_dir,
            preview_limit=preview_limit,
            verify_source_hash=not args.no_verify_source_hash,
        )
        view_rows, issues, counts = table.rows, table.issues, table.counts
        frontend = table.frontend_by_sample_rate
        print(f"[extract-features] windows built={counts['built']}/{counts['requested']} "
              f"({time.time() - t0:.1f}s)")

    status = ("blocked_mixed_sample_rate" if blocked
              else ("planned" if counts.get("built") else "failed_no_windows"))

    job = {
        "job_id": job_id,
        "dataset_id": args.dataset,
        "dataset_root": str(root),
        "job_dir": str(job_dir),
        "source": {"kind": "scan", "scan_id": args.source,
                   "mode": scan_meta.get("mode"), "status": scan_meta.get("status"),
                   "files_csv": str(files_csv)},
        "technical_only": True,
        "admission": (scan_meta.get("task_config") or {}).get("admission", "pending_review"),
        "preprocess": cfg,
        "preprocess_fingerprint": cfg_fp,
        "feature": None,
        "cache_mode": args.cache_mode,
        "selection": selection,
        "sample_rate_inventory": {str(k): v for k, v in sorted(sr_inventory.items())},
        "sample_rate_selected": {str(k): v for k, v in sorted(sr_selected.items())},
        "frontend_by_sample_rate": frontend,
        "counts": counts,
        "status": status,
        "dry_run": bool(args.dry_run),
        "verify_source_hash": not args.no_verify_source_hash,
        "outputs": {
            "windows_csv": str(job_dir / "windows.csv"),
            "issues_csv": str(job_dir / "issues.csv"),
            "inspection_md": str(job_dir / "inspection.md"),
            "config_yaml": str(job_dir / "config.yaml"),
            "previews_dir": str(preview_dir) if preview_dir else None,
        },
        "tools": tool_versions(),
        "code_version": __version__,
        "started_at": _now_iso(),
        "finished_at": _now_iso(),
        "elapsed_sec": round(time.time() - t0, 1),
        # private, used only for rendering
        "_window_rows": view_rows,
        "_issues": issues,
    }

    import yaml
    (job_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    pd.DataFrame(view_rows, columns=WINDOW_COLUMNS).to_csv(
        job_dir / "windows.csv", index=False)
    pd.DataFrame(issues, columns=ISSUE_COLUMNS).to_csv(job_dir / "issues.csv", index=False)
    write_inspection(job, job_dir / "inspection.md")
    public = {k: v for k, v in job.items() if not k.startswith("_")}
    (job_dir / "job.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[extract-features] status={status}")
    print(f"[extract-features] inspection: {job_dir / 'inspection.md'}")
    print(f"[extract-features] windows:    {job_dir / 'windows.csv'}")
    if blocked:
        print("[extract-features] next: switch sample_rate_policy to resample with a "
              "locked target_sample_rate, or set allow_mixed_native=true explicitly.")
        return 3
    if not counts.get("built"):
        print("[extract-features] no windows built; inspect issues.csv before retrying.")
        return 1
    print(f"[extract-features] next: review inspection.md and previews, then "
          f"compute MFCC for these views (D3).")
    return 0


def cmd_not_implemented(args) -> int:
    print(f"command '{args._cmd}' is NOT implemented yet (planned for a later stage; "
          f"see docs/construction_plan.md and docs/data_layer_plan.md). Refusing to fake success.")
    return 2


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sddbench",
                                description="Speech disease recognition dataset benchmark")
    p.add_argument("--version", action="version", version=f"sddbench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("inspect-file", help="probe + decode one file, parse its naming")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--relative-path", required=True)
    sp.add_argument("--paths", required=True)
    sp.set_defaults(func=cmd_inspect_file)

    sp = sub.add_parser("stats", help="scan + audit + statistics for a dataset (quick/full)")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--mode", choices=["quick", "full"], default="full")
    sp.add_argument("--paths", required=True)
    sp.add_argument("--task", default=None, help="dataset task config yaml (optional)")
    sp.add_argument("--workers", type=int, default=None)
    sp.add_argument("--scan-id", default=None,
                    help="resume this scan id in place (keeps progress.jsonl)")
    sp.add_argument("--repo-root", default=None, help="repo root holding configs/ (default: CWD)")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("dataset-report", help="rebuild report/figures from a saved scan")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--scan", required=True)
    sp.add_argument("--paths", required=True)
    sp.set_defaults(func=cmd_dataset_report)

    sp = sub.add_parser(
        "extract-features",
        help="build a deterministic view/window index (and later MFCC) from a saved scan")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--source", required=True, help="source scan id (full mode)")
    sp.add_argument("--preprocess", required=True, help="preprocessing config yaml")
    sp.add_argument("--features", default=None,
                    help="feature config yaml (reserved for the MFCC stage)")
    sp.add_argument("--limit", type=int, default=None,
                    help="deterministic stratified sample size (omit for all rows)")
    sp.add_argument("--cache-mode", choices=["run", "shared"], default="run")
    sp.add_argument("--dry-run", action="store_true",
                    help="build the view index/inspection only; do not compute MFCC")
    sp.add_argument("--previews", type=int, default=6,
                    help="number of derived preview clips to export (0 disables)")
    sp.add_argument("--no-verify-source-hash", action="store_true",
                    help="skip re-checking original content against the scan sha256")
    sp.add_argument("--paths", required=True)
    sp.add_argument("--repo-root", default=None)
    sp.set_defaults(func=cmd_extract_features)

    for name in NOT_IMPLEMENTED:
        sp = sub.add_parser(name, help=f"[not implemented yet] planned {name} command")
        sp.add_argument("rest", nargs="*")
        sp.set_defaults(func=cmd_not_implemented, _cmd=name)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
