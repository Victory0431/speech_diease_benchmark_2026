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
from .data.decoders import DecodeError, decode, file_sha256, probe, tool_versions
from .data.scan import scan_files
from .data.schema import (
    CATEGORY_AUDIO,
    COL_BLOCK_TOKEN,
    COL_CATEGORY,
    COL_CHANNELS,
    COL_DECODE_STATUS,
    COL_DIRECTORY_GROUP,
    COL_DURATION_DECODED_SEC,
    COL_FILE_NAME,
    COL_LABEL_RAW,
    COL_MIC_TOKEN,
    COL_N_SAMPLES,
    COL_ORIGIN_ID,
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
    MANIFEST_COLUMNS,
)
from .data.stats import collect_statistics
from .data.prepare import eligible_rows, stratified_sample
from .data.split import SplitError, assign_splits
from .reporting.split_report import render_split
from .data.admission import (
    AdmissionError,
    STATUS_APPROVED,
    STATUS_LABEL_REVIEW_INCOMPLETE,
    STATUS_LABEL_REVIEW_MISSING,
    STATUS_MIC_NOT_SELECTED,
    STATUS_NO_ROWS,
    STATUS_ORIGIN_CONFLICT,
    STATUS_ORIGIN_DUPLICATE,
    STATUS_SUBJECT_CONFLICT,
    build_manifest,
    load_label_review,
    manifest_fingerprint,
    review_inventory,
    review_status_rows,
    select_rows,
    validate_review,
    write_review_template,
)
from .reporting.admission_report import render_admission
from .data.windows import (
    WINDOW_COLUMNS,
    PreprocessConfigError,
    build_view_index,
    normalize_preprocess,
    preprocess_fingerprint,
)
from .reporting.dataset_report import build_report
from .reporting.job_report import write_inspection
from .modeling.experiment import (
    ExperimentConfigError,
    load_experiment,
    set_override,
)
from .modeling.features import FeatureLoadError
from .modeling.trainer import RunError, load_run_data, run_experiment
from .features.cache import FeatureCacheManager
from .features.mfcc import (
    FeatureConfigError,
    FeatureError,
    extract_mfcc,
    implementation_info,
    normalize_feature_config,
)

NOT_IMPLEMENTED = ("sweep", "cache", "train", "summarize", "dashboard")


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


def _read_scan(files_csv: Path, required: list[str]) -> list[dict]:
    missing = [c for c in required if c not in pd.read_csv(files_csv, nrows=0).columns]
    if missing:
        raise ConfigError(f"files.csv missing columns {missing}; regenerate the scan")
    return pd.read_csv(files_csv, usecols=required).to_dict("records")


def cmd_prepare(args) -> int:
    """D4: label review + mic/origin rules -> approved, person-isolated manifest."""
    t0 = time.time()
    paths = load_paths(args.paths)
    root = dataset_root(paths, args.dataset)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    task_cfg = _load_task_config(repo_root, args.dataset, args.task)
    task_id = str(task_cfg.get("task_id") or f"{args.dataset}_v1")
    selection = dict(task_cfg.get("selection") or {})
    if args.mic:
        selection["mic"] = args.mic

    scan_dir = workspace_dir(paths, "inventory", args.dataset, args.source, create=False)
    meta_file = scan_dir / "meta.json"
    if not meta_file.is_file():
        raise ConfigError(f"--source scan not found: {scan_dir}")
    scan_meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if scan_meta.get("status") != "completed" or scan_meta.get("mode") != "full":
        raise ConfigError(f"source scan {args.source} must be full/completed "
                          f"(got mode={scan_meta.get('mode')!r}, "
                          f"status={scan_meta.get('status')!r})")
    files_csv = scan_dir / "files.csv"
    if not files_csv.is_file():
        raise ConfigError(f"source scan has no files.csv: {files_csv}")

    required = [COL_RELATIVE_PATH, COL_CATEGORY, COL_DECODE_STATUS, COL_PARSE_STATUS,
                COL_DIRECTORY_GROUP, COL_SPEAKER_TOKEN, COL_ORIGIN_ID, COL_MIC_TOKEN,
                COL_LABEL_RAW, COL_SAMPLE_RATE, COL_CHANNELS, COL_DURATION_DECODED_SEC,
                COL_SHA256]
    all_rows = _read_scan(files_csv, required)
    inventory = review_inventory(all_rows)
    print(f"[prepare] dataset={args.dataset} task={task_id} source={args.source} "
          f"eligible={len(eligible_rows(all_rows))} speakers={len(inventory)}")

    review_rel = (task_cfg.get("label_review") or {}).get(
        "file", f"label_review/{args.dataset}_label_review.csv")
    review_path = (Path(args.review).expanduser() if args.review
                   else Path(paths["workspace_root"]) / str(review_rel))
    review_path = review_path.resolve()

    if args.write_review_template:
        try:
            write_review_template(review_path, inventory, force=args.force)
        except AdmissionError as e:
            raise ConfigError(str(e)) from e
        print(f"[prepare] wrote pending review template: {review_path}")

    review_exists = review_path.is_file()
    try:
        review = load_label_review(review_path)
    except AdmissionError as e:
        raise ConfigError(str(e)) from e

    selected, select_issues, select_blockers = select_rows(all_rows, selection)
    selected_inventory = review_inventory(selected)
    check = validate_review(review, selected_inventory)

    blockers: list[dict] = list(select_issues)
    if not review_exists:
        status = STATUS_LABEL_REVIEW_MISSING
        blockers.insert(0, {
            "issue_type": "label_review_missing", "severity": "error", "relative_path": "",
            "detail": f"no confirmed review file at {review_path}; run with "
                      "--write-review-template, fill evidence/labels, then rerun"})
    elif check["missing"] or check["unconfirmed"] or check["incomplete"]:
        status = STATUS_LABEL_REVIEW_INCOMPLETE
        blockers = check["blocking"] + blockers
    elif check["subject_conflicts"] or check["subject_aliases"]:
        status = STATUS_SUBJECT_CONFLICT
        blockers = check["blocking"] + blockers
    elif STATUS_NO_ROWS in select_blockers:
        status = STATUS_NO_ROWS
    elif STATUS_MIC_NOT_SELECTED in select_blockers:
        status = STATUS_MIC_NOT_SELECTED
    elif STATUS_ORIGIN_DUPLICATE in select_blockers:
        status = STATUS_ORIGIN_DUPLICATE
    else:
        status = STATUS_APPROVED

    manifest: list[dict] = []
    manifest_fp = ""
    if status == STATUS_APPROVED:
        manifest, manifest_issues = build_manifest(
            selected, check["confirmed"], args.dataset, review_path, selection)
        errors = [i for i in manifest_issues if i.get("severity") == "error"]
        if errors:
            status = STATUS_ORIGIN_CONFLICT
            blockers = errors + blockers
        elif not manifest:
            status = STATUS_NO_ROWS
            manifest = []
        else:
            manifest_fp = manifest_fingerprint(manifest)

    base_job_id = f"{_stamp()}_{args.dataset}_prepare"
    job_id = base_job_id
    suffix = 0
    while workspace_dir(paths, "prepared", task_id, job_id, create=False).exists():
        suffix += 1
        job_id = f"{base_job_id}_{suffix:02d}"
    out_dir = workspace_dir(paths, "prepared", task_id, job_id)
    print(f"[prepare] status={status}")
    print(f"[prepare] output dir: {out_dir}")

    review_rows = review_status_rows(inventory, review)
    label_counts = Counter(m["label"] for m in manifest)
    counts = {
        "n_eligible": len(eligible_rows(all_rows)),
        "n_selected": len(selected),
        "n_speakers": len(inventory),
        "n_selected_speakers": len(selected_inventory),
        "n_manifest": len(manifest),
        "n_subjects": len({m["subject_id"] for m in manifest}),
        "n_origins": len({m["origin_id"] for m in manifest}),
    }
    job = {
        "job_id": job_id, "task_id": task_id, "dataset_id": args.dataset,
        "dataset_root": str(root), "scan_id": args.source,
        "scan_dir": str(scan_dir), "out_dir": str(out_dir),
        "status": status, "selection": selection,
        "review_path": str(review_path), "review_exists": review_exists,
        "review": {
            "n_inventory": len(inventory),
            "n_confirmed": len(check["confirmed"]),
            "missing": check["missing"],
            "unconfirmed": check["unconfirmed"],
            "incomplete": check["incomplete"],
            "subject_conflicts": check["subject_conflicts"],
            "subject_aliases": check["subject_aliases"],
        },
        "review_rows": review_rows,
        "blockers": blockers,
        "counts": counts,
        "label_counts": dict(sorted(label_counts.items())),
        "manifest_fingerprint": manifest_fp,
        "started_at": _now_iso(), "finished_at": _now_iso(),
        "elapsed_sec": round(time.time() - t0, 3),
    }

    pd.DataFrame(review_rows, columns=[
        "speaker_token", "directory_group", "n_files", "n_origins", "mic_tokens",
        "review_status", "confirmed_subject_id", "confirmed_label", "evidence",
    ]).to_csv(out_dir / "label_review_status.csv", index=False)
    pd.DataFrame(blockers, columns=ISSUE_COLUMNS).to_csv(out_dir / "issues.csv", index=False)
    (out_dir / "admission.md").write_text(render_admission(job), encoding="utf-8")
    (out_dir / "admission.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    import yaml
    task_out = {k: v for k, v in task_cfg.items() if not k.startswith("_")}
    (out_dir / "task.yaml").write_text(
        yaml.safe_dump(task_out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    provenance = {
        "dataset_id": args.dataset, "task_id": task_id,
        "scan_id": args.source, "scan_meta": str(meta_file),
        "dataset_root": str(root), "review_file": str(review_path),
        "review_sha256": (file_sha256(review_path) if review_exists else None),
        "selection": selection, "generated_at": _now_iso(),
        "tool_version": __version__, "status": status,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    if status == STATUS_APPROVED:
        pd.DataFrame(manifest, columns=MANIFEST_COLUMNS).to_csv(
            out_dir / "manifest.csv", index=False)
        print(f"[prepare] APPROVED manifest rows={len(manifest)} "
              f"subjects={counts['n_subjects']} labels={job['label_counts']}")
        return 0

    print(f"[prepare] BLOCKED: {len(blockers)} blocker(s). "
          f"No manifest.csv written; see {out_dir / 'admission.md'}")
    for b in blockers[:5]:
        print(f"  - {b['issue_type']}: {b['detail'][:180]}")
    return 3


def cmd_split(args) -> int:
    """D5: deterministic person-disjoint, label-stratified split."""
    paths = load_paths(args.paths)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    task_cfg = _load_task_config(repo_root, args.dataset, args.task)
    task_id = str(task_cfg.get("task_id") or f"{args.dataset}_v1")
    prepared_dir = workspace_dir(paths, "prepared", task_id, args.prepared, create=False)
    manifest_file = prepared_dir / "manifest.csv"
    if not manifest_file.is_file():
        raise ConfigError(f"approved manifest not found: {manifest_file}")
    manifest = pd.read_csv(manifest_file).to_dict("records")
    fractions = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
    try:
        rows, summary = assign_splits(manifest, args.seed, fractions)
    except SplitError as e:
        raise ConfigError(str(e)) from e

    admission_file = prepared_dir / "admission.json"
    manifest_fp = ""
    if admission_file.is_file():
        manifest_fp = json.loads(admission_file.read_text(encoding="utf-8")).get(
            "manifest_fingerprint", "")
    identity = json.dumps({"manifest": manifest_fp, "seed": args.seed,
                           "fractions": fractions}, sort_keys=True)
    split_id = f"{_stamp()}_seed{args.seed}_{hashlib.sha256(identity.encode()).hexdigest()[:8]}"
    out_dir = workspace_dir(paths, "splits", task_id, split_id)
    job = {
        "split_id": split_id, "task_id": task_id, "dataset_id": args.dataset,
        "prepared_id": args.prepared, "manifest": str(manifest_file),
        "manifest_fingerprint": manifest_fp, "seed": args.seed,
        "fractions": fractions, "summary": summary, "out_dir": str(out_dir),
        "created_at": _now_iso(), "code_version": __version__,
    }
    pd.DataFrame(rows, columns=["sample_id", "group_id", "label", "split"]).to_csv(
        out_dir / "split.csv", index=False)
    (out_dir / "split.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "split.md").write_text(render_split(job), encoding="utf-8")
    print(f"[split] id={split_id}")
    print(f"[split] output dir: {out_dir}")
    print(f"[split] samples={summary['n_samples']} subjects={summary['n_subjects']}")
    for split in ("train", "val", "test"):
        b = summary["by_split"][split]
        s = summary["subjects_by_split"][split]
        print(f"[split] {split:5s}: samples={b['n_samples']:6d} "
              f"subjects={s['n_subjects']:3d} labels={b['labels']}")
    return 0


def cmd_run(args) -> int:
    """D6: train the frozen-feature MLP and write a self-contained result pack."""
    paths = load_paths(args.paths)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    workspace = Path(paths["workspace_root"])
    try:
        cfg = load_experiment(args.config)
        for assignment in args.set or []:
            set_override(cfg, assignment)
        if args.device:
            cfg["training"]["device"] = args.device
        print(f"[run] experiment={cfg['experiment_id']} dataset={cfg['dataset']} "
              f"prepared={cfg['prepared_id']} split={cfg['split_id']}")
        if args.dry_run:
            data = load_run_data(cfg, workspace, repo_root)
            merged = data["merged"]
            print(f"[run] DRY-RUN: samples={len(merged)} "
                  f"subjects={merged['group_id'].nunique()} "
                  f"feature_dim={data['X'].shape[1]} "
                  f"sample_rate={data['sample_rate']} "
                  f"cache={data['cache_info']['feature_key']}")
            for split in ("train", "val", "test"):
                part = merged[merged["split"] == split]
                print(f"[run]   {split:5s}: samples={len(part):6d} "
                      f"subjects={part['group_id'].nunique():3d} "
                      f"labels={part['label'].value_counts().to_dict()}")
            print("[run] DRY-RUN: config, data and feature cache are valid; nothing trained.")
            return 0
        job = run_experiment(cfg, workspace, repo_root)
    except (ExperimentConfigError, FeatureLoadError, RunError) as e:
        raise ConfigError(str(e)) from e

    print(f"[run] completed: {job['run_id']}")
    print(f"[run] output dir: {job['out_dir']}")
    print(f"[run] best epoch={job['best_epoch']} "
          f"val_subject_bacc={job['best_val_subject_balanced_accuracy']:.3f}")
    test_m = job["metrics"]["test"]["subject"]
    print(f"[run] test subject: n={test_m['n']} "
          f"bacc={test_m['balanced_accuracy']} "
          f"sens={test_m['sensitivity']} spec={test_m['specificity']} "
          f"auroc={test_m['auroc']}")
    print(f"[run] report: {job['outputs']['report']}")
    return 0


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
    """Build a deterministic view/window index and (optionally) MFCC statistics.

    D2 scope: channel selection, crop, optional resample, padding and
    inspection previews. D3 adds the fixed MFCC recipe + sharded cache.
    """
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

    feat_cfg = None
    feat_impl = None
    if args.features:
        fraw = _load_yaml(args.features)
        try:
            feat_cfg = normalize_feature_config(fraw)
        except FeatureConfigError as e:
            raise ConfigError(f"invalid feature config {args.features}: {e}") from e
        feat_cfg["name"] = str(fraw.get("name") or Path(args.features).stem)
        feat_cfg["description"] = str(fraw.get("description") or "")
        try:
            feat_impl = implementation_info()
        except FeatureConfigError as e:
            raise ConfigError(str(e)) from e

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
    if getattr(args, "mic", None):
        before = len(all_rows)
        all_rows = [r for r in all_rows
                    if str(r.get(COL_MIC_TOKEN) or "").strip() == args.mic]
        print(f"[extract-features] mic filter {args.mic!r}: {len(all_rows)}/{before} rows")
        if not all_rows:
            raise ConfigError(f"--mic {args.mic!r} matched no rows in {files_csv}")
    print(f"[extract-features] source={args.source} rows={len(all_rows)}")

    chosen, selection = stratified_sample(all_rows, args.limit)
    selection["mic"] = getattr(args, "mic", None)

    sr_inventory = Counter(
        sr for sr in (_int_or_none(r.get(COL_SAMPLE_RATE)) for r in all_rows
                      if r.get(COL_DECODE_STATUS) == "ok") if sr is not None)
    sr_selected = Counter(
        sr for sr in (_int_or_none(r.get(COL_SAMPLE_RATE)) for r in chosen) if sr is not None)
    mixed = len(sr_selected) > 1
    blocked = (cfg["sample_rate_policy"] == "native"
               and not cfg["allow_mixed_native"] and mixed)

    cfg_fp = preprocess_fingerprint(cfg)
    base_job_id = (f"{_stamp()}_{args.dataset}_views_{cfg['position']}_"
                   f"{float(cfg['observe_seconds']):g}s_{cfg_fp[:8]}")
    job_id = base_job_id
    suffix = 0
    while workspace_dir(paths, "feature_jobs", job_id, create=False).exists():
        suffix += 1
        job_id = f"{base_job_id}_{suffix:02d}"
    job_dir = workspace_dir(paths, "feature_jobs", job_id)
    print(f"[extract-features] job_id={job_id}")
    print(f"[extract-features] output dir: {job_dir}")

    preview_limit = max(0, int(args.previews or 0))
    preview_dir = (job_dir / "previews") if preview_limit and not blocked else None

    compute_features = bool(feat_cfg) and not args.dry_run and not blocked
    manager = None
    if compute_features:
        if args.cache_mode == "run":
            cache_root = job_dir / "cache" / "features" / "mfcc_stats"
        else:
            cache_root = Path(paths["workspace_root"]) / "cache" / "features" / "mfcc_stats"
        manager = FeatureCacheManager(cache_root, feat_cfg, cache_mode=args.cache_mode,
                                      view_fingerprint=cfg_fp)
        print(f"[extract-features] feature cache ({args.cache_mode}): {cache_root}")

    feature_counts = Counter({"n_cached": 0, "n_new": 0, "n_failed": 0})
    feature_failures: list[dict] = []

    def _on_view(row, valid, processed_sr):
        cache = manager.cache_for(processed_sr)
        existing = cache.existing(row["window_id"])
        if existing is not None:
            if existing.get("view_hash") != row["view_hash"]:
                feature_counts["n_failed"] += 1
                feature_failures.append({
                    "issue_type": "feature_cache_collision", "severity": "error",
                    "relative_path": row["relative_path"],
                    "detail": f"window_id {row['window_id']} has a different view_hash"})
                return
            feature_counts["n_cached"] += 1
            return
        try:
            result = extract_mfcc(valid, processed_sr, feat_cfg, feat_impl)
            cache.add(row["window_id"], row["sample_id"], result.vector,
                      result.n_frames, row["view_hash"])
            feature_counts["n_new"] += 1
        except (FeatureError, FeatureConfigError) as e:
            feature_counts["n_failed"] += 1
            feature_failures.append({
                "issue_type": "feature_failed", "severity": "error",
                "relative_path": row["relative_path"], "detail": str(e)[:500]})

    table = None
    if blocked:
        print(f"[extract-features] BLOCKED: sample_rate_policy=native with mixed selected Hz "
              f"{dict(sr_selected)} and allow_mixed_native=false. "
              f"Choose resample(target_sample_rate=...) or enable mixed-native explicitly.")
        view_rows = []
        issues = [{
            "issue_type": "blocked_mixed_native_sample_rate", "severity": "error",
            "relative_path": "",
            "detail": f"selected sample rates {dict(sr_selected)}; policy=native, "
                      f"allow_mixed_native=false",
        }]
        counts = {"requested": len(chosen), "built": 0, "blocked": True}
        frontend = {}
    else:
        table = build_view_index(
            root, chosen, cfg,
            source_scan_id=args.source,
            preview_dir=preview_dir,
            preview_limit=preview_limit,
            verify_source_hash=not args.no_verify_source_hash,
            on_view=_on_view if manager is not None else None,
        )
        view_rows = table.rows
        issues = table.issues + feature_failures
        counts = table.counts
        frontend = table.frontend_by_sample_rate
        print(f"[extract-features] windows built={counts['built']}/{counts['requested']} "
              f"({time.time() - t0:.1f}s)")
        if manager is not None:
            print(f"[extract-features] features: new={feature_counts['n_new']} "
                  f"cached={feature_counts['n_cached']} failed={feature_counts['n_failed']}")

    feature_summary = None
    if feat_cfg:
        dim = 2 * int(feat_cfg["n_mfcc"])
        if manager is not None:
            finalized = manager.finalize()
            feature_summary = {
                "config": feat_cfg,
                "dim": dim,
                "cache_mode": args.cache_mode,
                "n_new": feature_counts["n_new"],
                "n_from_cache": feature_counts["n_cached"],
                "n_failed": feature_counts["n_failed"],
                "n_items": finalized["n_items"],
                "caches": finalized["caches"],
                "feature_hashes": {str(c["sample_rate"]): c["feature_hash"]
                                   for c in finalized["caches"]},
            }
        else:
            feature_summary = {
                "config": feat_cfg, "dim": dim, "cache_mode": args.cache_mode,
                "n_new": 0, "n_from_cache": 0, "n_failed": 0, "n_items": 0,
                "caches": [], "planned": True,
                "note": "dry-run/blocked: features were not computed",
            }

    if blocked:
        status = "blocked_mixed_sample_rate"
    elif not counts.get("built"):
        status = "failed_no_windows"
    elif args.dry_run or feat_cfg is None:
        status = "planned"
    elif feature_counts["n_failed"]:
        status = "completed_with_feature_errors"
    else:
        status = "completed"

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
        "features_requested": bool(feat_cfg),
        "feature": feature_summary,
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
            "cache_root": (str(manager.base_root) if manager is not None else None),
            "cache_dirs": ([c["cache_dir"] for c in (feature_summary or {}).get("caches", [])]
                           if feature_summary else []),
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
    if feat_cfg:
        (job_dir / "feature_config.yaml").write_text(
            yaml.safe_dump(feat_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
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
    if feature_summary and feature_summary.get("n_failed"):
        print(f"[extract-features] {feature_summary['n_failed']} feature failures; "
              f"see issues.csv and the cache index.")
        return 4
    if manager is None:
        print("[extract-features] next: review inspection.md and previews, then pass "
              "--features configs/features/mfcc20_stats.yaml to compute MFCC (D3).")
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
        "split",
        help="deterministic person-disjoint, label-stratified train/val/test split")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--prepared", required=True, help="approved prepared job id")
    sp.add_argument("--task", default=None)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--train-frac", type=float, default=0.6)
    sp.add_argument("--val-frac", type=float, default=0.2)
    sp.add_argument("--test-frac", type=float, default=0.2)
    sp.add_argument("--paths", required=True)
    sp.add_argument("--repo-root", default=None)
    sp.set_defaults(func=cmd_split)

    sp = sub.add_parser(
        "run",
        help="train the frozen-feature MLP and write a reproducible result pack")
    sp.add_argument("--config", required=True, help="experiment config yaml")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="dot-key override, e.g. model.dropout=0.3 (repeatable)")
    sp.add_argument("--device", default=None, help="override training.device (cpu/cuda)")
    sp.add_argument("--dry-run", action="store_true",
                    help="validate config/data/features; do not train")
    sp.add_argument("--paths", required=True)
    sp.add_argument("--repo-root", default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser(
        "prepare",
        help="label review + mic/origin rules -> approved, person-isolated manifest")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--source", required=True, help="source scan id (full mode)")
    sp.add_argument("--task", default=None, help="task config yaml (default configs/datasets/<dataset>_v1.yaml)")
    sp.add_argument("--review", default=None,
                    help="label review csv (default: label_review.file from the task config)")
    sp.add_argument("--mic", default=None,
                    help="microphone token to keep (overrides selection.mic)")
    sp.add_argument("--write-review-template", action="store_true",
                    help="create a pending review template for every observed speaker")
    sp.add_argument("--force", action="store_true",
                    help="replace an existing review template")
    sp.add_argument("--paths", required=True)
    sp.add_argument("--repo-root", default=None)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser(
        "extract-features",
        help="build a deterministic view/window index (and later MFCC) from a saved scan")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--source", required=True, help="source scan id (full mode)")
    sp.add_argument("--preprocess", required=True, help="preprocessing config yaml")
    sp.add_argument("--features", default=None,
                    help="feature config yaml (reserved for the MFCC stage)")
    sp.add_argument("--mic", default=None,
                    help="keep only this microphone token before sampling (e.g. M3)")
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
