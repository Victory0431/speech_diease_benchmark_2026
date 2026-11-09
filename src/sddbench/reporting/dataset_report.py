"""Dataset statistics report: markdown + figures from saved scan results.

Reads statistics.json / files.csv produced by `sddbench stats`; never decodes
audio again and never invents its own quantiles. Figures are descriptive
assets for human review only (not training inputs).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _fmt(x, nd=3):
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _duration_table(dur: dict, include_total: bool = True) -> str:
    keys = ["n", "min", "p10", "p25", "p50", "p60", "p80", "p90", "p95", "p99",
            "max", "mean", "std", "total"]
    lines = ["| 指标 | 值 (秒) |", "| --- | --- |"]
    for k in keys:
        if k not in dur:
            continue
        v = dur[k]
        if k == "total":
            if not include_total:
                continue  # e.g. per-speaker medians: summing medians is meaningless
            lines.append(f"| total (小时) | {_fmt(v / 3600.0, 2)} |")
        else:
            lines.append(f"| {k} | {_fmt(v)} |")
    return "\n".join(lines)


def _grouped_duration_table(g: dict) -> str:
    lines = ["| 组 | n | p50 | p80 | p90 | p95 | max |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, v in g.items():
        if not v.get("n"):
            continue
        lines.append(f"| {k} | {v['n']} | {_fmt(v.get('p50'))} | {_fmt(v.get('p80'))} | "
                     f"{_fmt(v.get('p90'))} | {_fmt(v.get('p95'))} | {_fmt(v.get('max'))} |")
    return "\n".join(lines)


def _candidates_table(rows: list[dict]) -> str:
    cols = ["L_sec", "n_shorter_than_L", "ratio_shorter_than_L",
            "n_longer_than_L", "retention_ratio", "total_padding_ratio",
            "mean_per_recording_padding_ratio", "truncated_seconds_total"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(
            _fmt(r[c], 4) if isinstance(r[c], float) else str(r[c]) for c in cols
        ) + " |")
    return "\n".join([head, sep] + body)


def _count_table(d: dict, top: int = 15) -> str:
    lines = ["| 值 | 数量 | 占比 |", "| --- | --- | --- |"]
    for k, v in list(d.items())[:top]:
        ratio = v["ratio"]
        lines.append(f"| {k} | {v['count']} | {_fmt(ratio, 4) if ratio is not None else 'null'} |")
    return "\n".join(lines)


def _fig_duration(durs: np.ndarray, out: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(durs, bins=60, color="#4878a8")
    axes[0].set_xlabel("duration (s)"); axes[0].set_ylabel("files")
    axes[0].set_title("Histogram")
    axes[1].hist(durs, bins=200, density=True, cumulative=True, histtype="step", color="#a84848")
    axes[1].set_xlabel("duration (s)"); axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative distribution")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _fig_bar(labels, values, out: Path, title: str, xlabel: str, ylabel="files"):
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels) + 2), 4))
    ax.bar(range(len(labels)), values, color="#4878a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def build_report(scan_dir: Path, dataset_id: str) -> Path:
    """(Re)build report.md and figures/ from saved scan artifacts."""
    scan_dir = Path(scan_dir)
    statistics = json.loads((scan_dir / "statistics.json").read_text(encoding="utf-8"))
    meta = json.loads((scan_dir / "meta.json").read_text(encoding="utf-8"))
    files_csv = scan_dir / "files.csv"
    fig_dir = scan_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    counts = statistics["counts"]
    dur = statistics["duration_seconds"]["decoded_full"]
    hq = statistics["duration_seconds"]["header_estimate_quick"]
    scan_mode = meta.get("mode", "?")

    # figures
    if files_csv.exists():
        df = pd.read_csv(files_csv, usecols=["relative_path", "category", "decode_status",
                                             "duration_decoded_sec", "sample_rate",
                                             "directory_group", "mic_token", "container",
                                             "quality_peak", "quality_rms"])
        dec = df[df["decode_status"] == "ok"]
        if len(dec):
            _fig_duration(dec["duration_decoded_sec"].to_numpy(),
                          fig_dir / "duration_hist_cdf.png",
                          f"{dataset_id} decoded durations (n={len(dec)})")
        sr_counts = dec["sample_rate"].value_counts()
        if len(sr_counts):
            _fig_bar([str(k) for k in sr_counts.index], sr_counts.to_numpy(),
                     fig_dir / "sample_rates.png", "Sample rates (decoded)", "Hz")
        cont_counts = df["container"].fillna("unknown").value_counts().head(12)
        if len(cont_counts):
            _fig_bar([str(k) for k in cont_counts.index], cont_counts.to_numpy(),
                     fig_dir / "containers.png", "Actual containers (probe)", "container")
        mic = df.groupby(["directory_group", "mic_token"], dropna=False).size()
        if len(mic):
            labels = [f"{g}|{m}" for g, m in mic.index]
            _fig_bar(labels, mic.to_numpy(), fig_dir / "files_by_dir_mic.png",
                     "Files by directory_group × mic", "group|mic")
        if len(dec) and "quality_rms" in dec:
            _fig_duration(dec["quality_rms"].to_numpy(),
                          fig_dir / "rms_hist_cdf.png",
                          "Per-file RMS distribution (technical fact)")

    md = []
    md.append(f"# 数据集统计报告：{dataset_id}")
    md.append("")
    md.append(f"- scan_id: `{meta.get('scan_id')}`｜scan_mode: **{scan_mode}**｜status: {meta.get('status')}")
    md.append(f"- 数据根（只读）: `{meta.get('dataset_root')}`")
    md.append(f"- 开始: {meta.get('started_at')}｜结束: {meta.get('finished_at')}")
    md.append(f"- 工具: soundfile {meta.get('tools', {}).get('soundfile')} "
              f"(libsndfile {meta.get('tools', {}).get('libsndfile')})｜"
              f"ffprobe: {meta.get('tools', {}).get('ffprobe')}")
    md.append(f"- 目录指纹: `{meta.get('directory_fingerprint', '')[:16]}…`")
    md.append(f"- 覆盖: 扫描发现 {counts['total_discovered']} 文件；候选音频 "
              f"{counts['audio_candidates']}；probe OK {counts['probe_ok']}；"
              f"full 解码成功 {counts['decoded_ok']}；失败 {counts['decode_failed']}")
    md.append("")
    md.append("**这是技术统计报告，不构成任何标签核验或疾病判断。**")
    md.append("")

    md.append("## 1. 文件覆盖与格式")
    md.append("")
    md.append("| 类别 | 数量 |")
    md.append("| --- | --- |")
    for k, v in counts.items():
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append("### 实际容器（probe 结果，不只看扩展名）")
    md.append(_count_table(statistics["formats"]["containers"]))
    md.append("")
    md.append("### 编码/位深（lossy 格式无 PCM 位深，null 不编造）")
    md.append(_count_table(statistics["formats"]["encodings"]))
    md.append("")
    md.append(f"- 扩展名与实际容器不符的文件数：{counts['extension_mismatch']}")
    md.append("")

    md.append("## 2. 采样率与通道（解码实测）")
    md.append("")
    md.append(_count_table(statistics["sample_rates"]["decoded"]))
    md.append("")
    md.append(_count_table(statistics["channels"]["decoded"]))
    md.append("")

    md.append("## 3. 时长分布（等权每条录音；NumPy linear 分位数）")
    md.append("")
    if scan_mode == "full":
        md.append(_duration_table(dur))
        md.append("")
        md.append("每人中位时长分位数（防多录音者主导）：")
        md.append(_duration_table(statistics["duration_seconds"]["per_speaker_median"],
                                  include_total=False))
    else:
        md.append("**quick 模式：以下为 header/probe 估计，不能声称全部可解码。**")
        md.append(_duration_table(hq))
    md.append("")

    md.append("## 4. 分组概览（directory_group 是目录事实，不是确认标签）")
    md.append("")
    md.append("| directory_group | 文件数 | 解码OK | 解码失败 | 人数 | 中位时长(s) |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for g, v in statistics["groups"]["by_directory_group"].items():
        md.append(f"| {g} | {v['n_files']} | {v['n_decoded_ok']} | {v['n_decode_failed']} | "
                  f"{v['n_speakers']} | {_fmt(v['duration_sec'].get('p50'))} |")
    md.append("")
    md.append("### 词项类别（文件命名 token 统计）")
    md.append("")
    md.append("| 类别 | 文件数 |")
    md.append("| --- | --- |")
    for k, v in statistics["groups"]["prompt_classes"].items():
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append("### 时长分位数按词项类别（离群值抽查重点：max 远超 p90 的类别）")
    md.append("")
    md.append(_grouped_duration_table(statistics["groups"]["duration_by_prompt_class"]))
    md.append("")
    md.append("### 时长分位数按麦克风")
    md.append("")
    md.append(_grouped_duration_table(statistics["groups"]["duration_by_mic"]))
    md.append("")

    md.append("## 5. 长度候选表（供人工决策；不自动选定 L）")
    md.append("")
    md.append(_candidates_table(statistics["length_candidates"]))
    md.append("")
    md.append("分组（按目录组）的候选表见 `length_candidates.csv`。")
    md.append("")

    md.append("## 6. 信号质量摘要（技术事实，只告警不排除）")
    md.append("")
    q = statistics["quality"]
    for k in ("peak", "rms", "near_full_ratio"):
        md.append(f"- {k}: n={q[k].get('n')}，p50={_fmt(q[k].get('p50'))}，"
                  f"p95={_fmt(q[k].get('p95'))}，max={_fmt(q[k].get('max'))}")
    md.append(f"- 全零文件数: {q['n_all_zero']}；含非有限样本的文件数: {q['n_nonfinite_samples_files']}；"
              f"极短(<{meta.get('quality_cfg', {}).get('short_duration_sec', 0.1)}s)文件数: {q['n_extremely_short']}")
    md.append("")

    md.append("## 7. 异常与待核查")
    md.append("")
    anomalies = statistics["anomalies"]
    md.append(f"- probe 失败: {counts['probe_failed']}（示例见 `issues.csv`）")
    md.append(f"- 解码失败: {counts['decode_failed']}；原因分布: " +
              ", ".join(f"{k}×{v['count']}" for k, v in list(anomalies["decode_failed_reasons"].items())[:8]))
    md.append(f"- 文件名解析失败: {counts['parse_failed']}")
    md.append("- **标签疑点（必须人工核查后才可用于训练）**: C 前缀说话人与目录组的完全分离关系、"
              "mic 语义（M2-M8）、以及每人标签归属——见 `issues.csv` 与 docs/datasets/uaspeech.md。")
    md.append("")
    md.append("## 8. 人工决策点（D1 → D2 之间）")
    md.append("")
    md.append("1. 依据第 3/5 节决定观察长度 L（候选表仅为事实）。")
    md.append("2. 决定采样率策略：native 或统一 resample（含目标 Hz）。")
    md.append("3. 依据第 4 节 mic 覆盖决定首任务采用的 mic（缺麦克风的一组不能被悄悄筛光）。")
    md.append("4. 启动 label_review（D4）：核对 speaker_token 与官方语料角色，填 confirmed_label。")
    md.append("")

    report_path = scan_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path
