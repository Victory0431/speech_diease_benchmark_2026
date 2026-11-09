"""Inspection report for a feature job (Markdown, no re-computation of stats).

Reads the window table + job metadata produced by `extract-features`; it never
decodes audio or invents values. The goal is a human-checkable plan before any
full extraction (data-layer doc §10).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

_MAX_TABLE_ROWS = 50


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _counts_table(counts: dict) -> str:
    lines = ["| 指标 | 值 |", "| --- | --- |"]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _window_table(rows: list[dict]) -> str:
    if not rows:
        return "_没有生成任何窗口。_"
    head = ("| sample_id | group | speaker | mic | 原时长(s) | start(s) | end(s) | "
            "有效(s) | 补零(s) | 补零比例 | 处理后Hz |")
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [head, sep]
    for r in rows[:_MAX_TABLE_ROWS]:
        lines.append(
            f"| {r['sample_id']} | {r.get('directory_group', '')} | "
            f"{r.get('speaker_token', '')} | {r.get('mic_token', '')} | "
            f"{_fmt(r.get('source_duration_sec'), 3)} | "
            f"{_fmt(r.get('original_start_sec'), 3)} | "
            f"{_fmt(r.get('original_end_sec'), 3)} | "
            f"{_fmt(r['valid_num_samples'] / r['processed_sample_rate'], 3)} | "
            f"{_fmt(r['pad_num_samples'] / r['processed_sample_rate'], 3)} | "
            f"{_fmt(r.get('padding_ratio'), 3)} | {r.get('processed_sample_rate')} |")
    if len(rows) > _MAX_TABLE_ROWS:
        lines.append(f"\n_仅显示前 {_MAX_TABLE_ROWS} 行，共 {len(rows)} 行；完整表见 windows.csv。_")
    return "\n".join(lines)


def render_inspection(job: dict) -> str:
    pre = job["preprocess"]
    sel = job.get("selection", {})
    counts = job.get("counts", {})
    rows = job.get("_window_rows", [])
    issues = job.get("_issues", [])
    issue_counts = Counter(i["issue_type"] for i in issues)

    lines = [
        f"# 特征作业预检：{job['job_id']}",
        "",
        f"- 数据集：`{job['dataset_id']}`｜源扫描：`{job['source']['scan_id']}`"
        f"（{job['source']['mode']}）｜状态：**{job['status']}**",
        f"- 技术验证：`technical_only={str(job.get('technical_only')).lower()}`；"
        f"标签状态：**未核验（pending）**，本作业不产生训练标签。",
        f"- 预处理配置：`{pre.get('name', '(inline)')}`｜profile：`{pre.get('profile')}`｜"
        f"fingerprint：`{job.get('preprocess_fingerprint', '')[:16]}…`",
        f"- 采样率策略：`{pre['sample_rate_policy']}`"
        + (f" → {pre['target_sample_rate']} Hz" if pre.get("target_sample_rate") else "")
        + f"｜通道：`{pre['channel_policy']}`(index={pre['channel_index']})"
        + f"｜位置：`{pre['position']}`｜窗口：{pre['window_seconds']} s",
        f"- 源文件路径（只读）：`{job['dataset_root']}`",
        f"- 输出目录：`{job['job_dir']}`",
        "",
        "## 1. 样本选择",
        "",
        f"- 策略：`{sel.get('strategy')}`｜合格样本：{sel.get('n_eligible')}｜"
        f"本次选取：{sel.get('n_selected')}（limit={sel.get('limit')}）",
        f"- 目录组分布：{sel.get('selected_by_group')}",
        f"- 涉及说话人数：{len(sel.get('selected_by_speaker') or {})}",
        "",
        "## 2. 处理计数",
        "",
        _counts_table(counts),
        "",
        "## 3. 采样率（源清单 vs 本次解码）",
        "",
        f"- 源清单分布：{job.get('sample_rate_inventory')}",
        f"- 本次选中分布：{job.get('sample_rate_selected')}",
        f"- 实际前端几何：{job.get('frontend_by_sample_rate')}",
        "",
        "## 4. 窗口索引（坐标均为原始采样点换算，补零不伪造静音）",
        "",
        _window_table(rows),
        "",
        "## 5. 问题清单",
        "",
    ]
    if issue_counts:
        lines.append("| issue_type | 数量 |")
        lines.append("| --- | --- |")
        for k, v in sorted(issue_counts.items()):
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("_无问题记录。_")
    if issues:
        lines.append("")
        lines.append("示例（最多 10 条）：")
        for i in issues[:10]:
            lines.append(f"- `{i['issue_type']}` [{i['severity']}] {i['relative_path']}: "
                         f"{i['detail']}")

    previews = [r["preview_path"] for r in rows if r.get("preview_path")]
    fs = job.get("feature")
    section = 6
    if fs:
        lines += ["", f"## {section}. MFCC 特征（仅有效声音帧）", ""]
        fcfg = fs.get("config") or {}
        lines += [
            f"- 配方：`{fcfg.get('name', 'mfcc_stats')}`｜n_mfcc={fcfg.get('n_mfcc')}"
            f"｜n_mels={fcfg.get('n_mels')}｜stats=mean+std(ddof={fcfg.get('std_ddof')})"
            f"｜log_mels=true(eps={fcfg.get('log_epsilon')})",
            f"- 输出维度：{fs.get('dim')}｜cache_mode：`{fs.get('cache_mode')}`"
            f"｜新增 {fs.get('n_new')} 条｜命中缓存 {fs.get('n_from_cache')} 条"
            f"｜失败 {fs.get('n_failed')} 条",
        ]
        for c in fs.get("caches", []):
            lines.append(f"- 缓存：`{c['cache_dir']}`｜{c['sample_rate']} Hz｜dim={c['dim']}"
                         f"｜{c['n_items']} 条 / {c['n_shards']} 个分片"
                         f"｜feature_hash=`{c['feature_hash'][:16]}…`")
        lines.append("- 补零段不进入 MFCC 统计；`n_frames` 仅按有效声音推导。")
        section += 1
    lines += [
        "",
        f"## {section}. 可审听派生片段",
        "",
        ("共 " + str(len(previews)) + f" 个，位于 `{job['outputs'].get('previews_dir', '')}`：\n"
         + "\n".join(f"- `{p}`" for p in previews))
        if previews else "_未导出派生音频（dry-run 预览数量为 0）。_",
        "",
        f"## {section + 1}. 边界与下一步",
        "",
        "- 补零只改变张量容纳长度；有效样本数 `valid_num_samples` 与补零数分开记录。",
        "- `too_short_for_frontend` 的样本被排除并记录，未私改帧长。",
        "- 标签全部为 `pending`（目录名不是确认标签）；进入正式训练前必须先完成 D4 准入。",
        f"- 下一步：{'先完成 D4 标签复核/准入，再导出训练清单' if fs else '用相同 view_hash 计算 MFCC 统计量；改位置/长度/采样率会使 view_hash 变化'}。",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_inspection(job: dict, path: Path) -> Path:
    path.write_text(render_inspection(job), encoding="utf-8")
    return path
