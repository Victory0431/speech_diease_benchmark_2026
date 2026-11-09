"""Render `admission.md` from a prepare job (facts already computed)."""
from __future__ import annotations


def _table(rows: list[dict], columns: list[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |",
           "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return out


def render_admission(job: dict) -> str:
    lines = [f"# 准入预检：{job['job_id']}", ""]
    lines.append(f"- 数据集：`{job['dataset_id']}`｜任务：`{job['task_id']}`")
    lines.append(f"- 源扫描：`{job['scan_id']}`（full / completed）")
    lines.append(f"- 状态：**{job['status']}**")
    lines.append(f"- 标签复核文件：`{job['review_path']}`"
                 f"（{'存在' if job.get('review_exists') else '缺失'}）")
    lines.append(f"- 输出目录：`{job['out_dir']}`")
    lines += ["", "## 1. 选择规则（来自任务配置）", ""]
    sel = job.get("selection", {})
    lines.append(f"- mic：`{sel.get('mic') or '(未指定)'}`"
                 f"｜每个 origin 只保留一条：`{sel.get('one_per_origin', True)}`")
    lines.append(f"- variant：`{sel.get('variant', 'original')}`"
                 f"｜elicitation：`{sel.get('elicitation', 'unknown')}`")
    counts = job.get("counts", {})
    lines.append(f"- 合格音频：{counts.get('n_eligible', 0)}"
                 f"｜应用规则后：{counts.get('n_selected', 0)}"
                 f"｜speaker 数：{counts.get('n_speakers', 0)}"
                 f"（其中参与本次清单 {counts.get('n_selected_speakers', 0)}）")

    lines += ["", "## 2. 说话人复核覆盖", ""]
    rows = job.get("review_rows", [])
    if rows:
        lines += _table(rows, ["speaker_token", "directory_group", "n_files",
                               "n_origins", "mic_tokens", "review_status"])
    else:
        lines.append("_没有可复核的说话人（检查扫描与解析状态）。_")

    lines += ["", "## 3. 阻断项", ""]
    blockers = job.get("blockers", [])
    if blockers:
        for b in blockers:
            lines.append(f"- `{b['issue_type']}`：{b['detail']}")
    else:
        lines.append("_无阻断项。_")

    lines += ["", "## 4. 批准清单", ""]
    if job["status"] == "approved":
        lines.append(f"- 行数：{counts.get('n_manifest', 0)}"
                     f"｜subject 数：{counts.get('n_subjects', 0)}"
                     f"｜origin 数：{counts.get('n_origins', 0)}")
        by_label = job.get("label_counts", {})
        lines.append(f"- 标签分布：{dict(sorted(by_label.items()))}")
        lines.append(f"- 指纹：`{job.get('manifest_fingerprint', '')}`")
        lines.append("- 文件：`manifest.csv`、`task.yaml`、`provenance.json`")
    else:
        lines.append("_未生成 `manifest.csv`：准入未通过，不能把本轮结果用于训练。_")

    lines += ["", "## 5. 人员隔离规则", ""]
    lines.append("- `group_id = <dataset>:<confirmed_subject_id>`；同一人的所有录音必须同组。")
    lines.append("- 同一 subject 出现多个 label、同一 origin 归属多个 subject、"
                 "或同一 subject 绑定多个 speaker token 都会阻断。")
    lines.append("- 同一 utterances 的多麦克风文件只保留 `selection.mic` 指定的一条。")

    lines += ["", "## 6. 下一步", ""]
    if job["status"] == "approved":
        lines.append("- 进入 split：按 `group_id` 做人员隔离划分，再关联特征缓存。")
    else:
        lines.append("- 按 `label_review.csv` 的列填写证据并改为 `review_status=confirmed`；"
                     "重跑 `sddbench prepare`。")
        lines.append("- 未确认前可继续做技术统计/特征，但不得导出为训练清单。")
    lines.append("")
    return "\n".join(lines)
