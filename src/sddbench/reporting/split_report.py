"""Render `split.md` for a person-disjoint split job."""
from __future__ import annotations


def render_split(job: dict) -> str:
    lines = [f"# 数据划分：{job['split_id']}", ""]
    lines.append(f"- 数据集：`{job['dataset_id']}`｜任务：`{job['task_id']}`")
    lines.append(f"- 源清单：`{job['prepared_id']}`（manifest 指纹 "
                 f"`{job.get('manifest_fingerprint', '')[:16]}`）")
    lines.append(f"- 随机种子：`{job['seed']}`｜比例："
                 f"train {job['fractions']['train']} / val {job['fractions']['val']} / "
                 f"test {job['fractions']['test']}")
    lines.append(f"- 单位：`group_id`（确认后的 subject），任何一个人只出现在一个 split")
    lines.append("")

    lines += ["## 1. 总量", ""]
    summary = job["summary"]
    lines.append(f"- 录音：{summary['n_samples']}｜subject：{summary['n_subjects']}")
    lines.append("")
    lines.append("| split | 录音数 | subject 数 | 标签录音分布 | 标签 subject 分布 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for split in ("train", "val", "test"):
        b = summary["by_split"][split]
        s = summary["subjects_by_split"][split]
        lines.append(f"| {split} | {b['n_samples']} | {s['n_subjects']} | "
                     f"{b['labels']} | {s['labels']} |")

    lines += ["", "## 2. 按类别的 subject 分配", ""]
    lines.append("| 标签 | train | val | test |")
    lines.append("| --- | --- | --- | --- |")
    for label, alloc in sorted(summary["allocation_by_label"].items()):
        lines.append(f"| {label} | {alloc['train']} | {alloc['val']} | {alloc['test']} |")

    lines += ["", "## 3. 隔离检查", ""]
    lines.append("- 已断言：同一 `group_id` 的所有录音只属于一个 split。")
    lines.append("- 已断言：split 行数 = manifest 行数，无漏样本、无重复。")
    lines.append("- 该划分是确定性的：相同 seed + manifest 会得到相同结果。")

    lines += ["", "## 4. 边界", ""]
    lines.append("- 这是**按人划分**，不是按录音随机划分；因此同一人的不同句子不会跨集。")
    lines.append("- 训练时标准化统计量只能来自 train，不得使用 val/test（由 `run` 强制）。")
    lines.append("- 该 split 不改变原始数据，只写索引文件。")
    lines.append("")
    return "\n".join(lines)
