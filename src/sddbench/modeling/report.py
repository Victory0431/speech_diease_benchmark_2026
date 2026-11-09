"""Result-pack report and figures for a completed MLP run.

Everything here is derived from the saved predictions / history of one run.
The report is meant to answer, without opening the code: what data, which
split, which configuration, which epoch, and what did the model score.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import binary_metrics, confusion

COLORS = {"control": "#2f6f9f", "disorder": "#c0504d"}


def _fmt(value, nd: int = 3) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _subject_probs(frame: pd.DataFrame, probs: np.ndarray,
                   class_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tmp = frame[["group_id", "label"]].copy()
    for i, name in enumerate(class_order):
        tmp[f"prob_{name}"] = probs[:, i]
    agg = tmp.groupby(["group_id", "label"], as_index=False).agg(
        **{f"prob_{name}": (f"prob_{name}", "mean") for name in class_order})
    y = (agg["label"].astype(str) == class_order[1]).astype(int).to_numpy()
    return y, agg[[f"prob_{name}" for name in class_order]].to_numpy()


def _roc_points(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y = y[order]
    score = score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def write_figures(out_dir: Path, job: dict, probs: np.ndarray,
                  splits: np.ndarray, merged: pd.DataFrame,
                  class_order: list[str]) -> list[str]:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    positive = class_order.index(job["positive_class"])
    written: list[str] = []

    test = splits == "test"
    if test.any():
        unit = job.get("primary_unit") or job.get("evaluation_unit") or "subject"
        y_true = (merged.loc[test, "label"].astype(str) == class_order[1]).astype(int).to_numpy()
        p_pos = probs[test, positive]

        # Confusion matrices: window-level and primary-unit level (subject or recording).
        y_sub, p_sub = _subject_probs(
            merged.loc[test, ["group_id", "label"]], probs[test], class_order)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        for ax, (name, y, p) in zip(axes, [
                ("window", y_true, p_pos), (unit, y_sub, p_sub[:, positive])]):
            cm = confusion(y, (p >= 0.5).astype(int))
            matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            ax.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                            color="white" if matrix[i, j] > matrix.max() / 2 else "#333")
            ax.set_xticks([0, 1], class_order)
            ax.set_yticks([0, 1], class_order)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"test {name} (n={len(y)})")
        fig.tight_layout()
        fig.savefig(fig_dir / "confusion_test.png", dpi=140)
        plt.close(fig)
        written.append("confusion_test.png")

        # ROC curves.
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        for name, y, score in (("window", y_true, p_pos),
                               (unit, y_sub, p_sub[:, positive])):
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr = _roc_points(y, score)
            auc = job["metrics"]["test"][name]["auroc"]
            ax.plot(fpr, tpr, label=f"{name} (AUC={_fmt(auc)})")
        ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("test ROC")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "roc_test.png", dpi=140)
        plt.close(fig)
        written.append("roc_test.png")

        # Probability distribution by true class.
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        for value, name in ((0, class_order[0]), (1, class_order[1])):
            part = p_pos[y_true == value]
            if len(part):
                ax.hist(part, bins=30, alpha=0.65, label=f"{name} (n={len(part)})",
                        color=COLORS.get(name, None))
        ax.set_xlabel(f"P({job['positive_class']})")
        ax.set_ylabel("windows")
        ax.set_title("test score distribution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "scores_test.png", dpi=140)
        plt.close(fig)
        written.append("scores_test.png")

    history_file = out_dir / "history.csv"
    if history_file.is_file():
        history = pd.read_csv(history_file)
        unit = job.get("primary_unit") or job.get("evaluation_unit") or "subject"
        score_col = f"val_{unit}_balanced_accuracy"
        if score_col not in history.columns:
            score_col = "val_subject_balanced_accuracy"
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
        axes[0].plot(history["epoch"], history["train_loss"], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend(fontsize=8)
        axes[0].set_title("loss")
        axes[1].plot(history["epoch"], history["train_accuracy"], label="train acc")
        axes[1].plot(history["epoch"], history[score_col],
                     label=f"val {unit} bAcc")
        axes[1].axvline(job.get("best_epoch", 0), color="#999", linestyle="--", linewidth=1)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylim(0, 1.02)
        axes[1].legend(fontsize=8)
        axes[1].set_title(f"accuracy (best epoch={job.get('best_epoch')})")
        fig.tight_layout()
        fig.savefig(fig_dir / "learning_curve.png", dpi=140)
        plt.close(fig)
        written.append("learning_curve.png")
    return written


def _metrics_table(job: dict) -> list[str]:
    metrics = job["metrics"]
    unit = job.get("primary_unit") or job.get("evaluation_unit") or "subject"
    lines = [
        "| split | 单位 | n | accuracy | balanced acc | sensitivity | specificity | F1 | AUROC | AP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        for level in ("window", "recording", "subject"):
            m = metrics[split][level]
            if m is None:
                continue
            mark = " **←主指标**" if (split == "test" and level == unit) else ""
            lines.append(
                f"| {split} | {level} | {m['n']} | {_fmt(m['accuracy'])} | "
                f"{_fmt(m['balanced_accuracy'])}{mark} | {_fmt(m['sensitivity'])} | "
                f"{_fmt(m['specificity'])} | {_fmt(m['f1'])} | {_fmt(m['auroc'])} | "
                f"{_fmt(m['average_precision'])} |")
    return lines


def _confusion_lines(job: dict) -> list[str]:
    unit = job.get("primary_unit") or job.get("evaluation_unit") or "subject"
    m = job["metrics"]["test"][unit]
    cm = m["confusion"]
    order = job["class_order"]
    return [
        "| true \\ predicted | " + " | ".join(order) + " |",
        "| --- | ---: | ---: |",
        f"| {order[0]} | {cm['tn']} | {cm['fp']} |",
        f"| {order[1]} | {cm['fn']} | {cm['tp']} |",
    ]


def render_run_report(job: dict) -> str:
    data = job["data"]
    unit = job.get("primary_unit") or job.get("evaluation_unit") or "subject"
    grouping = job.get("grouping") or "subject"
    group_label = "确认后的 subject" if grouping == "subject" else "单条录音（本库无说话人信息）"
    subject_counts = data["subject_counts"]
    label_counts = data["label_counts"]
    outputs = job.get("outputs", {})
    run_dir = Path(outputs.get("report", "report.md")).parent
    history = pd.read_csv(run_dir / "history.csv") if (run_dir / "history.csv").is_file() else None
    best_row = None
    if history is not None and len(history):
        best_row = history.loc[history["epoch"] == job.get("best_epoch")].head(1)

    lines = [
        f"# 实验运行：{job['run_id']}",
        "",
        "## 1. 这次运行做了什么",
        "",
        f"- 任务：`{job['task_id']}`（数据集 `{job['dataset']}`）",
        f"- 数据准备：`{job['prepared_id']}`；固定划分：`{job['split_id']}`",
        f"- 特征：D3 缓存的 MFCC 统计量，`{data['feature_dim']} 维/窗口`，"
        f"采样率 `{data['sample_rate']} Hz`，缓存 key `{data['feature_cache'].get('feature_key')}`",
        f"- 模型：`{job['model']['name']}`，隐藏层 `{job['model']['hidden_dims']}`，"
        f"dropout `{job['model']['dropout']}`，输入 `{data['feature_dim']}` → 输出 `{len(job['class_order'])}`",
        f"- 训练：epochs `{job['training']['epochs']}`，batch `{job['training']['batch_size']}`，"
        f"lr `{job['training']['learning_rate']}`，优化器 `{job['training']['optimizer']}`，"
        f"seed `{job['seed']}`，权重 `{job['training'].get('class_weighting')}`",
        f"- 早停/选模型：验证集 **{unit} balanced accuracy**，"
        f"最佳 epoch `{job['best_epoch']}`"
        f"（`{_fmt(job.get('best_val_balanced_accuracy', job.get('best_val_subject_balanced_accuracy')))}`）",
        f"- 运行状态：`{job['status']}`；开始 `{job['started_at']}`，结束 `{job['finished_at']}`",
        f"- 代码：`{job['code_version']}`，git `{(job.get('git') or {}).get('commit')}`"
        f"（dirty={(job.get('git') or {}).get('dirty')}）",
        "",
        "## 2. 数据与划分",
        "",
        f"- 总录音：`{data['n_samples']}`；分组（group）：`{data['n_subjects']}`",
        "| split | 录音数 | 分组数 | 标签录音分布 |",
        "| --- | ---: | ---: | --- |",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {sum(label_counts[split].values())} | "
                     f"{subject_counts[split]} | {label_counts[split]} |")
    lines += [
        "",
        "- 划分按 `group_id` 进行：本库的分组单位是" + group_label +
        "，train/val/test 的分组集合互斥；训练时标准化统计量只用 train 拟合。",
        "",
        "## 3. 指标",
        "",
        *(_metrics_table(job)),
        "",
        f"正类：`{job['positive_class']}`；概率阈值 0.5；主评价单位：{unit}。",
        "",
        f"### Test 混淆矩阵（{unit}）",
        "",
        *(_confusion_lines(job)),
        "",
        "## 4. 怎么读这个结果",
        "",
        "- **主评价单位**由任务的分组策略决定：有说话人时按人等权平均，"
        "没有说话人时按录音；**window 指标**只用于诊断，不能拿来当最终成绩。",
        "- 本基线每条录音只取 1 个观察窗（head 4 s），因此 window 与 recording 指标相同；"
        "两者都保留，是为了以后多窗口协议不改报告结构。",
        "- 测试集从未参与早停或阈值选择；`test` 只在最后评估一次。",
        "- 分数高低不是工程验收标准；先看流程是否可追溯、预测是否覆盖完整、指标是否可由预测重算。",
        "",
        "## 5. 限制",
        "",
        f"- val 只有 `{subject_counts['val']}` 个分组，单次结果的方差较大；"
        "正式比较应多 seed 并报告不确定性。",
        "- MFCC 统计量丢弃了时间顺序信息，MLP 只能利用全局均值和标准差；"
        "这是第一条可复现基线，不代表方法上限。",
        "- 4 s 观察窗对约 42.6% 的录音需要右补零；补零帧不进入 MFCC 统计，"
        "但短录音的信息量确实更少。",
        "",
        "## 6. 产物索引",
        "",
        "| 文件 | 内容 |",
        "| --- | --- |",
        "| `config.yaml` | 完整生效配置（含 CLI 覆盖） |",
        "| `run.json` | 运行身份、环境、数据指纹、指标与状态 |",
        "| `history.csv` | 每个 epoch 的 loss / accuracy / 验证主指标 |",
        "| `best.pt` | 被选中的模型权重、类序、标准化参数 |",
        "| `predictions_windows.csv` | 窗口级概率与预测 |",
        "| `predictions_subjects.csv` | 人员级概率与预测（主指标来源） |",
        "| `metrics.json` / `metrics_per_class.csv` | 统一指标与每类明细 |",
        "| `figures/` | 混淆矩阵、ROC、分数分布、学习曲线 |",
        "| `inputs/` | 本次使用的 manifest / split / task 快照 |",
        "",
    ]
    if best_row is not None and len(best_row):
        row = best_row.iloc[0]
        lines += [
            "## 7. 最佳 epoch 记录",
            "",
            f"- epoch `{int(row['epoch'])}`：train_loss `{_fmt(float(row['train_loss']))}`，"
            f"val_loss `{_fmt(float(row['val_loss']))}`，"
            f"val {unit} balanced accuracy "
            f"`{_fmt(float(row.get(f'val_{unit}_balanced_accuracy', row.get('val_subject_balanced_accuracy'))))}`",
            "",
        ]
    return "\n".join(lines)
