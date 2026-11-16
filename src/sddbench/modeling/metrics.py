"""Metrics for binary disease/control benchmarks (NumPy only).

Everything here follows one convention: probabilities are turned into labels
with ``p_positive >= threshold`` (see :func:`decide_binary`), and ROC/AUROC/AP
are computed on **unique score thresholds** so tied scores are handled once,
not once per row.  Row order must never change a metric.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _safe_div(a: float, b: float):
    return None if b == 0 else float(a) / float(b)


def _f1(precision, recall):
    """F1 with the sklearn ``zero_division=0`` policy.

    A classifier that never predicts a class has a *defined* F1 of 0 for it;
    returning ``None`` in that case would let reports silently drop a class.
    """
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2.0 * precision * recall / denom


def decide_binary(positive_scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """The one and only binary decision rule used by metrics, plots and CSV."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
    return (np.asarray(positive_scores, dtype=float) >= float(threshold)).astype(int)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _curve(y_true: np.ndarray, scores: np.ndarray):
    """ROC/precision-recall points at each distinct score (ties grouped).

    Returns ``None`` when a class is absent, because AUROC/AP are undefined.
    The returned arrays include the ``(0, 0)`` ROC start point; the final
    point is ``(1, 1)`` by construction.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0 or len(scores) == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]
    last_of_group = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp = np.cumsum(y)[last_of_group].astype(float)
    fp = np.cumsum(1 - y)[last_of_group].astype(float)
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    return fpr, tpr, precision, recall


def roc_points(y_true: np.ndarray, scores: np.ndarray):
    """ROC points used by both the numeric AUROC and the plotted curve."""
    curve = _curve(y_true, scores)
    if curve is None:
        return None
    return curve[0], curve[1]


def auroc(y_true: np.ndarray, scores: np.ndarray):
    """Trapezoidal area under the ROC curve built from distinct thresholds.

    This is the same curve drawn in ``report.py``, so the printed AUROC and the
    plotted ROC can no longer disagree on tied scores.
    """
    points = roc_points(y_true, scores)
    if points is None:
        return None
    fpr, tpr = points
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def average_precision(y_true: np.ndarray, scores: np.ndarray):
    """Threshold-based AP: ``sum (recall_i - recall_{i-1}) * precision_i``.

    Tied scores form one operating point, which makes the value invariant to
    row order and consistent with the ROC thresholds.
    """
    curve = _curve(y_true, scores)
    if curve is None:
        return None
    _, _, precision, recall = curve
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def binary_metrics(y_true, probs, positive_index: int = 1,
                   threshold: float = 0.5) -> dict:
    """Return point metrics for binary labels with probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    p_pos = probs[:, positive_index] if probs.ndim == 2 else probs
    y_pred = decide_binary(p_pos, threshold)
    cm = confusion(y_true, y_pred)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    n = tn + fp + fn + tp
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = _f1(precision, sensitivity)
    neg_precision = _safe_div(tn, tn + fn)
    neg_sensitivity = specificity
    neg_f1 = _f1(neg_precision, neg_sensitivity)
    accuracy = _safe_div(tp + tn, n)
    balanced = (sensitivity + specificity) / 2.0 \
        if sensitivity is not None and specificity is not None else None
    p = np.clip(p_pos, EPS, 1 - EPS)
    y = y_true.astype(float)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if n else None
    brier = float(np.mean((p_pos - y_true) ** 2)) if n else None
    return {
        "n": n,
        "threshold": float(threshold),
        "positive_rate": _safe_div(tp + fn, n),
        "confusion": cm,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "negative_precision": neg_precision,
        "negative_f1": neg_f1,
        "macro_f1": None if f1 is None or neg_f1 is None else (f1 + neg_f1) / 2.0,
        "auroc": auroc(y_true, p_pos),
        "average_precision": average_precision(y_true, p_pos),
        "log_loss": log_loss,
        "brier": brier,
    }


def per_class_rows(y_true, probs, class_names: list[str],
                   positive_index: int = 1, threshold: float = 0.5) -> list[dict]:
    """One row per class, honouring ``positive_index``.

    ``class_names[0]`` is not assumed to be the negative class: only
    ``positive_index`` decides which class the sensitivity/F1 refer to.
    """
    y_true = np.asarray(y_true, dtype=int)
    m = binary_metrics(y_true, probs, positive_index, threshold)
    cm = m["confusion"]
    # y_true inside this module is always 0/1 with 1 = positive, so the
    # positive class support/predicted counts come from (tp+fn)/(tp+fp).
    pos_row = {"support": cm["fn"] + cm["tp"], "predicted": cm["fp"] + cm["tp"],
               "precision": m["precision"], "recall": m["sensitivity"], "f1": m["f1"]}
    neg_row = {"support": cm["tn"] + cm["fp"], "predicted": cm["tn"] + cm["fn"],
               "precision": m["negative_precision"], "recall": m["specificity"],
               "f1": m["negative_f1"]}
    rows = []
    for i, name in enumerate(class_names):
        stats = pos_row if i == positive_index else neg_row
        rows.append({"class": name, **stats})
    return rows
