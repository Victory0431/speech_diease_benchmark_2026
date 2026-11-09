"""Metrics for binary disease/control benchmarks (NumPy only)."""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def _safe_div(a: float, b: float):
    return None if b == 0 else float(a) / float(b)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def auroc(y_true: np.ndarray, scores: np.ndarray):
    """Rank-based AUROC; None when only one class is present."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0  # average rank, 1-based
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / n_pos)


def binary_metrics(y_true, probs, positive_index: int = 1) -> dict:
    """Return point metrics for binary labels with probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    p_pos = probs[:, positive_index] if probs.ndim == 2 else probs
    y_pred = (p_pos >= 0.5).astype(int)
    cm = confusion(y_true, y_pred)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    n = tn + fp + fn + tp
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity) \
        if precision is not None and sensitivity is not None else None
    neg_precision = _safe_div(tn, tn + fn)
    neg_sensitivity = specificity
    neg_f1 = _safe_div(2 * neg_precision * neg_sensitivity,
                       neg_precision + neg_sensitivity) \
        if neg_precision is not None and neg_sensitivity is not None else None
    accuracy = _safe_div(tp + tn, n)
    balanced = (sensitivity + specificity) / 2.0 \
        if sensitivity is not None and specificity is not None else None
    p = np.clip(p_pos, EPS, 1 - EPS)
    y = y_true.astype(float)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if n else None
    brier = float(np.mean((p_pos - y_true) ** 2)) if n else None
    return {
        "n": n,
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


def per_class_rows(y_true, probs, class_names: list[str], positive_index: int = 1) -> list[dict]:
    y_true = np.asarray(y_true, dtype=int)
    m = binary_metrics(y_true, probs, positive_index)
    p_pos = probs[:, positive_index] if probs.ndim == 2 else probs
    y_pred = (p_pos >= 0.5).astype(int)
    cm = m["confusion"]
    neg_support, pos_support = cm["tn"] + cm["fp"], cm["fn"] + cm["tp"]
    neg_pred = cm["tn"] + cm["fn"]
    pos_pred = cm["fp"] + cm["tp"]
    return [
        {"class": class_names[0], "support": neg_support, "predicted": neg_pred,
         "precision": m["negative_precision"], "recall": m["specificity"],
         "f1": m["negative_f1"]},
        {"class": class_names[1], "support": pos_support, "predicted": pos_pred,
         "precision": m["precision"], "recall": m["sensitivity"], "f1": m["f1"]},
    ]
