"""F03/F06/F10/F12: metric ties, threshold, config validation, length ratio, weights."""
from __future__ import annotations

import numpy as np
import pytest

from sddbench.data.stats import _length_candidates
from sddbench.modeling.experiment import (
    DEFAULTS,
    ExperimentConfigError,
    _merge,
    set_override,
    validate_experiment,
)
from sddbench.modeling.metrics import (
    average_precision,
    auroc,
    binary_metrics,
    decide_binary,
    per_class_rows,
    roc_points,
)
from sddbench.modeling.trainer import _sample_weights


# ------------------------------------------------------------------ F03

def test_ap_and_auroc_are_invariant_to_row_order_with_ties():
    y = np.array([1, 0])
    for scores in (np.array([0.5, 0.5]), np.array([0.5, 0.5])[::-1]):
        assert average_precision(y, scores) == pytest.approx(0.5)
        assert auroc(y, scores) == pytest.approx(0.5)
    assert average_precision(y[::-1], np.array([0.5, 0.5])[::-1]) == pytest.approx(0.5)


def test_roc_numeric_area_matches_plotted_points():
    y = np.array([0, 1, 0, 1, 1])
    scores = np.array([0.2, 0.8, 0.8, 0.5, 0.1])
    fpr, tpr = roc_points(y, scores)
    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0)
    assert area == pytest.approx(auroc(y, scores))


def test_extreme_predictions_and_missing_class():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.zeros(4)) == pytest.approx(0.5)
    assert average_precision(y, np.zeros(4)) == pytest.approx(0.5)
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, perfect) == pytest.approx(1.0)
    assert average_precision(y, perfect) == pytest.approx(1.0)
    assert auroc(np.zeros(4, dtype=int), perfect) is None
    assert average_precision(np.zeros(4, dtype=int), perfect) is None


def test_f1_zero_denominator_is_zero_not_null():
    y = np.array([1, 1, 0])
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    m = binary_metrics(y, probs, 1)
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0           # never predicted positive
    assert m["negative_f1"] == 0.0  # negative class never detected either
    assert m["macro_f1"] is not None


def test_per_class_rows_follow_positive_index():
    y = np.array([1, 0, 1, 0])           # 1 = positive class
    probs = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    rows1 = per_class_rows(y, probs, ["control", "disorder"], positive_index=1)
    assert rows1[0]["class"] == "control" and rows1[0]["support"] == 2
    assert rows1[1]["class"] == "disorder" and rows1[1]["support"] == 2
    assert rows1[1]["precision"] == pytest.approx(0.5)

    # Same numbers, but now class 0 is the positive one.
    rows0 = per_class_rows(y, probs, ["control", "disorder"], positive_index=0)
    assert rows0[0]["support"] == 2 and rows0[1]["support"] == 2
    assert rows0[0]["precision"] == pytest.approx(0.5)


# ------------------------------------------------------------------ F06

def _base_cfg() -> dict:
    return _merge(DEFAULTS, {
        "dataset": "demo", "prepared_id": "p", "split_id": "s",
    })


def test_threshold_changes_labels_metrics_and_records():
    y = np.array([0, 1, 1, 0])
    probs = np.array([[0.7, 0.3], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4]])
    low = binary_metrics(y, probs, 1, threshold=0.2)
    high = binary_metrics(y, probs, 1, threshold=0.8)
    assert low["confusion"] != high["confusion"]
    assert low["threshold"] == 0.2 and high["threshold"] == 0.8
    # Exactly-at-threshold is positive everywhere, same as decide_binary().
    assert decide_binary(np.array([0.5]), 0.5).tolist() == [1]


def test_unknown_experiment_keys_are_rejected():
    cfg = _base_cfg()
    cfg["mystery_option"] = True
    with pytest.raises(ExperimentConfigError, match="unknown experiment key"):
        validate_experiment(cfg)
    cfg = _base_cfg()
    cfg["training"]["mystery"] = 1
    with pytest.raises(ExperimentConfigError, match="unknown key"):
        validate_experiment(cfg)


def test_invalid_values_are_rejected_early():
    cfg = _base_cfg()
    cfg["training"]["epochs"] = 0
    with pytest.raises(ExperimentConfigError, match="epochs"):
        validate_experiment(cfg)

    cfg = _base_cfg()
    cfg["evaluation"]["binary_threshold"] = 1.5
    with pytest.raises(ExperimentConfigError, match="binary_threshold"):
        validate_experiment(cfg)

    cfg = _base_cfg()
    cfg["model"]["activation"] = "tanh"
    with pytest.raises(ExperimentConfigError, match="activation"):
        validate_experiment(cfg)


def test_unimplemented_options_are_rejected_not_ignored():
    cfg = _base_cfg()
    cfg["training"]["selection_direction"] = "min"
    with pytest.raises(ExperimentConfigError, match="selection_direction"):
        validate_experiment(cfg)

    cfg = _base_cfg()
    cfg["labels"]["class_order"] = ["control", "disorder", "other"]
    with pytest.raises(ExperimentConfigError, match="two distinct"):
        validate_experiment(cfg)


def test_override_is_revalidated():
    cfg = _base_cfg()
    set_override(cfg, "training.selection_direction=min")
    with pytest.raises(ExperimentConfigError):
        validate_experiment(cfg)


# ------------------------------------------------------------------ F10

def test_length_ratios_are_fractions():
    rows = _length_candidates([1.0, 2.0, 8.0, 12.0], [4.0])
    row = rows[0]
    assert row["n_longer_than_L"] == 2
    assert row["ratio_longer_than_L"] == pytest.approx(0.5)
    assert row["ratio_shorter_than_L"] == pytest.approx(0.5)
    assert 0.0 <= row["ratio_longer_than_L"] <= 1.0


# ------------------------------------------------------------------ F12

def test_subject_weights_are_fitted_on_train_only():
    groups = np.array(["A", "A", "B", "C", "D", "D", "D"])
    train = np.array([True, True, True, False, False, False, False])
    before = _sample_weights(groups, train, "subject_balanced")

    # Adding/removing validation and test groups must not move train weights.
    changed = groups.copy()
    changed[3:] = ["X", "Y", "Y", "Y"]  # val/test composition changes
    after = _sample_weights(changed, train, "subject_balanced")
    assert np.allclose(before[train], after[train])
    assert np.allclose(before[~train], 1.0)
    assert np.allclose(after[~train], 1.0)

    unique_train = set(groups[train].tolist())
    mass = {g: float(sum(before[train][groups[train] == g])) for g in unique_train}
    assert len(set(round(v, 6) for v in mass.values())) == 1  # equal total per subject
