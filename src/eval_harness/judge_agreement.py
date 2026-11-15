"""Judge agreement: do two judges agree on the same responses?

When you swap judges (heuristic -> LLM judge) or compare two LLM judges,
the first question is whether they even agree. This module answers it with
scikit-learn's standard agreement metrics, on the per-item scores the
runner writes to its report JSON.

- Continuous agreement: mean absolute score difference + Pearson r.
- Pass/fail agreement: scores are binarized at a threshold (default 0.5),
  then accuracy, Cohen's kappa (chance-corrected), confusion matrix, and a
  per-class precision/recall/F1 report.

Kappa is the headline number: 1.0 is perfect agreement, 0 is chance-level,
negative is worse than chance. A new judge that scores higher on average
but disagrees with the old one on *which* items pass is not a drop-in
replacement — kappa catches that where a mean comparison does not.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

DEFAULT_THRESHOLD = 0.5
_LABELS = ["fail", "pass"]

_MISSING = object()


def binarize(scores: list[float], *, threshold: float = DEFAULT_THRESHOLD) -> list[int]:
    """Binarize continuous scores at `threshold` (1 = pass, 0 = fail)."""
    return [1 if s >= threshold else 0 for s in scores]


def align_reports(report_a: dict, report_b: dict) -> tuple[list[float], list[float], int]:
    """Pair per-item scores from two runner reports by item_id.

    Drops pairs where either judge skipped (score None) or the item appears
    in only one report. Returns (scores_a, scores_b, n_dropped).
    """
    ids_b = {e.get("item_id") for e in report_b.get("items", [])}
    scores_b = {e.get("item_id"): e.get("score") for e in report_b.get("items", [])}
    ids_a = set()
    a_scores: list[float] = []
    b_scores: list[float] = []
    dropped = 0
    for entry in report_a.get("items", []):
        item_id = entry.get("item_id")
        ids_a.add(item_id)
        a = entry.get("score")
        b = scores_b.get(item_id, _MISSING)
        if a is None or b is _MISSING or b is None:
            dropped += 1
            continue
        a_scores.append(float(a))
        b_scores.append(float(b))
    dropped += len(ids_b - ids_a)
    return a_scores, b_scores, dropped


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r, or None when either side is constant (r undefined)."""
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def agreement_report(
    scores_a: list[float],
    scores_b: list[float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    judge_a: str = "a",
    judge_b: str = "b",
) -> dict:
    """Agreement metrics between two judges' scores on the same items.

    `scores_a` is the reference (rows of the confusion matrix), `scores_b`
    the comparison (columns). Raises ValueError on empty or mismatched input.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"score lists differ in length: {len(scores_a)} vs {len(scores_b)}"
        )
    if not scores_a:
        raise ValueError("no paired scores to compare")
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    a_bin = binarize(a, threshold=threshold)
    b_bin = binarize(b, threshold=threshold)

    # Single-class input makes kappa undefined (nan) and sklearn warns about
    # it; the nan -> None mapping below is deliberate, so silence the noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = float(cohen_kappa_score(a_bin, b_bin))
    if math.isnan(kappa):
        # Single-class input (e.g. both judges pass everything): kappa is
        # undefined, not zero. Accuracy + mean_abs_diff still describe it.
        kappa = None
    cm = confusion_matrix(a_bin, b_bin, labels=[0, 1])
    per_class = classification_report(
        a_bin, b_bin, labels=[0, 1], target_names=_LABELS,
        output_dict=True, zero_division=0,
    )
    pearson = _pearson(a, b)

    return {
        "judge_a": judge_a,
        "judge_b": judge_b,
        "n": len(a),
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(a_bin, b_bin)), 4),
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 4),
        "pearson_r": round(pearson, 4) if pearson is not None else None,
        "confusion_matrix": {
            "labels": _LABELS,
            "matrix": [[int(x) for x in row] for row in cm.tolist()],
        },
        "per_class": {
            label: {
                "precision": round(float(per_class[label]["precision"]), 4),
                "recall": round(float(per_class[label]["recall"]), 4),
                "f1": round(float(per_class[label]["f1-score"]), 4),
                "support": int(per_class[label]["support"]),
            }
            for label in _LABELS
        },
    }


def compare_reports(
    report_a: dict, report_b: dict, *, threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """Agreement between the judges of two runner report dicts.

    Judge names come from each report's "judge" field. Pairs dropped by
    alignment (skipped or unmatched items) are counted in "n_dropped".
    """
    a, b, dropped = align_reports(report_a, report_b)
    out = agreement_report(
        a, b, threshold=threshold,
        judge_a=report_a.get("judge", "a"),
        judge_b=report_b.get("judge", "b"),
    )
    out["n_dropped"] = dropped
    return out
