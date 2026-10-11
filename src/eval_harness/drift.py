"""Drift detection: PSI and mean-delta, fully deterministic.

PSI (Population Stability Index) bins scores into 10 equal-width bins over
[0, 1] and compares baseline vs current distributions:

    PSI = sum((cur% - base%) * ln(cur% / base%))

Conventions: PSI < 0.1 no significant change; 0.1-0.25 moderate;
> 0.25 significant drift. Epsilon smoothing avoids division by zero.
"""
from __future__ import annotations

import math

N_BINS = 10
EPSILON = 1e-4
PSI_DRIFT_THRESHOLD = 0.25
MEAN_DELTA_THRESHOLD = 0.1


def _bin_scores(scores: list[float], n_bins: int = N_BINS) -> list[int]:
    counts = [0] * n_bins
    for s in scores:
        s = max(0.0, min(1.0, s))
        idx = min(int(s * n_bins), n_bins - 1)
        counts[idx] += 1
    return counts


def psi(baseline: list[float], current: list[float],
        n_bins: int = N_BINS) -> float:
    """Population Stability Index between two score distributions."""
    if not baseline or not current:
        raise ValueError("psi requires non-empty baseline and current lists")
    base_counts = _bin_scores(baseline, n_bins)
    cur_counts = _bin_scores(current, n_bins)
    n_base, n_cur = len(baseline), len(current)
    total = 0.0
    for b, c in zip(base_counts, cur_counts):
        p_base = b / n_base
        p_cur = c / n_cur
        p_base = max(p_base, EPSILON)
        p_cur = max(p_cur, EPSILON)
        total += (p_cur - p_base) * math.log(p_cur / p_base)
    return total


def mean_delta(baseline: list[float], current: list[float]) -> float:
    if not baseline or not current:
        raise ValueError("mean_delta requires non-empty lists")
    return abs(sum(current) / len(current) - sum(baseline) / len(baseline))


def detect_drift(baseline: list[float], current: list[float]) -> dict:
    """Return {psi, mean_delta, drift_detected, severity, details}."""
    psi_val = psi(baseline, current)
    md_val = mean_delta(baseline, current)
    drift = psi_val > PSI_DRIFT_THRESHOLD or md_val > MEAN_DELTA_THRESHOLD
    if psi_val > PSI_DRIFT_THRESHOLD or md_val > 2 * MEAN_DELTA_THRESHOLD:
        severity = "high"
    elif drift:
        severity = "moderate"
    else:
        severity = "none"
    return {
        "psi": round(psi_val, 4),
        "mean_delta": round(md_val, 4),
        "drift_detected": drift,
        "severity": severity,
        "thresholds": {"psi": PSI_DRIFT_THRESHOLD,
                       "mean_delta": MEAN_DELTA_THRESHOLD},
    }
