"""Tests for PSI and mean-delta drift detection."""
import pytest

from eval_harness import drift


def test_identical_distributions_no_drift():
    scores = [0.9, 0.8, 0.7, 0.85, 0.75] * 10
    r = drift.detect_drift(scores, scores)
    assert r["drift_detected"] is False
    assert r["psi"] == pytest.approx(0.0)
    assert r["severity"] == "none"


def test_psi_zero_for_same_distribution():
    assert drift.psi([0.5] * 20, [0.5] * 20) == pytest.approx(0.0)


def test_shifted_distribution_detects_drift():
    baseline = [0.9] * 50
    current = [0.2] * 50
    r = drift.detect_drift(baseline, current)
    assert r["drift_detected"] is True
    assert r["severity"] == "high"
    assert r["psi"] > 0.25


def test_mean_delta_threshold():
    baseline = [0.8] * 50
    current = [0.65] * 50  # delta 0.15 > 0.1, but same bin -> PSI ~0
    assert drift.mean_delta(baseline, current) == pytest.approx(0.15)
    r = drift.detect_drift(baseline, current)
    assert r["drift_detected"] is True


def test_small_change_no_drift():
    baseline = [0.80, 0.82, 0.78, 0.81, 0.79] * 10
    current = [0.81, 0.83, 0.79, 0.82, 0.80] * 10
    r = drift.detect_drift(baseline, current)
    assert r["drift_detected"] is False


def test_empty_lists_raise():
    with pytest.raises(ValueError):
        drift.psi([], [0.5])
    with pytest.raises(ValueError):
        drift.mean_delta([0.5], [])


def test_deterministic():
    b = [0.9, 0.3, 0.7] * 20
    c = [0.4, 0.5, 0.6] * 20
    assert drift.detect_drift(b, c) == drift.detect_drift(b, c)
