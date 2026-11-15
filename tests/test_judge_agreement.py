"""Tests for judge agreement metrics (scikit-learn)."""
import pytest

from eval_harness.golden_sets import load_golden_set
from eval_harness.judges import HeuristicJudge
from eval_harness.responders import healthy_responder
from eval_harness.runner import run_eval
from eval_harness.judge_agreement import (
    agreement_report,
    align_reports,
    binarize,
    compare_reports,
)

ITEMS = load_golden_set()


class LengthJudge:
    """Stub judge: 1.0 for long responses, 0.0 for short ones."""

    def score(self, item, response):
        return {"score": 1.0 if len(response or "") > 50 else 0.0,
                "judge": "length"}


def _report(judge_name, scores_by_id):
    return {"judge": judge_name,
            "items": [{"item_id": i, "score": s}
                      for i, s in scores_by_id.items()]}


def test_binarize_threshold():
    assert binarize([0.5, 0.49, 0.9, 0.1]) == [1, 0, 1, 0]
    assert binarize([0.5, 0.49, 0.9, 0.1], threshold=0.7) == [0, 0, 1, 0]


def test_identical_judges_agree_perfectly():
    a = [0.9, 0.2, 0.8, 0.1, 0.7]
    r = agreement_report(a, list(a))
    assert r["n"] == 5
    assert r["accuracy"] == 1.0
    assert r["cohen_kappa"] == 1.0
    assert r["mean_abs_diff"] == 0.0
    assert r["pearson_r"] == 1.0
    assert r["confusion_matrix"]["labels"] == ["fail", "pass"]
    assert r["confusion_matrix"]["matrix"] == [[2, 0], [0, 3]]
    assert r["per_class"]["pass"]["precision"] == 1.0
    assert r["per_class"]["pass"]["recall"] == 1.0


def test_opposite_judges_disagree_fully():
    r = agreement_report([0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9])
    assert r["accuracy"] == 0.0
    assert r["cohen_kappa"] == -1.0
    assert r["mean_abs_diff"] == 0.8
    assert r["pearson_r"] == -1.0


def test_partial_agreement_kappa():
    # 3 of 4 pass/fail calls agree; chance agreement is 0.5, so kappa 0.5.
    r = agreement_report([0.9, 0.8, 0.2, 0.1], [0.9, 0.2, 0.2, 0.1])
    assert r["accuracy"] == 0.75
    assert r["cohen_kappa"] == pytest.approx(0.5)
    assert r["mean_abs_diff"] == pytest.approx(0.15)


def test_single_class_kappa_undefined_not_zero():
    r = agreement_report([0.9, 0.9], [0.8, 0.8])
    assert r["accuracy"] == 1.0
    assert r["cohen_kappa"] is None
    assert r["pearson_r"] is None
    assert r["mean_abs_diff"] == pytest.approx(0.1)


def test_threshold_changes_labels():
    scores = [0.6, 0.4]
    other = [0.9, 0.1]
    assert agreement_report(scores, other, threshold=0.5)["accuracy"] == 1.0
    assert agreement_report(scores, other, threshold=0.7)["accuracy"] == 0.5


def test_empty_and_mismatched_raise():
    with pytest.raises(ValueError, match="no paired scores"):
        agreement_report([], [])
    with pytest.raises(ValueError, match="differ in length"):
        agreement_report([0.9], [0.9, 0.1])


def test_align_drops_skipped_scores():
    a = _report("heuristic", {"BIL-001": 0.9, "BIL-002": 0.2, "BIL-003": 0.8})
    b = _report("llm", {"BIL-001": 0.8, "BIL-002": None, "BIL-003": 0.7})
    scores_a, scores_b, dropped = align_reports(a, b)
    assert scores_a == [0.9, 0.8]
    assert scores_b == [0.8, 0.7]
    assert dropped == 1


def test_align_drops_unmatched_ids():
    a = _report("heuristic", {"BIL-001": 0.9, "BIL-002": 0.2})
    b = _report("llm", {"BIL-001": 0.8, "SEC-009": 0.5})
    _, _, dropped = align_reports(a, b)
    assert dropped == 2  # BIL-002 missing from B, SEC-009 missing from A


def test_compare_reports_end_to_end():
    scored_a = run_eval(ITEMS, healthy_responder, HeuristicJudge())
    scored_b = run_eval(ITEMS, healthy_responder, LengthJudge())
    report_a = {"judge": "heuristic",
                "items": [{"item_id": e["id"], "score": e["score"]}
                          for e in scored_a]}
    report_b = {"judge": "length",
                "items": [{"item_id": e["id"], "score": e["score"]}
                          for e in scored_b]}
    r = compare_reports(report_a, report_b)
    assert r["n"] == 50
    assert r["n_dropped"] == 0
    assert r["judge_a"] == "heuristic"
    assert r["judge_b"] == "length"
    assert 0.0 <= r["accuracy"] <= 1.0
    assert r["cohen_kappa"] is not None
    assert set(r["per_class"]) == {"fail", "pass"}


def test_compare_reports_counts_skipped():
    scored = run_eval(ITEMS, healthy_responder, HeuristicJudge())
    report_a = {"judge": "heuristic",
                "items": [{"item_id": e["id"], "score": e["score"]}
                          for e in scored]}
    items_b = [{"item_id": e["id"], "score": e["score"]} for e in scored]
    items_b[0]["score"] = None  # LLM judge skipped one item
    r = compare_reports(report_a, {"judge": "llm", "items": items_b})
    assert r["n"] == 49
    assert r["n_dropped"] == 1
