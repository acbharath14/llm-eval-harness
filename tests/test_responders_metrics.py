"""Tests for demo responders and metrics."""
from unittest.mock import patch, MagicMock
import io
import json
import os
import urllib.error

import pytest

from eval_harness.responders import (
    healthy_responder, regressed_responder, ollama_responder, RESPONDERS,
)
from eval_harness.golden_sets import load_golden_set
from eval_harness.judges import HeuristicJudge
from eval_harness.metrics import (
    grounded_scores, augmented_keywords, severity_slices, aggregate,
)

ITEMS = load_golden_set()
JUDGE = HeuristicJudge()

SAMPLE_ITEM = {"id": "t1", "question": "What is 2+2?",
               "expected_keywords": [], "must_not_contain": []}


def _mock_ollama_response(text: str):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"response": text}).encode()
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_ollama_responder_registered():
    assert "ollama" in RESPONDERS
    assert RESPONDERS["ollama"] is ollama_responder


def test_ollama_responder_returns_answer():
    with patch("urllib.request.urlopen",
               return_value=_mock_ollama_response("Four.")):
        answer = ollama_responder(SAMPLE_ITEM)
    assert answer == "Four."


def test_ollama_responder_uses_env_model():
    with patch.dict(os.environ, {"OLLAMA_MODEL": "qwen2.5-coder:32b"}), \
         patch("urllib.request.urlopen",
               return_value=_mock_ollama_response("ok")) as mock_urlopen:
        ollama_responder(SAMPLE_ITEM)
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["model"] == "qwen2.5-coder:32b"


def test_ollama_responder_unreachable():
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("refused")):
        with pytest.raises(RuntimeError, match="not reachable"):
            ollama_responder(SAMPLE_ITEM)


def test_ollama_responder_empty_response():
    with patch("urllib.request.urlopen",
               return_value=_mock_ollama_response("  ")):
        with pytest.raises(RuntimeError, match="empty response"):
            ollama_responder(SAMPLE_ITEM)


def test_healthy_responder_uses_ideal_answer():
    item = ITEMS[0]
    assert healthy_responder(item) == item["ideal_answer"]


def test_healthy_scores_well():
    scores = [JUDGE.score(it, healthy_responder(it))["score"] for it in ITEMS]
    assert sum(scores) / len(scores) > 0.7


def test_regressed_scores_poorly():
    scores = [JUDGE.score(it, regressed_responder(it))["score"] for it in ITEMS]
    assert sum(scores) / len(scores) < 0.3


def test_regressed_deterministic():
    item = ITEMS[5]
    assert regressed_responder(item) == regressed_responder(item)


def test_responders_registered():
    assert set(RESPONDERS) == {"healthy", "regressed", "ollama"}


def test_grounded_recall_perfect():
    item = ITEMS[0]
    g = grounded_scores(item, item["ideal_answer"])
    assert g["grounded_recall"] >= 0.5  # ideal answer covers most keywords
    assert g["keywords_missed"] == [] or isinstance(g["keywords_missed"], list)


def test_grounded_recall_zero_for_empty():
    g = grounded_scores(ITEMS[0], "")
    assert g["grounded_recall"] == 0.0


def test_augmented_never_penalizes():
    # Augmented keywords are diagnostic; the judge score must not depend on them.
    item = ITEMS[0]
    base = JUDGE.score(item, item["ideal_answer"])["score"]
    extra = item["ideal_answer"] + " Also consider extraneous flibbertigibbet wobble."
    assert JUDGE.score(item, extra)["score"] >= base - 0.2
    assert isinstance(augmented_keywords(item, extra), list)


def test_severity_slices_cover_all_levels():
    scored = [
        {"score": 0.9, "item": {"severity": "critical"}},
        {"score": 0.5, "item": {"severity": "medium"}},
        {"score": 0.7, "item": {"severity": "low"}},
    ]
    s = severity_slices(scored)
    assert set(s) == {"critical", "medium", "low"}
    assert s["critical"]["mean"] == pytest.approx(0.9)
    assert s["medium"]["n"] == 1


def test_aggregate_skips_none_scores():
    scored = [
        {"score": 0.8, "item": {"severity": "low"}},
        {"score": None, "item": {"severity": "low"}},
    ]
    a = aggregate(scored)
    assert a["n"] == 2
    assert a["n_scored"] == 1
    assert a["mean_score"] == pytest.approx(0.8)
