"""Tests for the heuristic and LLM judges."""
import pytest

from eval_harness.judges import HeuristicJudge, LLMJudge, _stable_bucket

ITEM = {
    "id": "TST-001", "category": "api", "severity": "medium",
    "question": "How do I authenticate?",
    "expected_keywords": ["API key", "Authorization", "Bearer"],
    "must_not_contain": ["username and password"],
    "ideal_answer": "Pass your API key in the Authorization header as Bearer token.",
}


def test_perfect_answer_scores_high():
    j = HeuristicJudge()
    r = j.score(ITEM, ITEM["ideal_answer"])
    assert r["score"] >= 0.9
    assert r["judge"] == "heuristic"


def test_keyword_coverage_fraction():
    j = HeuristicJudge()
    # Only 1 of 3 keywords present.
    r = j.score(ITEM, "You need an API key for this.")
    assert r["keyword_coverage"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["score"] == pytest.approx(1 / 3, abs=1e-3)


def test_forbidden_phrase_penalty():
    j = HeuristicJudge()
    good = "Use your API key in the Authorization Bearer header."
    bad = good + " Just use username and password instead."
    assert j.score(ITEM, bad)["score"] < j.score(ITEM, good)["score"]
    assert "username and password" in j.score(ITEM, bad)["forbidden_hits"]


def test_short_answer_penalized():
    j = HeuristicJudge()
    r = j.score(ITEM, "API key.")
    assert r["length_penalty"] > 0


def test_score_clamped():
    j = HeuristicJudge()
    r = j.score(ITEM, "username and password username and password")
    assert 0.0 <= r["score"] <= 1.0


def test_empty_response_scores_zero():
    j = HeuristicJudge()
    r = j.score(ITEM, "")
    assert r["score"] == 0.0


def test_case_insensitive_matching():
    j = HeuristicJudge()
    r = j.score(ITEM, "YOUR API KEY GOES IN THE AUTHORIZATION BEARER HEADER PLEASE")
    assert r["keyword_coverage"] == 1.0


def test_deterministic_across_runs():
    j = HeuristicJudge()
    a = j.score(ITEM, "Some response with API key and Bearer.")
    b = j.score(ITEM, "Some response with API key and Bearer.")
    assert a == b


def test_stable_bucket_uses_md5_not_hash():
    # Same input -> same bucket, every process run.
    assert _stable_bucket("hello") == _stable_bucket("hello")
    assert 0 <= _stable_bucket("hello", 100) < 100


def test_llm_judge_skipped_without_key(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    j = LLMJudge(api_key="")
    assert not j.available
    r = j.score(ITEM, "anything")
    assert r["skipped"] is True
    assert r["score"] is None


def test_llm_judge_available_for_ollama_without_key(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    j = LLMJudge(api_base="http://localhost:11434/v1", api_key="")
    assert j.available is True


def test_llm_judge_available_with_key():
    j = LLMJudge(api_base="https://api.openai.com/v1", api_key="sk-test")
    assert j.available is True
