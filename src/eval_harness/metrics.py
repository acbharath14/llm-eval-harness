"""ReviewBench-inspired metrics.

- Grounded precision/recall: measured against the golden expected_keywords.
- Augmented (diagnostic): novel content words in the response that are NOT in
  the expected set — surfaced for inspection, never penalized.
- Severity slices: aggregate scores by critical/medium/low.
"""
from __future__ import annotations

import re
from collections import defaultdict

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been it its "
    "this that these those you your we our they their he she him her as at "
    "by from can will would should could may might do does did not no yes "
    "if then than so such via per all any each other into out up down over "
    "under my me i".split()
)


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) - _STOPWORDS


def grounded_scores(item: dict, response: str) -> dict:
    """Precision/recall of expected keywords in the response (grounded)."""
    expected = {kw.lower() for kw in item.get("expected_keywords", [])}
    lowered = (response or "").lower()
    hits = {kw for kw in expected if kw in lowered}
    recall = len(hits) / len(expected) if expected else 1.0
    # Grounded "precision": fraction of response keyword-ish words that are expected.
    resp_words = _words(response or "")
    expected_words = set()
    for kw in expected:
        expected_words |= _words(kw)
    overlap = resp_words & expected_words
    precision = len(overlap) / len(resp_words) if resp_words else 0.0
    return {
        "grounded_precision": round(precision, 4),
        "grounded_recall": round(recall, 4),
        "keywords_hit": sorted(hits),
        "keywords_missed": sorted(expected - hits),
    }


def augmented_keywords(item: dict, response: str, top_n: int = 10) -> list[str]:
    """Novel content words in the response not covered by expected keywords.

    Diagnostic only — a system is never penalized for novel true content.
    """
    expected_words = set()
    for kw in item.get("expected_keywords", []):
        expected_words |= _words(kw)
    expected_words |= _words(item.get("ideal_answer", ""))
    novel = _words(response or "") - expected_words
    return sorted(novel)[:top_n]


def severity_slices(scored_items: list[dict]) -> dict[str, dict]:
    """Aggregate {mean, min, max, n} of scores per severity level."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for entry in scored_items:
        score = entry.get("score")
        if score is None:
            continue
        buckets[entry["item"]["severity"]].append(score)
    out = {}
    for sev in ("critical", "medium", "low"):
        vals = buckets.get(sev, [])
        out[sev] = {
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 4) if vals else 0.0,
            "min": round(min(vals), 4) if vals else 0.0,
            "max": round(max(vals), 4) if vals else 0.0,
        }
    return out


def aggregate(scored_items: list[dict]) -> dict:
    scores = [e["score"] for e in scored_items if e.get("score") is not None]
    mean = sum(scores) / len(scores) if scores else 0.0
    return {
        "n": len(scored_items),
        "n_scored": len(scores),
        "mean_score": round(mean, 4),
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "by_severity": severity_slices(scored_items),
    }
