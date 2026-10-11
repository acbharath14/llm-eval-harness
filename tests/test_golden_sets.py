"""Tests for golden set loading and validation."""
import json
import pytest

from eval_harness.golden_sets import (
    load_golden_set, filter_by_category, filter_by_severity,
)


def test_loads_50_items():
    items = load_golden_set()
    assert len(items) == 50


def test_all_required_fields_present():
    required = {"id", "category", "severity", "question",
                "expected_keywords", "must_not_contain", "ideal_answer"}
    for item in load_golden_set():
        assert required <= set(item), item["id"]


def test_ids_unique():
    ids = [it["id"] for it in load_golden_set()]
    assert len(ids) == len(set(ids))


def test_severities_valid():
    for item in load_golden_set():
        assert item["severity"] in {"critical", "medium", "low"}


def test_categories_covered():
    cats = {it["category"] for it in load_golden_set()}
    assert {"billing", "security", "troubleshooting", "limits", "api", "adversarial"} <= cats


def test_rejects_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        load_golden_set(tmp_path / "nope.json")


def test_rejects_invalid_item(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "X-001"}]))
    with pytest.raises(ValueError, match="missing fields"):
        load_golden_set(bad)


def test_rejects_duplicate_ids(tmp_path):
    dup = tmp_path / "dup.json"
    item = {"id": "A-001", "category": "api", "severity": "low",
            "question": "q", "expected_keywords": [], "must_not_contain": [],
            "ideal_answer": "a"}
    dup.write_text(json.dumps([item, item]))
    with pytest.raises(ValueError, match="Duplicate id"):
        load_golden_set(dup)


def test_filter_helpers():
    items = load_golden_set()
    billing = filter_by_category(items, "billing")
    assert billing and all(it["category"] == "billing" for it in billing)
    crit = filter_by_severity(items, "critical")
    assert crit and all(it["severity"] == "critical" for it in crit)
