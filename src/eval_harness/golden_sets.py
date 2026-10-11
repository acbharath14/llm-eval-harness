"""Load and validate the golden Q/A dataset."""
from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FIELDS = {
    "id", "category", "severity", "question",
    "expected_keywords", "must_not_contain", "ideal_answer",
}
VALID_SEVERITIES = {"critical", "medium", "low"}

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "golden_qa.json"


def load_golden_set(path: str | Path | None = None) -> list[dict]:
    """Load golden items, validating schema. Raises ValueError on problems."""
    p = Path(path) if path else DEFAULT_PATH
    with open(p, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Golden set at {p} must be a non-empty list")
    seen_ids: set[str] = set()
    for idx, item in enumerate(items):
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            raise ValueError(f"Item {idx} missing fields: {sorted(missing)}")
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate id: {item['id']}")
        seen_ids.add(item["id"])
        if item["severity"] not in VALID_SEVERITIES:
            raise ValueError(f"Item {item['id']} has invalid severity: {item['severity']}")
        if not isinstance(item["expected_keywords"], list):
            raise ValueError(f"Item {item['id']}: expected_keywords must be a list")
        if not isinstance(item["must_not_contain"], list):
            raise ValueError(f"Item {item['id']}: must_not_contain must be a list")
    return items


def filter_by_category(items: list[dict], category: str) -> list[dict]:
    return [it for it in items if it["category"] == category]


def filter_by_severity(items: list[dict], severity: str) -> list[dict]:
    return [it for it in items if it["severity"] == severity]
