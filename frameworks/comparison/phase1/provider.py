"""Promptfoo file provider: serves Phase 1's saved responses.

The test case's `id` var selects the pre-generated response, so promptfoo
scores fixed outputs instead of calling an LLM. Set COMPARISON_RESPONSES
to the responses file being scored (relative to the phase1 dir, which is
where promptfoo runs).
"""
import json
import os

_RESP_PATH = os.environ.get("COMPARISON_RESPONSES", "responses_ollama.json")

with open(_RESP_PATH, encoding="utf-8") as f:
    _RESPONSES = {r["id"]: r["response"] for r in json.load(f)}


def call_api(prompt, options, context):
    item_id = (context.get("vars") or {}).get("id")
    return {"output": _RESPONSES.get(item_id, "")}
