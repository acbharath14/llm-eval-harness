"""Judges that score a response against a golden item.

HeuristicJudge: deterministic, keyword-based. Uses hashlib.md5 only —
never Python's randomized hash().
LLMJudge: optional OpenAI-compatible API judge with graceful fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import urllib.error

MIN_LEN = 20
MAX_LEN = 2000
FORBIDDEN_PENALTY = 0.3


def _stable_bucket(text: str, n_buckets: int = 1000) -> int:
    """Deterministic bucket for a string (md5, not hash())."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_buckets


class HeuristicJudge:
    """Deterministic keyword-coverage judge. Scores 0.0-1.0."""

    def score(self, item: dict, response: str) -> dict:
        response = response or ""
        lowered = response.lower()

        keywords = item.get("expected_keywords", [])
        if keywords:
            hits = sum(1 for kw in keywords if kw.lower() in lowered)
            keyword_score = hits / len(keywords)
        else:
            keyword_score = 1.0

        penalty = 0.0
        forbidden_hits = []
        for phrase in item.get("must_not_contain", []):
            if phrase.lower() in lowered:
                penalty += FORBIDDEN_PENALTY
                forbidden_hits.append(phrase)

        length_penalty = 0.0
        if len(response) < MIN_LEN:
            length_penalty = 0.25
        elif len(response) > MAX_LEN:
            length_penalty = 0.15

        raw = keyword_score - penalty - length_penalty
        final = max(0.0, min(1.0, raw))

        return {
            "score": round(final, 4),
            "keyword_coverage": round(keyword_score, 4),
            "keywords_hit": [kw for kw in keywords if kw.lower() in lowered],
            "forbidden_hits": forbidden_hits,
            "length_penalty": length_penalty,
            "judge": "heuristic",
        }


class LLMJudge:
    """OpenAI-compatible API judge. Falls back gracefully without a key."""

    def __init__(self, api_base: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: int = 30):
        self.api_base = (api_base or os.environ.get("JUDGE_API_BASE",
                         "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", "")
        self.model = model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        # Ollama's local API needs no key — detect by host.
        if "localhost" in self.api_base or "127.0.0.1" in self.api_base:
            return True
        return bool(self.api_key)

    def _prompt(self, item: dict, response: str) -> str:
        return (
            "You are an impartial evaluator. Score the RESPONSE against the "
            "REFERENCE answer for the QUESTION below.\n\n"
            f"QUESTION: {item['question']}\n\n"
            f"REFERENCE: {item['ideal_answer']}\n\n"
            f"RESPONSE: {response}\n\n"
            "Expected key points: " + ", ".join(item.get("expected_keywords", [])) + "\n"
            "The response must NOT contain: " + ", ".join(item.get("must_not_contain", []) or ["(none)"]) + "\n\n"
            'Reply with JSON only: {"score": <0-10 integer>, "reasoning": "<one sentence>"}'
        )

    def score(self, item: dict, response: str) -> dict:
        if not self.available:
            return {"score": None, "judge": "llm", "skipped": True,
                    "reason": "JUDGE_API_KEY not set"}
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": self._prompt(item, response)}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            raw = float(parsed.get("score", 0))
            return {"score": round(max(0.0, min(1.0, raw / 10.0)), 4),
                    "reasoning": parsed.get("reasoning", ""),
                    "judge": "llm", "model": self.model}
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as e:
            return {"score": None, "judge": "llm", "skipped": True,
                    "reason": f"LLM judge call failed: {e}"}
