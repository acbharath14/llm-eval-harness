#!/usr/bin/env python3
"""Phase 1, step 2b: DeepEval AnswerRelevancy over saved responses.

One LLMTestCase per item (input=question, actual_output=saved response),
scored by AnswerRelevancyMetric with a local Ollama judge. Saves
deepeval_scores_<tag>.json incrementally (resume-safe).

NOTE: Faithfulness is intentionally NOT run here — the golden items carry
no retrieval contexts, and DeepEval's Faithfulness verifies claims against
retrieval_context. It gets its proper home in Phase 2 (Sidekick RAG),
where real retrieved passages exist.

Usage (from repo root):
    python3 frameworks/comparison/phase1/score_deepeval.py \\
        --responses frameworks/comparison/phase1/responses_ollama.json

~1-3 min per item on CPU (12 items). Progress prints per item.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# CPU-bound Ollama judges are slow: a single verdict round-trip can exceed
# DeepEval's default per-attempt timeout (~90s). Raise it for local runs.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

PHASE1_DIR = Path(__file__).resolve().parent


def tag_for(responses_path: str) -> str:
    stem = Path(responses_path).stem
    return stem[10:] if stem.startswith("responses_") else stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="responses JSON file")
    args = ap.parse_args()

    records = json.loads(Path(args.responses).read_text())
    tag = tag_for(args.responses)
    out_path = PHASE1_DIR / f"deepeval_scores_{tag}.json"

    done: dict[str, dict] = {}
    if out_path.exists():
        for entry in json.loads(out_path.read_text())["items"]:
            done[entry["id"]] = entry
        print(f"resuming: {len(done)} already scored")

    model_name = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    model = OllamaModel(model=model_name)
    metric = AnswerRelevancyMetric(model=model)

    items = []
    total_started = time.perf_counter()
    for i, r in enumerate(records, 1):
        if r["id"] in done:
            print(f"[{i:2d}/{len(records)}] {r['id']} — already scored, skipping")
            items.append(done[r["id"]])
            continue
        print(f"[{i:2d}/{len(records)}] {r['id']} — judging...", flush=True)
        started = time.perf_counter()
        try:
            metric.measure(LLMTestCase(input=r["question"], actual_output=r["response"]))
            entry = {
                "id": r["id"],
                "score": round(float(metric.score), 4),
                "reason": metric.reason,
                "elapsed_s": round(time.perf_counter() - started, 1),
            }
        except Exception as e:  # keep partial results; re-run resumes
            print(f"  FAILED: {e}")
            out_path.write_text(json.dumps(
                {"metric": "AnswerRelevancy", "model": model_name,
                 "items": items}, indent=1) + "\n")
            return 1
        items.append(entry)
        out_path.write_text(json.dumps(
            {"metric": "AnswerRelevancy", "model": model_name,
             "items": items}, indent=1) + "\n")
        print(f"  score={entry['score']} ({entry['elapsed_s']}s)")

    total = round(time.perf_counter() - total_started, 1)
    payload = {"metric": "AnswerRelevancy", "model": model_name,
               "total_elapsed_s": total, "items": items}
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"done: {len(items)} scored -> {out_path} ({total}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
