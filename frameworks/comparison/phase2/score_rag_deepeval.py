#!/usr/bin/env python3
"""Phase 2, step 2: DeepEval RAG metrics over the frozen Sidekick capture.

Scores each captured item with three RAG metrics, all judged by a local
Ollama model:
  - Faithfulness: do the answer's claims hold against the retrieved contexts?
  - ContextualRecall: does the answer cover the expected key facts?
  - ContextualRelevancy: are the retrieved contexts relevant to the question?

Saves deepeval_rag_scores.json incrementally (resume-safe). No live
Sidekick calls — everything comes from sidekick_capture.json.

Usage (from repo root):
    python3 frameworks/comparison/phase2/score_rag_deepeval.py

~2-4 min per item per metric on CPU (12 items x 3 metrics). Progress
prints per item.
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
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

PHASE2_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=str(PHASE2_DIR / "sidekick_capture.json"),
                    help="frozen capture file from capture_sidekick.py")
    args = ap.parse_args()

    capture_path = Path(args.capture)
    if not capture_path.exists():
        print(f"ERROR: {capture_path} not found — "
              f"run capture_sidekick.py first")
        return 1
    records = json.loads(capture_path.read_text())["items"]
    out_path = PHASE2_DIR / "deepeval_rag_scores.json"

    done: dict[str, dict] = {}
    if out_path.exists():
        for entry in json.loads(out_path.read_text())["items"]:
            done[entry["id"]] = entry
        print(f"resuming: {len(done)} already scored")

    model_name = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    model = OllamaModel(model=model_name)
    metrics = {
        "faithfulness": FaithfulnessMetric(model=model),
        "contextual_recall": ContextualRecallMetric(model=model),
        "contextual_relevancy": ContextualRelevancyMetric(model=model),
    }

    items = []
    total_started = time.perf_counter()
    for i, r in enumerate(records, 1):
        if r["id"] in done:
            print(f"[{i:2d}/{len(records)}] {r['id']} — already scored, skipping")
            items.append(done[r["id"]])
            continue
        contexts = [c["text"] for c in r["contexts"]]
        if not contexts:
            print(f"[{i:2d}/{len(records)}] {r['id']} — no contexts, "
                  f"skipping (small-talk or ungrounded)")
            entry = {"id": r["id"], "skipped": "no retrieval contexts",
                     "grounded": r["grounded"]}
            items.append(entry)
            out_path.write_text(json.dumps(
                {"model": model_name, "items": items}, indent=1) + "\n")
            continue
        print(f"[{i:2d}/{len(records)}] {r['id']} — judging "
              f"({len(contexts)} contexts)...", flush=True)
        started = time.perf_counter()
        entry: dict = {"id": r["id"], "grounded": r["grounded"]}
        try:
            case = LLMTestCase(
                input=r["question"],
                actual_output=r["answer"],
                expected_output=r["expected"],
                retrieval_context=contexts,
            )
            for name, metric in metrics.items():
                metric.measure(case)
                entry[name] = round(float(metric.score), 4)
            entry["elapsed_s"] = round(time.perf_counter() - started, 1)
        except Exception as e:  # keep partial results; re-run resumes
            print(f"  FAILED: {e}")
            out_path.write_text(json.dumps(
                {"model": model_name, "items": items}, indent=1) + "\n")
            return 1
        items.append(entry)
        out_path.write_text(json.dumps(
            {"model": model_name, "items": items}, indent=1) + "\n")
        print(f"  faithfulness={entry['faithfulness']} "
              f"recall={entry['contextual_recall']} "
              f"relevancy={entry['contextual_relevancy']} "
              f"({entry['elapsed_s']}s)")

    total = round(time.perf_counter() - total_started, 1)
    payload = {"model": model_name,
               "total_elapsed_s": total, "items": items}
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"done: {len(items)} scored -> {out_path} ({total}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
