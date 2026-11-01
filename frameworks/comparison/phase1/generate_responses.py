#!/usr/bin/env python3
"""Phase 1, step 1: generate one fixed response set per responder.

Reads the frozen 12-item subset, runs a harness responder over every item,
and saves the responses. Every framework in Phase 1 scores THESE saved
responses — never live generations — so results are comparable and
re-runnable.

Usage (from repo root):
    python3 frameworks/comparison/phase1/generate_responses.py --responder ollama
    python3 frameworks/comparison/phase1/generate_responses.py --responder regressed

Output: frameworks/comparison/phase1/responses_<responder>.json
Each record: id, category, severity, question, expected_keywords,
must_not_contain, ideal_answer, response.

Env:
    OLLAMA_MODEL   (default qwen3:8b)
    OLLAMA_TIMEOUT (default 300s — CPU judges need headroom)

The ollama run takes ~15-35 min on CPU for 12 items. Progress prints per
item and the file is saved incrementally, so Ctrl+C loses nothing already
captured — re-running resumes where it stopped.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
ROOT = PHASE1_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_harness.responders import RESPONDERS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responder", choices=sorted(RESPONDERS), required=True)
    ap.add_argument("--out", default=None,
                    help="output path (default: phase1/responses_<responder>.json)")
    args = ap.parse_args()

    os.environ.setdefault("OLLAMA_TIMEOUT", "300")

    subset = json.loads((ROOT / "frameworks" / "comparison" / "subset.json").read_text())
    items = subset["items"]
    responder = RESPONDERS[args.responder]
    out_path = Path(args.out) if args.out else PHASE1_DIR / f"responses_{args.responder}.json"

    records: list[dict] = []
    if out_path.exists():
        records = json.loads(out_path.read_text())
        done = {r["id"] for r in records}
        print(f"resuming: {len(done)} already captured")
    else:
        done = set()

    started = time.perf_counter()
    for i, item in enumerate(items, 1):
        if item["id"] in done:
            print(f"[{i:2d}/{len(items)}] {item['id']} — already captured, skipping")
            continue
        print(f"[{i:2d}/{len(items)}] {item['id']} ({args.responder})...", flush=True)
        try:
            response = responder(item)
        except Exception as e:  # e.g. Ollama unreachable — keep partial file
            print(f"  FAILED: {e}")
            out_path.write_text(json.dumps(records, indent=1) + "\n")
            return 1
        records.append({
            "id": item["id"],
            "category": item["category"],
            "severity": item["severity"],
            "question": item["question"],
            "expected_keywords": item.get("expected_keywords", []),
            "must_not_contain": item.get("must_not_contain", []),
            "ideal_answer": item.get("ideal_answer", ""),
            "response": response,
        })
        out_path.write_text(json.dumps(records, indent=1) + "\n")
        print(f"  saved ({len(response)} chars)")

    print(f"done: {len(records)} responses -> {out_path} "
          f"({time.perf_counter() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
