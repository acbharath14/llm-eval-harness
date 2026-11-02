#!/usr/bin/env python3
"""Phase 2, step 1: capture Sidekick RAG answers (capture-once, score-many).

Runs sidekick-rag-assistant's ask() over the fixed question set in
questions.json and saves one record per question:
  {id, category, question, expected, answer, grounded, latency_ms,
   sources, contexts: [{source, text, score}]}

The saved file is the frozen input for every scoring framework — no live
Sidekick calls during scoring, so results are comparable and re-runnable.

Black-box target: Sidekick is imported from its own repo via
--sidekick-root (no repo-to-repo imports, no code copied).

Determinism gotcha: ask() reads AND writes episodic memory, so answers
depend on question order. Questions always run in questions.json order;
do not reorder without re-capturing. For a clean capture, point Sidekick
at a fresh episodic store (see README).

Usage (from llm-eval-harness repo root, on the machine with Sidekick):
    python3 frameworks/comparison/phase2/capture_sidekick.py \\
        --sidekick-root /path/to/sidekick-rag-assistant

Resume-safe: already-captured ids are skipped.
~1-3 min per question on CPU (12 questions).
"""
import argparse
import json
import sys
import time
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidekick-root", required=True,
                    help="path to the sidekick-rag-assistant repo checkout")
    args = ap.parse_args()

    root = Path(args.sidekick_root).expanduser().resolve()
    src = root / "src"
    if not (src / "rag_demo" / "chain.py").exists():
        print(f"ERROR: {src}/rag_demo/chain.py not found — "
              f"is {root} the sidekick-rag-assistant repo?")
        return 1
    sys.path.insert(0, str(src))

    from rag_demo.chain import ask  # noqa: E402  (path set above)

    questions = json.loads((PHASE2_DIR / "questions.json").read_text())["items"]
    out_path = PHASE2_DIR / "sidekick_capture.json"

    done: dict[str, dict] = {}
    if out_path.exists():
        for entry in json.loads(out_path.read_text())["items"]:
            done[entry["id"]] = entry
        print(f"resuming: {len(done)} already captured")

    items = []
    total_started = time.perf_counter()
    for i, q in enumerate(questions, 1):
        if q["id"] in done:
            print(f"[{i:2d}/{len(questions)}] {q['id']} — already captured, skipping")
            items.append(done[q["id"]])
            continue
        print(f"[{i:2d}/{len(questions)}] {q['id']} — asking Sidekick...", flush=True)
        started = time.perf_counter()
        try:
            ans = ask(q["question"])
        except Exception as e:  # keep partial results; re-run resumes
            print(f"  FAILED: {e}")
            out_path.write_text(json.dumps(
                {"sidekick_root": str(root), "items": items}, indent=1) + "\n")
            return 1
        entry = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected": q["expected"],
            "answer": ans.text,
            "grounded": ans.grounded,
            "latency_ms": ans.latency_ms,
            "sources": ans.sources,
            "contexts": [
                {"source": h.source, "text": h.text, "score": h.score}
                for h in ans.hits
            ],
            "capture_elapsed_s": round(time.perf_counter() - started, 1),
        }
        items.append(entry)
        out_path.write_text(json.dumps(
            {"sidekick_root": str(root), "items": items}, indent=1) + "\n")
        print(f"  grounded={ans.grounded} "
              f"contexts={len(ans.hits)} ({entry['capture_elapsed_s']}s)")

    total = round(time.perf_counter() - total_started, 1)
    payload = {"sidekick_root": str(root),
               "total_elapsed_s": total, "items": items}
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"done: {len(items)} captured -> {out_path} ({total}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
