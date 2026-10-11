"""CLI runner: run an eval, compare against a baseline, exit on drift.

Usage:
    python -m eval_harness.runner --responder healthy|regressed \\
        --judge heuristic|llm --baseline baselines/baseline_results.json

Exit code 0 = no drift detected, 1 = drift detected (or run failure).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .golden_sets import load_golden_set
from .judges import HeuristicJudge, LLMJudge
from .responders import RESPONDERS
from .metrics import grounded_scores, augmented_keywords, aggregate
from . import drift as drift_mod


def run_eval(items: list[dict], responder, judge) -> list[dict]:
    scored = []
    for item in items:
        try:
            response = responder(item)
        except Exception as e:  # noqa: BLE001 - responder must not kill a run
            response = ""
            responder_error = str(e)
        else:
            responder_error = None
        judged = judge.score(item, response)
        g = grounded_scores(item, response)
        entry = {
            "id": item["id"],
            "category": item["category"],
            "severity": item["severity"],
            "score": judged.get("score"),
            "judge": judged.get("judge"),
            "keyword_coverage": judged.get("keyword_coverage"),
            "forbidden_hits": judged.get("forbidden_hits", []),
            "grounded_precision": g["grounded_precision"],
            "grounded_recall": g["grounded_recall"],
            "augmented_keywords": augmented_keywords(item, response),
            "response_preview": (response or "")[:200],
            "item": item,
        }
        if responder_error:
            entry["responder_error"] = responder_error
        if judged.get("skipped"):
            entry["judge_skipped_reason"] = judged.get("reason")
        scored.append(entry)
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LLM eval harness")
    parser.add_argument("--responder", choices=sorted(RESPONDERS), default="healthy")
    parser.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline_results.json for drift comparison")
    parser.add_argument("--golden", type=str, default=None,
                        help="Path to golden_qa.json (default: bundled)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write full JSON report to this path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N golden items (for quick smoke tests)")
    args = parser.parse_args(argv)

    try:
        items = load_golden_set(args.golden)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot load golden set: {e}", file=sys.stderr)
        return 1

    if args.limit is not None:
        if args.limit < 1:
            print("ERROR: --limit must be >= 1", file=sys.stderr)
            return 1
        items = items[:args.limit]

    responder = RESPONDERS[args.responder]
    judge = LLMJudge() if args.judge == "llm" else HeuristicJudge()

    scored = run_eval(items, responder, judge)
    report = {
        "responder": args.responder,
        "judge": args.judge,
        "aggregate": aggregate(scored),
        "items": [{k: v for k, v in e.items() if k != "item"} | {"item_id": e["item"]["id"]}
                   for e in scored],
    }

    drift_result = None
    baseline_scores = None
    if args.baseline:
        try:
            with open(args.baseline, encoding="utf-8") as f:
                baseline_data = json.load(f)
            baseline_scores = baseline_data["aggregate"]["per_item_scores"]
            current_scores = [e["score"] for e in scored if e["score"] is not None]
            drift_result = drift_mod.detect_drift(baseline_scores, current_scores)
            report["drift"] = drift_result
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"WARNING: baseline comparison skipped: {e}", file=sys.stderr)

    print(json.dumps({
        "responder": report["responder"],
        "judge": report["judge"],
        "aggregate": report["aggregate"],
        "drift": drift_result,
    }, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Full report written to {args.output}", file=sys.stderr)

    if drift_result and drift_result["drift_detected"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
