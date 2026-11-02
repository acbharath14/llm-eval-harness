#!/usr/bin/env python3
"""Phase 2, step 3: RAG-metrics report.

Reads the frozen Sidekick capture and the DeepEval RAG scores and writes
report.md:

  - per-question table: Faithfulness / ContextualRecall / ContextualRelevancy
  - means across scored questions, skipped count
  - metric interpretation + what the scores say about Sidekick's retrieval
  - comparison to Phase 1 (AnswerRelevancy = topicality, Faithfulness =
    the correctness check Phase 1 lacked)
  - recommendation for the retrieval config

Handles missing inputs gracefully: no scores -> "scores pending" report;
no capture -> "capture pending" report.

Usage (from repo root), after steps 1-2:
    python3 frameworks/comparison/phase2/build_report.py

Expected inputs in frameworks/comparison/phase2/:
    questions.json (always present — frozen 12-item set),
    sidekick_capture.json (step 1), deepeval_rag_scores.json (step 2)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parent
ROOT = PHASE2_DIR.parents[2]

METRICS = ("faithfulness", "contextual_recall", "contextual_relevancy")
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "contextual_recall": "ContextualRecall",
    "contextual_relevancy": "ContextualRelevancy",
}
THRESHOLD = 0.5  # documented choice, same as Phase 1


def load_json(name: str):
    p = PHASE2_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def normalize_scores(raw):
    """Accept the scorer's {"items": [...]} shape or a flat
    {question_id: {metric: score}} dict; return id -> entry dict."""
    if raw is None:
        return {}, None, None
    if isinstance(raw, dict) and "items" in raw:
        model = raw.get("model")
        total = raw.get("total_elapsed_s")
        return {e["id"]: e for e in raw["items"]}, model, total
    if isinstance(raw, dict):
        return raw, raw.get("model"), raw.get("total_elapsed_s")
    return {}, None, None


def fmt(v):
    return f"{v:.2f}" if v is not None else "—"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    questions_raw = load_json("questions.json")
    questions = {q["id"]: q for q in (questions_raw or {}).get("items", [])}
    capture = load_json("sidekick_capture.json")
    scores_raw = load_json("deepeval_rag_scores.json")
    scores, model, total_s = normalize_scores(scores_raw)

    lines = []
    A = lines.append
    A("# Phase 2 report: DeepEval RAG metrics vs sidekick-rag-assistant")
    A("")
    A("12 fixed questions (RAG-01..RAG-12) over the Meridian Logistics demo "
      "corpus. Capture-once, score-many: every metric below scores the same "
      "frozen `sidekick_capture.json` — no live Sidekick calls during "
      "scoring.")
    A("")

    # ---- provenance ----
    if capture:
        items = capture.get("items", [])
        n_ctx = sum(len(i.get("contexts", [])) for i in items)
        grounded = sum(1 for i in items if i.get("grounded"))
        A(f"Capture: {len(items)} items, {grounded} grounded, "
          f"{n_ctx} total contexts.")
    else:
        A("Capture: **pending** — run "
          "`python3 frameworks/comparison/phase2/capture_sidekick.py "
          "--sidekick-root <path>` first.")
    if scores_raw:
        A(f"Scoring model: `{model or 'unknown'}`" +
          (f", {total_s:.0f}s total" if total_s else ""))
    else:
        A("Scores: **pending** — run "
          "`python3 frameworks/comparison/phase2/score_rag_deepeval.py`.")
    A("")

    if not scores_raw:
        A("## Status: scores pending")
        A("")
        A("Re-run this script after step 2 (scoring) completes; the tables "
          "below will populate. Nothing else in the harness changed — "
          "Phase 1's layered-evaluation recommendation stands.")
        out = PHASE2_DIR / "report.md"
        out.write_text("\n".join(lines) + "\n")
        print(f"wrote {out} (scores pending)")
        return 0

    # ---- per-question table ----
    scored = {qid: e for qid, e in scores.items() if "skipped" not in e}
    skipped = [qid for qid, e in scores.items() if "skipped" in e]
    order = [q["id"] for q in (questions_raw or {}).get("items", [])] or \
        sorted(scores)

    A("## Per-question scores")
    A("")
    A(f"Verdict threshold {THRESHOLD} (documented choice, same as Phase 1). "
      "Items with no retrieval contexts were skipped, not scored.")
    A("")
    A("| id | category | faithfulness | recall | relevancy | grounded |")
    A("|---|---|---|---|---|---|")
    for qid in order:
        q = questions.get(qid, {})
        e = scores.get(qid)
        if e is None:
            A(f"| {qid} | {q.get('category', '?')} | not scored | | | |")
            continue
        if "skipped" in e:
            A(f"| {qid} | {q.get('category', '?')} | skipped ({e.get('skipped', '')}) "
              f"| | | {'yes' if e.get('grounded') else 'no'} |")
            continue
        g = "yes" if e.get("grounded") else "no"
        A(f"| {qid} | {q.get('category', '?')} | {fmt(e.get('faithfulness'))} "
          f"| {fmt(e.get('contextual_recall'))} | "
          f"{fmt(e.get('contextual_relevancy'))} | {g} |")
    A("")
    A(f"Scored: {len(scored)}/{len(order)} — skipped: {len(skipped)}"
      + (f" ({', '.join(skipped)})" if skipped else ""))
    A("")

    # ---- means ----
    A("## Means (scored items only)")
    A("")
    A("| metric | mean | below-0.5 count |")
    A("|---|---|---|")
    for m in METRICS:
        vals = [scores[qid].get(m) for qid in scored]
        below = sum(1 for v in vals if v is not None and v < THRESHOLD)
        mval = mean(vals)
        A(f"| {METRIC_LABELS[m]} | {fmt(mval)} | {below} |")
    A("")

    # ---- interpretation ----
    A("## What the metrics mean (and what the scores say)")
    A("")
    A("- **Faithfulness** (0-1): do the answer's claims hold against the "
      "retrieved contexts? This is the *correctness* check Phase 1 lacked — "
      "AnswerRelevancy scored topicality, not whether the answer was right. "
      "Low faithfulness means the LLM is drifting beyond what retrieval "
      "found (a generation/prompt problem, not a retrieval problem).")
    A("- **ContextualRecall** (0-1): does the answer cover the expected key "
      "facts (the `expected` field per question)? Low recall means the key "
      "facts never made it into the answer — either retrieval didn't fetch "
      "them or generation dropped them.")
    A("- **ContextualRelevancy** (0-1): are the retrieved contexts relevant "
      "to the question? This is the pure retrieval-quality signal. The "
      "overhaul spec made the hybrid-default decision conditional on this "
      "metric (flip dense→hybrid if mean relevancy < 0.7).")
    A("")

    # weakest-question callout
    weakest = {}
    for m in METRICS:
        cand = [(qid, scores[qid].get(m)) for qid in scored
                if scores[qid].get(m) is not None]
        if cand:
            weakest[m] = min(cand, key=lambda kv: kv[1])
    if weakest:
        A("Weakest per metric:")
        A("")
        for m in METRICS:
            if m in weakest:
                qid, v = weakest[m]
                A(f"- {METRIC_LABELS[m]}: {qid} ({v:.2f}) — "
                  f"\"{questions.get(qid, {}).get('question', '')[:60]}…\"")
        A("")

    # ---- comparison to Phase 1 ----
    A("## Comparison to Phase 1")
    A("")
    A("Phase 1 (same 12-item subset, ollama responder, qwen3:8b): "
      "DeepEval AnswerRelevancy scored the ollama set 12/12, mean ≈ 0.96 — "
      "but also scored vague regressed non-answers 1.0 (AnswerRelevancy "
      "measures topicality, not correctness; agreement with the heuristic "
      "judge was only 42%). promptfoo keyword asserts went 3/12 on ollama, "
      "0/12 on regressed (brittle but honest).")
    A("")
    A("Phase 2's Faithfulness is the metric Phase 1's Recommendation asked "
      "for ('for correctness, prefer Faithfulness or a custom G-Eval "
      "criterion'). Where Phase 1 could only say 'the answer is on topic', "
      "Phase 2 can say 'the answer's claims are grounded in what retrieval "
      "returned'. The two phases are complementary, not competing: Phase 1 "
      "compared frameworks, Phase 2 measures the RAG system itself.")
    A("")

    # ---- RAG-10 caveat ----
    A("## Caveat: capture predates the overhaul fixes")
    A("")
    A("The capture (2026-11-01) predates two Sidekick fixes that landed on "
      "`feat/fastapi-backend` the same evening: the relevance-floor default "
      "0.5→0 (RAG-10 returned 0 hits because the MS MARCO cross-encoder "
      "scores the fictional Meridian corpus below 0.5 — best hit 0.371) and "
      "the episodic-memory rewrite isolation (episodic Q&As no longer feed "
      "`rewrite_query()`). RAG-10's captured answer reflects the old "
      "behavior (grounded=False, 0 contexts). A re-capture with the fixes "
      "is the acceptance gate before treating these numbers as the "
      "overhaul baseline.")
    A("")

    # ---- recommendation ----
    A("## Recommendation")
    A("")
    rel_mean = mean([scores[qid].get("contextual_relevancy") for qid in scored])
    fai_mean = mean([scores[qid].get("faithfulness") for qid in scored])
    A("1. **Re-run scoring after re-capture.** The numbers above measure the "
      "pre-fix pipeline; the floor and episodic fixes specifically target "
      "retrieval behavior (RAG-10) that these metrics score. Do not tune "
      "further on stale numbers.")
    if rel_mean is not None:
        A(f"2. **Hybrid-default decision:** mean ContextualRelevancy "
          f"{rel_mean:.2f} " +
          ("is below the 0.7 bar — after re-capture, if it stays there, "
           "flip the default retrieval to hybrid (spec Part 3.2)." if rel_mean < 0.7
           else "clears the 0.7 bar — keep dense as the default and let "
                "hybrid stay opt-in."))
    if fai_mean is not None and fai_mean < THRESHOLD:
        A(f"3. **Faithfulness below 0.5 ({fai_mean:.2f})** points at "
          "generation, not retrieval: the answer drifts beyond the "
          "contexts. Look at the system prompt / chain before touching "
          "retrieval.")
    A("4. **Adopt the Phase 1 layering for RAG:** heuristic checks where "
      "they apply, Faithfulness as the offline correctness gate for "
      "retrieval changes (it caught what AnswerRelevancy couldn't), "
      "ContextualRelevancy as the retrieval-quality dial. Keep these runs "
      "CPU-offline — they are slow by design, not by accident.")
    A("")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    A(f"_Generated {stamp} by "
      "`frameworks/comparison/phase2/build_report.py`._")

    out = PHASE2_DIR / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print(f"scored {len(scored)}/{len(order)}, skipped {len(skipped)}")
    for m in METRICS:
        mval = mean([scores[qid].get(m) for qid in scored])
        print(f"  {METRIC_LABELS[m]}: {fmt(mval)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
