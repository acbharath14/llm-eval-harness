# Phase 2: RAG metrics vs sidekick-rag-assistant

DeepEval's RAG metrics (Faithfulness, ContextualRecall, ContextualRelevancy)
against a real RAG system. Sidekick is a **black-box target** — the eval
imports nothing from it at module level; the capture script takes
`--sidekick-root` and imports `rag_demo.chain.ask` from that checkout.

ragas is parked (see Phase 1 report), so DeepEval carries the RAG metrics.

## Run order (from repo root, on the machine with Sidekick + Ollama)

**Step 1 — capture** (~15-35 min on CPU for 12 questions; resume-safe):

```bash
python3 frameworks/comparison/phase2/capture_sidekick.py \
    --sidekick-root /path/to/sidekick-rag-assistant
```

Writes `sidekick_capture.json`: one record per question with the answer,
`grounded` flag, latency, sources, and the full retrieved contexts
(`contexts: [{source, text, score}]`).

**Step 2 — score** (~2-4 min per item per metric on CPU; resume-safe):

```bash
python3 frameworks/comparison/phase2/score_rag_deepeval.py
```

Writes `deepeval_rag_scores.json`: faithfulness, contextual_recall, and
contextual_relevancy per item, judged by `OLLAMA_MODEL` (default qwen3:8b).

## Design notes

- **Capture-once, score-many.** Scoring never calls Sidekick — every
  framework scores the same frozen `sidekick_capture.json`.
- **Fixed question order.** `ask()` reads and writes episodic memory, so
  answers are order-dependent. Questions run in `questions.json` order;
  do not reorder without re-capturing. For a clean capture, use a fresh
  episodic store (Sidekick's default location — back it up or point
  Sidekick at an empty one before capturing).
- **Ungrounded answers are skipped in scoring.** If Sidekick answers from
  general knowledge (`grounded: false`, no contexts), there is nothing
  for Faithfulness/ContextualRelevancy to verify — the item is recorded
  as skipped, which is itself a finding about retrieval coverage.
- **Intermediates are gitignored** (`sidekick_capture.json`,
  `deepeval_rag_scores.json`); the committed deliverable is the report
  (build_report.py — to be added once scores land).
- The `expected` field in `questions.json` holds the key facts a good
  answer should contain; it feeds ContextualRecall's `expected_output`.
