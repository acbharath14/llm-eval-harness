# Phase 1: same-dataset comparison

One frozen response set per responder, scored by every framework. No live
generation during scoring — results are comparable and re-runnable.

## Run order (from repo root)

**Step 1 — generate responses** (~15-35 min on CPU for the ollama set;
regressed is instant; resume-safe):

```bash
python3 frameworks/comparison/phase1/generate_responses.py --responder ollama
python3 frameworks/comparison/phase1/generate_responses.py --responder regressed
```

**Step 2a — promptfoo** (seconds per set):

```bash
python3 frameworks/comparison/phase1/score_promptfoo.py --responses responses_ollama.json
python3 frameworks/comparison/phase1/score_promptfoo.py --responses responses_regressed.json
cd frameworks/comparison/phase1
COMPARISON_RESPONSES=responses_ollama.json npx promptfoo@latest eval -c promptfoo_ollama.json -o pf_results_ollama.json
COMPARISON_RESPONSES=responses_regressed.json npx promptfoo@latest eval -c promptfoo_regressed.json -o pf_results_regressed.json
cd ../../..
```

The asserts mirror the heuristic judge: `icontains` per expected keyword,
`not-icontains` per must_not_contain phrase.

**Step 2b — DeepEval** (~15-30 min per set on CPU; resume-safe):

```bash
python3 frameworks/comparison/phase1/score_deepeval.py --responses frameworks/comparison/phase1/responses_ollama.json
python3 frameworks/comparison/phase1/score_deepeval.py --responses frameworks/comparison/phase1/responses_regressed.json
```

AnswerRelevancy only. Faithfulness needs retrieval contexts, which the
golden items don't carry — it runs in Phase 2 against Sidekick RAG.

**Step 3 — report:**

```bash
python3 frameworks/comparison/phase1/build_report.py
```

Writes `report.md`: per-item verdicts, agreement-with-heuristic rates,
healthy-vs-regressed separation, runtimes, friction notes, recommendation.

## Design notes

- Verdict threshold 0.5 throughout (documented choice): heuristic and
  DeepEval scores are 0-1, promptfoo is binary pass/fail.
- `provider.py` is promptfoo's file provider — serves the saved response
  for each test's `id` var. If promptfoo's file-provider API changed,
  paste the error back.
- Intermediates (`responses_*.json`, `promptfoo_*.json`,
  `pf_results_*.json`, `deepeval_scores_*.json`) are gitignored;
  `report.md` is the committed deliverable.
