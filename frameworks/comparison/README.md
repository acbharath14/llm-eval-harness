# Framework comparison (Phase 1)

`subset.json` is the fixed 12-item dataset: 2 per category
(adversarial, api, billing, limits, security, troubleshooting),
severity-spread, with full item bodies. Do not change the IDs without
re-running all three frameworks.

## Method (when Phase 1 starts)

1. Generate responses once via the harness's ollama responder
   (`qwen3:8b`) for these 12 IDs; save to `responses.json`. All three
   frameworks score the *same* saved responses.
2. Promptfoo: assert pass/fail on expected_keywords / must_not_contain
   (mirrors the heuristic judge).
3. DeepEval: AnswerRelevancy + Faithfulness via OllamaModel.
4. Ragas: faithfulness + answer_relevancy (+ context_precision with
   ideal_answer as ground truth).
5. Negative control: run the regressed responder's outputs through each
   framework — do the scores distinguish healthy from regressed?
6. Write `report.md`: per-framework scores, agreement-with-heuristic
   table, CPU runtimes, friction notes, recommendation.
