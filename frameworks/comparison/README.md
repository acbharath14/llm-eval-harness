# Framework comparison (Phase 0 results + Phase 1)

`subset.json` is the fixed 12-item dataset: 2 per category
(adversarial, api, billing, limits, security, troubleshooting),
severity-spread, with full item bodies. Do not change the IDs without
re-running all three frameworks.

## Phase 0 gate results (2026-10-31, iMac, Ollama 0.40.10, qwen3:8b CPU)

| Framework | Verdict | Notes |
|---|---|---|
| promptfoo | **PASS** | `npx promptfoo@latest eval` — 2/2 asserts green, 9m32s, zero config friction. Cleanest pass. |
| deepeval | PENDING re-run | First run hit the ~90s per-attempt timeout on CPU judges; fixed via `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=600` baked into the smoke script. Re-run not yet pasted. |
| ragas 0.4.3 | **FAIL (recommended)** | Four layers: (1) LangChain 1.x removed `langchain_community.chat_models.vertexai` — needs `pip install "langchain-community<0.4"`; (2) default 60s per-metric timeout kills CPU judges — needs `RunConfig(timeout=600)`; (3) metric API churn: deprecated `ragas.metrics` singletons → `ragas.metrics.collections` classes, instantiated with `llm=`/`embeddings=`; (4) architectural: collections metrics validate `isinstance` against `InstructorLLM` and reject `ChatOllama` outright — local Ollama is not a supported judge path, only OpenAI-style endpoints via `llm_factory`. Layers 1–3 are fixable friction; layer 4 is by design. |

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
