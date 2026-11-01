# Framework comparison (Phase 0 results + Phase 1)

`subset.json` is the fixed 12-item dataset: 2 per category
(adversarial, api, billing, limits, security, troubleshooting),
severity-spread, with full item bodies. Do not change the IDs without
re-running all three frameworks.

## Phase 0 gate results (2026-10-31, iMac, Ollama 0.40.10, qwen3:8b CPU)

| Framework | Verdict | Notes |
|---|---|---|
| promptfoo | **PASS** | `npx promptfoo@latest eval` — 2/2 asserts green, 9m32s, zero config friction. Cleanest pass. |
| deepeval | **PASS** | Needed `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=600` (baked into the smoke script) — CPU judges exceed the ~90s default. Score 1.0 on the smoke case with a sensible reason. |
| ragas 0.4.3 | **FAIL (parked)** | Four layers: (1) LangChain 1.x removed `langchain_community.chat_models.vertexai` — needs `pip install "langchain-community<0.4"`; (2) default 60s per-metric timeout kills CPU judges — needs `RunConfig(timeout=600)`; (3) metric API churn: deprecated `ragas.metrics` singletons → `ragas.metrics.collections` classes, instantiated with `llm=`/`embeddings=`; (4) collections metrics validate `isinstance` against `InstructorLLM` and reject `ChatOllama` outright. One-shot follow-up per the docs' own Ollama recipe (`llm_factory("qwen3:8b", provider="openai", client=OpenAI(base_url="http://localhost:11434/v1"))`): metrics constructed fine — the InstructorLLM wall clears — but the legacy `evaluate()` entry point still rejects the new-style metrics, consistent with 0.4.x's experiment-based architecture (their migration guide) where collections metrics go through the Experiment API, not `evaluate()`. Parked after the agreed one shot; Phase 2 re-homes to DeepEval's RAG metrics. |

## Method (Phase 1 — scripts in `phase1/`, runbook in `phase1/README.md`)

1. `generate_responses.py --responder ollama` → `responses_ollama.json`
   (qwen3:8b, the system under test); `--responder regressed` →
   `responses_regressed.json` (negative control). All frameworks score
   the *same* saved responses — never live generations.
2. Promptfoo: `score_promptfoo.py` builds keyword asserts (`icontains`
   per expected keyword, `not-icontains` per must_not_contain phrase —
   mirroring the heuristic judge) over a file provider serving the saved
   responses.
3. DeepEval: `score_deepeval.py` runs AnswerRelevancy via OllamaModel.
   Faithfulness is intentionally Phase-1-out: the golden items carry no
   retrieval contexts; it runs in Phase 2 against Sidekick RAG.
4. Ragas: parked in Phase 0 (local Ollama not a supported judge path).
5. `build_report.py` → `report.md`: per-item verdicts,
   agreement-with-heuristic table, healthy-vs-regressed separation,
   CPU runtimes, friction notes, recommendation. Verdict threshold 0.5
   throughout (documented choice).
