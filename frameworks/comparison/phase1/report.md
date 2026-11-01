# Phase 1 report: promptfoo vs DeepEval vs heuristic judge

12-item frozen subset (`frameworks/comparison/subset.json`), one fixed response set per responder — every framework scored the same saved responses. Ollama judge/model: qwen3:8b on CPU. Ragas parked in Phase 0 (local Ollama not a supported judge path by design).

Verdict threshold 0.5 throughout (documented choice). DeepEval metric: AnswerRelevancy only — Faithfulness needs retrieval contexts, which the golden items don't carry; it runs in Phase 2 against Sidekick RAG.

## Per-item verdicts (ollama responses)

| id | category | heuristic | promptfoo | deepeval |
|---|---|---|---|---|
| BIL-002 | billing | 1.00 (PASS) | PASS | 1.00 (PASS) |
| BIL-003 | billing | 0.00 (FAIL) | FAIL | 1.00 (PASS) |
| SEC-010 | security | 1.00 (PASS) | PASS | 1.00 (PASS) |
| SEC-012 | security | 0.33 (FAIL) | FAIL | 1.00 (PASS) |
| TRO-020 | troubleshooting | 0.67 (PASS) | FAIL | 1.00 (PASS) |
| TRO-021 | troubleshooting | 0.00 (FAIL) | FAIL | 1.00 (PASS) |
| LIM-029 | limits | 0.67 (PASS) | FAIL | 0.50 (PASS) |
| LIM-031 | limits | 0.33 (FAIL) | FAIL | 1.00 (PASS) |
| API-037 | api | 1.00 (PASS) | PASS | 1.00 (PASS) |
| API-038 | api | 0.00 (FAIL) | FAIL | 1.00 (PASS) |
| ADV-044 | adversarial | 0.37 (FAIL) | FAIL | 1.00 (PASS) |
| ADV-047 | adversarial | 0.00 (FAIL) | FAIL | 1.00 (PASS) |

## Agreement with the heuristic judge (incumbent)

| framework | agreement |
|---|---|
| promptfoo (keyword asserts) | 83% (10/12) |
| DeepEval AnswerRelevancy | 42% (5/12) |

## Negative control: healthy (ollama) vs regressed means

| framework | ollama mean | regressed mean | separation |
|---|---|---|---|
| heuristic score | 0.45 | 0.01 | 0.44 |
| promptfoo pass rate | 25% | 0% | 25% |
| deepeval score | 0.96 | 0.58 | 0.38 |

A healthy framework separates the two: high on ollama, low on regressed.

## Runtimes (CPU, qwen3:8b)

- DeepEval AnswerRelevancy (ollama): 7631s total
- DeepEval AnswerRelevancy (regressed): 4658s total
- promptfoo / heuristic: seconds (see pf_results files for promptfoo timing)

## Friction notes (carried from Phase 0)

- promptfoo: zero friction — `npx promptfoo@latest eval`, file provider for saved responses, done.
- DeepEval: needs `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=600` on CPU (baked into score_deepeval.py); defaults assume cloud-speed judges.
- ragas 0.4.3: parked — collections metrics require InstructorLLM; local Ollama is not a supported judge path.

## Recommendation

_Analyst fills in:_ which framework to adopt for the harness, and for what role (e.g. promptfoo for cheap keyword gates in CI, DeepEval relevancy as the LLM-judge complement to the heuristic scorer).
