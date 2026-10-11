# llm-eval-harness

Evaluate an LLM system against a golden set of 50 Q/A items, score responses
with a deterministic heuristic judge (or an optional LLM judge), and detect
regressions with PSI + mean-delta drift detection.

No API keys needed to try it — the heuristic judge and demo responders run
fully offline.

## Quickstart

```bash
pip install -r requirements.txt

# Run the healthy demo responder (expect exit 0, no drift)
PYTHONPATH=src python -m eval_harness.runner \
  --responder healthy --judge heuristic \
  --baseline baselines/baseline_results.json

# Run the regressed demo responder (expect exit 1, drift detected)
PYTHONPATH=src python -m eval_harness.runner \
  --responder regressed --judge heuristic \
  --baseline baselines/baseline_results.json

# Run the test suite
PYTHONPATH=src pytest tests/ -q

# Launch the dashboard
streamlit run dashboard.py
```

## Full evaluation flow: from runs to model selection

**Step 1 — Evaluate each candidate.** Run the harness once per model,
saving a report each time:

```bash
# Candidate A: qwen3:8b
OLLAMA_MODEL=qwen3:8b PYTHONPATH=src python -m eval_harness.runner \
  --responder ollama --judge heuristic \
  --baseline baselines/baseline_results.json \
  --output reports/qwen3-8b.json

# Candidate B: qwen2.5-coder:32b
OLLAMA_MODEL=qwen2.5-coder:32b PYTHONPATH=src python -m eval_harness.runner \
  --responder ollama --judge heuristic \
  --baseline baselines/baseline_results.json \
  --output reports/qwen25-coder-32b.json
```

Each report's `config` block records the exact model, timestamp, and item
count — so you always know what was evaluated against what.

**Step 2 — Compare.** Open the dashboard, switch to **Compare models**,
upload both JSONs. You get side-by-side:

- Mean score, drift vs baseline, per-model columns
- Mean by severity (critical / medium / low)
- Mean by category (billing, security, troubleshooting, …)
- **Selection guidance** — who leads per severity, with a ⚠️ flag when
  critical-severity scores diverge significantly

**Step 3 — Interpret and decide.** Reading the comparison:

1. **Check critical severity first.** If one model scores 0.85 on critical
   items and another scores 0.60, the choice is made — critical covers
   security and adversarial refusal, the things that cause incidents.
2. **Then look at the mean.** Among models that pass the critical bar,
   higher mean = better overall coverage.
3. **Check drift.** A model with good scores but PSI > 0.25 vs your
   production baseline is behaving differently from what's deployed —
   investigate before switching.
4. **Drill into categories.** If a model wins overall but tanks on `api`
   questions and your use case is API-heavy, it's the wrong pick.
5. **Read the worst 5 items.** The per-item table (in View saved report
   mode) shows exactly where each model fails — patterns here beat
   aggregates for debugging.

**The rule:** one eval run tells you almost nothing. The value is the
*trend* — same golden set, same judge, every model change. That's what the
provenance block and the comparison view are for.

## Evaluating real models with Ollama

The `ollama` responder queries a local Ollama instance, so you can evaluate
Qwen (or any Ollama model) against the golden set:

```bash
# Make sure Ollama is running with a model pulled
ollama serve &
ollama pull qwen3:8b

# Evaluate qwen3:8b with the heuristic judge
OLLAMA_MODEL=qwen3:8b PYTHONPATH=src python -m eval_harness.runner \
  --responder ollama --judge heuristic \
  --baseline baselines/baseline_results.json

# Use a bigger model
OLLAMA_MODEL=qwen2.5-coder:32b PYTHONPATH=src python -m eval_harness.runner \
  --responder ollama --judge heuristic

# Quick smoke test with just 5 questions (useful for slow local models)
OLLAMA_MODEL=qwen3:8b PYTHONPATH=src python -m eval_harness.runner \
  --responder ollama --judge heuristic --limit 5

# Use Qwen as the LLM judge too (via Ollama's OpenAI-compatible API)
JUDGE_API_BASE=http://localhost:11434/v1 JUDGE_MODEL=qwen3:8b \
  PYTHONPATH=src python -m eval_harness.runner \
  --responder ollama --judge llm
```

Env vars:

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3:8b` | Model for the ollama responder |
| `OLLAMA_TIMEOUT` | `120` | Seconds per question |
| `JUDGE_API_BASE` | `https://api.openai.com/v1` | LLM judge endpoint (Ollama: `http://localhost:11434/v1`) |
| `JUDGE_MODEL` | `gpt-4o-mini` | LLM judge model |
| `JUDGE_API_KEY` | (empty) | Not needed for local Ollama |

## What it does

1. **Golden set** — 50 hand-written Q/A items across billing, security,
   troubleshooting, limits, API, and adversarial (refusal) cases. Each item
   has expected keywords, forbidden phrases, severity, and a reference answer.
2. **Judging** — the heuristic judge scores keyword coverage (0–1), subtracts
   0.3 per forbidden phrase, and penalizes very short/long answers. Fully
   deterministic: same input → same score, every run.
3. **Metrics** — grounded precision/recall against expected keywords,
   severity slices (critical/medium/low), and *augmented* keywords: novel
   content words the response used that aren't in the golden set. Augmented
   keywords are diagnostic only — a system is never penalized for them.
4. **Drift detection** — compares the current score distribution to the
   checked-in healthy baseline with PSI (bins scores into 10 bins;
   PSI > 0.25 = significant drift) and mean-delta (|Δmean| > 0.1 = drift).
   Exit code 1 when drift is detected, so CI can gate on it.

## Project structure

```
├── src/eval_harness/
│   ├── golden_sets.py   # load + validate the 50-item dataset
│   ├── judges.py        # HeuristicJudge (deterministic) + LLMJudge (optional API)
│   ├── drift.py         # PSI + mean-delta
│   ├── responders.py    # healthy + regressed demo responders
│   ├── metrics.py       # grounded/augmented metrics, severity slices
│   └── runner.py        # CLI entry point
├── data/golden_qa.json
├── baselines/baseline_results.json   # healthy baseline (mean 0.814)
├── tests/               # 42 tests
├── dashboard.py         # Streamlit UI
└── METHODOLOGY.md       # judge rubric + drift math
```

## Optional: LLM judge

Set these env vars to use an OpenAI-compatible judge instead of the heuristic:

```bash
export JUDGE_API_BASE="https://api.openai.com/v1"  # or any compatible endpoint
export JUDGE_API_KEY="sk-..."
export JUDGE_MODEL="gpt-4o-mini"
PYTHONPATH=src python -m eval_harness.runner --judge llm
```

Without `JUDGE_API_KEY`, the LLM judge skips gracefully and reports why.

## CI

`.github/workflows/ci.yml` runs the test suite on push. (Added via the
GitHub web UI — the API cannot create workflow files.)

## Honest limits

- The heuristic judge measures keyword overlap, not true answer quality.
  It is a regression tripwire, not a replacement for human review.
- The demo responders are synthetic. Plug in a real system by writing a
  function `my_responder(item: dict) -> str` and passing it to `run_eval`.
- The 50-item golden set is a starter set, not a benchmark. Extend it for
  your domain before trusting it for release decisions.
