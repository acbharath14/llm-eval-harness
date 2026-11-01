# DeepEval smoke test (Phase 0b)

Proves DeepEval runs with a local Ollama judge instead of its default
GPT judge — no API keys.

## Prereqs

- Python 3.9+
- Ollama running with `qwen3:8b` pulled

## Run

From the repo root:

```bash
pip install deepeval ollama
python3 frameworks/deepeval/smoke_ollama.py
```

## Green looks like

Prints `score:` and `reason:`. Metric *quality* with an 8B judge vs
GPT-4 is Phase 1's question, not this gate's — this gate only proves
it runs local.
