# Ragas smoke test (Phase 0c)

Proves Ragas runs with local Ollama for both the judge LLM and the
embeddings — no API keys.

## Prereqs

- Python 3.9+
- Ollama running with `qwen3:8b` pulled
- `ollama pull nomic-embed-text` (embeddings for the metrics; CPU-cheap)

## Run

From the repo root:

```bash
pip install ragas datasets langchain-ollama
pip install "langchain-community<0.4"   # ragas 0.4.3 imports ChatVertexAI
                                        # from a path 0.4.x removed
python3 frameworks/ragas/smoke_ollama.py
```

## Notes

- `RunConfig(timeout=600)` is set in the script: CPU Ollama judges exceed
  ragas's default 60s per-metric timeout. Expect ~2-5 min for both metrics.
- Ignore the `multiprocess`/`ResourceTracker` traceback at exit — cosmetic,
  Python 3.12 fork cleanup noise.

## Green looks like

Prints faithfulness and answer_relevancy scores. If the LangChain/Ollama
wiring fights back, paste the error — that friction is a finding.
