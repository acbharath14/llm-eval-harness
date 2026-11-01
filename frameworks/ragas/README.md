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
python3 frameworks/ragas/smoke_ollama.py
```

## Green looks like

Prints faithfulness and answer_relevancy scores. If the LangChain/Ollama
wiring fights back, paste the error — that friction is a finding.
