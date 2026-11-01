# Promptfoo smoke test (Phase 0a)

Proves promptfoo runs against local Ollama with no API keys.

## Prereqs

- Node 20+
- Ollama running with `qwen3:8b` pulled

## Run

From this directory:

```bash
npx promptfoo@latest eval
```

## Green looks like

Exit 0, both tests evaluated, results under `promptfoo-results/`
(gitignored). If Ollama isn't reachable you get a provider error, not
a silent pass — that counts as failing the gate.
