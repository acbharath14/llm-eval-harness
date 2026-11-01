#!/usr/bin/env python3
"""Phase 0c smoke: Ragas with local Ollama LLM + embeddings (no API keys).

Ragas is LangChain-glued, so the Ollama wiring goes through
langchain-ollama. Run from the repo root:

    pip install ragas datasets langchain-ollama
    ollama pull nomic-embed-text   # embeddings for the metrics; CPU-cheap
    python3 frameworks/ragas/smoke_ollama.py

Green = prints faithfulness and answer_relevancy scores, no API key.
If the Ollama wiring fights back, that friction is itself a finding —
paste the error back.
"""
import sys

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics.collections import answer_relevancy, faithfulness
from ragas.run_config import RunConfig


def main() -> int:
    data = {
        "question": ["How do I update the credit card on my account?"],
        "contexts": [
            [
                "Go to Settings > Billing > Payment Method, click Update, "
                "enter the new card details, and save."
            ]
        ],
        "answer": [
            "Go to Settings > Billing, click Update under Payment Method, "
            "enter the new card, and save."
        ],
        "ground_truth": [
            "Go to Settings > Billing > Payment Method, click Update, "
            "enter the new card details, and save."
        ],
    }
    result = evaluate(
        Dataset.from_dict(data),
        metrics=[faithfulness, answer_relevancy],
        llm=ChatOllama(model="qwen3:8b"),
        embeddings=OllamaEmbeddings(model="nomic-embed-text"),
        # CPU Ollama judges are slow: the default 60s per-metric timeout
        # kills both jobs. 600s each is plenty; expect ~2-5 min total.
        run_config=RunConfig(timeout=600),
    )
    df = result.to_pandas()
    print(df[["faithfulness", "answer_relevancy"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
