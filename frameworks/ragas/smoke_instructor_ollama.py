#!/usr/bin/env python3
"""Phase 0c-alt smoke: Ragas 0.4.x via llm_factory + Ollama's OpenAI-compatible API.

Ragas 0.4.x collections metrics only accept InstructorLLM judges — a
LangChain ChatOllama is rejected by design (ValueError). The documented
local path is llm_factory() with an OpenAI-compatible client pointed at
Ollama's /v1 endpoint (docs.ragas.io quickstart, "Local Models (Ollama)").

No API keys: api_key="ollama" is a dummy; every call stays on localhost.
No new pip installs needed beyond the Phase 0c set (ragas pulls in
instructor + openai itself).

Run from the repo root:

    ollama pull nomic-embed-text   # served via /v1/embeddings; CPU-cheap
    python3 frameworks/ragas/smoke_instructor_ollama.py

Green = prints faithfulness and answer_relevancy scores.
One shot: if this fails, ragas is parked and Phase 2 re-homes to DeepEval.
"""
import sys

from datasets import Dataset
from openai import AsyncOpenAI, OpenAI

from ragas import evaluate
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness
from ragas.run_config import RunConfig


def main() -> int:
    # Judge LLM: qwen3:8b through Ollama's OpenAI-compatible endpoint.
    # Model name is passed through verbatim (docs use "mistral" here).
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    llm = llm_factory("qwen3:8b", provider="openai", client=client)

    # Embeddings: nomic-embed-text through Ollama's /v1/embeddings endpoint.
    embeddings = embedding_factory(
        provider="openai",
        model="nomic-embed-text",
        client=AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1"),
    )

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
        metrics=[Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)],
        llm=llm,
        embeddings=embeddings,
        # CPU Ollama judges are slow; the 60s default is tuned for cloud APIs.
        run_config=RunConfig(timeout=600),
    )
    df = result.to_pandas()
    print(df[["faithfulness", "answer_relevancy"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
