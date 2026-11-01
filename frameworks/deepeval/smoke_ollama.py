#!/usr/bin/env python3
"""Phase 0b smoke: DeepEval with a local Ollama judge (no API keys).

Swaps DeepEval's default GPT judge for OllamaModel so everything runs
local. Run from the repo root:

    python3 frameworks/deepeval/smoke_ollama.py

Green = prints a score and a reason, no API key configured.
"""
import sys

from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


def main() -> int:
    model = OllamaModel(model="qwen3:8b")
    metric = AnswerRelevancyMetric(model=model)
    case = LLMTestCase(
        input="How do I update the credit card on my account?",
        actual_output=(
            "Go to Settings > Billing > Payment Method, click Update, "
            "enter the new card details, and save."
        ),
    )
    metric.measure(case)
    print("score:", metric.score)
    print("reason:", metric.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
