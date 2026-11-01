#!/usr/bin/env python3
"""Phase 1, step 2a: build a promptfoo config that asserts on saved responses.

Mirrors the heuristic judge exactly: `icontains` per expected keyword
(case-insensitive substring, like the judge), `not-icontains` per
must_not_contain phrase.

Usage (from repo root):
    python3 frameworks/comparison/phase1/score_promptfoo.py --responses responses_ollama.json

Then:
    cd frameworks/comparison/phase1
    COMPARISON_RESPONSES=responses_ollama.json npx promptfoo@latest eval \\
        -c promptfoo_ollama.json -o pf_results_ollama.json

The -o file feeds build_report.py.
"""
import argparse
import json
import sys
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
ROOT = PHASE1_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))


def tag_for(responses_name: str) -> str:
    stem = Path(responses_name).stem  # responses_ollama
    return stem[10:] if stem.startswith("responses_") else stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True,
                    help="responses file in phase1/ (e.g. responses_ollama.json)")
    args = ap.parse_args()

    records = json.loads((PHASE1_DIR / args.responses).read_text())
    tag = tag_for(args.responses)

    tests = []
    for r in records:
        asserts = (
            [{"type": "icontains", "value": kw} for kw in r.get("expected_keywords", [])]
            + [{"type": "not-icontains", "value": ph}
               for ph in r.get("must_not_contain", [])]
        )
        tests.append({
            "vars": {"id": r["id"], "question": r["question"]},
            "assert": list(asserts),
        })

    config = {
        "description": f"Phase 1 keyword asserts over saved responses ({args.responses})",
        "prompts": ["{{question}}"],
        "providers": [{"id": "file://./provider.py", "label": "saved-responses"}],
        "tests": tests,
    }
    out = PHASE1_DIR / f"promptfoo_{tag}.json"
    out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {out} ({len(tests)} tests)")
    print("next:")
    print("  cd frameworks/comparison/phase1")
    print(f"  COMPARISON_RESPONSES={args.responses} npx promptfoo@latest eval "
          f"-c promptfoo_{tag}.json -o pf_results_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
