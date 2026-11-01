#!/usr/bin/env python3
"""Phase 1, step 3: agreement report.

Reads the scored outputs for BOTH response sets (ollama = system under
test, regressed = negative control) and writes report.md:

  - per-item table: heuristic vs promptfoo vs DeepEval verdicts
  - agreement-with-heuristic rates (the incumbent judge is ground truth here)
  - negative-control separation: does each framework score healthy
    responses higher than regressed ones?
  - runtimes, friction notes, recommendation

Verdict threshold is 0.5 throughout — a documented choice, not a tuned one:
heuristic and DeepEval scores are 0-1, promptfoo is binary pass/fail.

Usage (from repo root), after steps 1-2 are done for both sets:
    python3 frameworks/comparison/phase1/build_report.py

Expected inputs in frameworks/comparison/phase1/:
    responses_ollama.json, responses_regressed.json,
    pf_results_ollama.json, pf_results_regressed.json,
    deepeval_scores_ollama.json, deepeval_scores_regressed.json
"""
import json
import sys
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
ROOT = PHASE1_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval_harness.judges import HeuristicJudge

THRESHOLD = 0.5
SETS = ("ollama", "regressed")


def load_json(name: str):
    p = PHASE1_DIR / name
    if not p.exists():
        raise SystemExit(f"missing {p} — run steps 1-2 for both sets first")
    return json.loads(p.read_text())


def promptfoo_verdicts(path: Path) -> dict[str, bool]:
    """Defensive parse of `promptfoo eval -o` JSON across shapes."""
    data = json.loads(path.read_text())
    node = data.get("results", data)
    rows = node.get("results", node) if isinstance(node, dict) else node
    if not isinstance(rows, list):
        raise SystemExit(f"unrecognized promptfoo shape in {path}: "
                         f"top keys={sorted(data.keys())}")
    out = {}
    for r in rows:
        vars_ = r.get("vars") or (r.get("testCase") or {}).get("vars") or {}
        rid = vars_.get("id")
        if rid is None:
            raise SystemExit(f"unrecognized promptfoo row in {path}: "
                             f"keys={sorted(r.keys())}")
        out[rid] = bool(r.get("success", r.get("pass", False)))
    return out


def main() -> int:
    judge = HeuristicJudge()
    per_set: dict[str, dict] = {}

    for tag in SETS:
        records = load_json(f"responses_{tag}.json")
        de = {e["id"]: e for e in load_json(f"deepeval_scores_{tag}.json")["items"]}
        pf = promptfoo_verdicts(PHASE1_DIR / f"pf_results_{tag}.json")

        rows = []
        for r in records:
            h = judge.score(r, r["response"])
            h_pass = h["score"] >= THRESHOLD
            d = de.get(r["id"], {})
            d_score = d.get("score")
            d_pass = (d_score is not None) and d_score >= THRESHOLD
            p_pass = pf.get(r["id"])
            rows.append({
                "id": r["id"], "category": r["category"],
                "heuristic": h["score"], "h_pass": h_pass,
                "promptfoo": p_pass, "deepeval": d_score, "d_pass": d_pass,
            })
        de_time = load_json(f"deepeval_scores_{tag}.json").get("total_elapsed_s")
        per_set[tag] = {"rows": rows, "deepeval_s": de_time}

    # ---- agreement (ollama set): framework verdict vs heuristic verdict ----
    o = per_set["ollama"]["rows"]
    n = len(o)
    agree_pf = sum(1 for r in o if r["promptfoo"] == r["h_pass"]) / n
    agree_de = sum(1 for r in o if r["d_pass"] == r["h_pass"]) / n

    # ---- negative control: healthy mean vs regressed mean ----
    def means(tag, key):
        vals = [r[key] for r in per_set[tag]["rows"] if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    lines = []
    A = lines.append
    A("# Phase 1 report: promptfoo vs DeepEval vs heuristic judge")
    A("")
    A("12-item frozen subset (`frameworks/comparison/subset.json`), one fixed "
      "response set per responder — every framework scored the same saved "
      "responses. Ollama judge/model: qwen3:8b on CPU. Ragas parked in Phase 0 "
      "(local Ollama not a supported judge path by design).")
    A("")
    A(f"Verdict threshold {THRESHOLD} throughout (documented choice). "
      "DeepEval metric: AnswerRelevancy only — Faithfulness needs retrieval "
      "contexts, which the golden items don't carry; it runs in Phase 2 "
      "against Sidekick RAG.")
    A("")
    A("## Per-item verdicts (ollama responses)")
    A("")
    A("| id | category | heuristic | promptfoo | deepeval |")
    A("|---|---|---|---|---|")
    for r in o:
        pf_mark = "PASS" if r["promptfoo"] else ("FAIL" if r["promptfoo"] is False else "?")
        A(f"| {r['id']} | {r['category']} | {r['heuristic']:.2f} "
          f"({'PASS' if r['h_pass'] else 'FAIL'}) | {pf_mark} | "
          f"{r['deepeval']:.2f} ({'PASS' if r['d_pass'] else 'FAIL'}) |")
    A("")
    A("## Agreement with the heuristic judge (incumbent)")
    A("")
    A("| framework | agreement |")
    A("|---|---|")
    A(f"| promptfoo (keyword asserts) | {agree_pf:.0%} ({sum(1 for r in o if r['promptfoo'] == r['h_pass'])}/{n}) |")
    A(f"| DeepEval AnswerRelevancy | {agree_de:.0%} ({sum(1 for r in o if r['d_pass'] == r['h_pass'])}/{n}) |")
    A("")
    A("## Negative control: healthy (ollama) vs regressed means")
    A("")
    A("| framework | ollama mean | regressed mean | separation |")
    A("|---|---|---|---|")
    for label, key, is_bool in (("heuristic score", "heuristic", False),
                                ("promptfoo pass rate", "promptfoo", True),
                                ("deepeval score", "deepeval", False)):
        mh = means("ollama", key)
        mr = means("regressed", key)
        fmt = (lambda v: f"{v:.0%}") if is_bool else (lambda v: f"{v:.2f}")
        A(f"| {label} | {fmt(mh)} | {fmt(mr)} | {fmt(mh - mr)} |")
    A("")
    A("A healthy framework separates the two: high on ollama, low on regressed.")
    A("")
    A("## Runtimes (CPU, qwen3:8b)")
    A("")
    for tag in SETS:
        s = per_set[tag]["deepeval_s"]
        A(f"- DeepEval AnswerRelevancy ({tag}): "
          f"{s:.0f}s total" if s else f"- DeepEval ({tag}): n/a")
    A("- promptfoo / heuristic: seconds (see pf_results files for promptfoo timing)")
    A("")
    A("## Friction notes (carried from Phase 0)")
    A("")
    A("- promptfoo: zero friction — `npx promptfoo@latest eval`, file provider "
      "for saved responses, done.")
    A("- DeepEval: needs `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=600` "
      "on CPU (baked into score_deepeval.py); defaults assume cloud-speed judges.")
    A("- ragas 0.4.3: parked — collections metrics require InstructorLLM; "
      "local Ollama is not a supported judge path.")
    A("")
    A("## Recommendation")
    A("")
    A("_Analyst fills in:_ which framework to adopt for the harness, and for "
      "what role (e.g. promptfoo for cheap keyword gates in CI, DeepEval "
      "relevancy as the LLM-judge complement to the heuristic scorer).")

    out = PHASE1_DIR / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print(f"agreement — promptfoo: {agree_pf:.0%}, deepeval: {agree_de:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
