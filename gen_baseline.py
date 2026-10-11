"""Generate baselines/baseline_results.json from the healthy responder."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval_harness.golden_sets import load_golden_set
from eval_harness.judges import HeuristicJudge
from eval_harness.responders import healthy_responder
from eval_harness.runner import run_eval
from eval_harness.metrics import aggregate

items = load_golden_set()
scored = run_eval(items, healthy_responder, HeuristicJudge())
agg = aggregate(scored)

baseline = {
    "responder": "healthy",
    "judge": "heuristic",
    "n_items": len(items),
    "aggregate": {
        **agg,
        "per_item_scores": [e["score"] for e in scored],
    },
    "per_item": [
        {"id": e["id"], "score": e["score"], "severity": e["severity"],
         "category": e["category"]}
        for e in scored
    ],
}

out = Path(__file__).parent / "baselines" / "baseline_results.json"
out.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
print(f"Wrote {out}")
print(f"Mean score: {agg['mean_score']}, n={agg['n_scored']}")
print("By severity:", {k: v["mean"] for k, v in agg["by_severity"].items()})
