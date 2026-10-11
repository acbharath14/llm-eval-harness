"""Streamlit dashboard for the LLM eval harness.

Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval_harness.golden_sets import load_golden_set
from eval_harness.judges import HeuristicJudge, LLMJudge
from eval_harness.responders import RESPONDERS
from eval_harness.runner import run_eval
from eval_harness.metrics import aggregate
from eval_harness import drift as drift_mod

st.set_page_config(page_title="LLM Eval Harness", layout="wide")
st.title("LLM Eval Harness")
st.caption("Golden-set evaluation with heuristic judging and drift detection.")

ROOT = Path(__file__).parent

with st.sidebar:
    st.header("Run configuration")
    responder_name = st.selectbox("Responder", sorted(RESPONDERS))
    judge_name = st.selectbox("Judge", ["heuristic", "llm"])
    use_baseline = st.checkbox("Compare against baseline", value=True)
    run_btn = st.button("Run evaluation", type="primary")

if run_btn:
    with st.spinner("Running eval over 50 golden items..."):
        items = load_golden_set()
        judge = LLMJudge() if judge_name == "llm" else HeuristicJudge()
        scored = run_eval(items, RESPONDERS[responder_name], judge)
        agg = aggregate(scored)

    if judge_name == "llm" and not LLMJudge().available:
        st.warning("JUDGE_API_KEY not set — LLM judge scores were skipped.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean score", agg["mean_score"])
    c2.metric("Items scored", f"{agg['n_scored']}/{agg['n']}")
    c3.metric("Min / Max", f"{agg['min_score']} / {agg['max_score']}")
    drift_info = None
    if use_baseline:
        try:
            baseline = json.loads((ROOT / "baselines" / "baseline_results.json").read_text())
            base_scores = baseline["aggregate"]["per_item_scores"]
            cur_scores = [e["score"] for e in scored if e["score"] is not None]
            drift_info = drift_mod.detect_drift(base_scores, cur_scores)
            c4.metric("Drift", "YES 🔴" if drift_info["drift_detected"] else "no 🟢",
                      f"PSI {drift_info['psi']}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            st.warning(f"Baseline comparison skipped: {e}")
            c4.metric("Drift", "n/a")

    st.subheader("Score distribution")
    scores = [e["score"] for e in scored if e["score"] is not None]
    if scores:
        st.bar_chart({f"{i/10:.1f}-{(i+1)/10:.1f}": sum(1 for s in scores if i/10 <= s < (i+1)/10 or (i == 9 and s == 1.0))
                      for i in range(10)})

    st.subheader("By severity")
    st.table({sev: {"n": v["n"], "mean": v["mean"], "min": v["min"], "max": v["max"]}
              for sev, v in agg["by_severity"].items()})

    if drift_info:
        st.subheader("Drift vs baseline")
        st.json(drift_info)

    st.subheader("Per-item results")
    rows = [{
        "id": e["id"], "category": e["category"], "severity": e["severity"],
        "score": e["score"], "keyword_coverage": e.get("keyword_coverage"),
        "forbidden_hits": ", ".join(e.get("forbidden_hits", [])),
        "grounded_recall": e["grounded_recall"],
        "novel_keywords": ", ".join(e["augmented_keywords"][:5]),
        "response": e["response_preview"],
    } for e in scored]
    st.dataframe(rows, use_container_width=True)
else:
    st.info("Choose a responder and judge on the left, then click **Run evaluation**.")
    st.markdown("""
**What this does:** runs 50 golden Q/A items through a demo responder,
scores each response with the heuristic judge (keyword coverage − forbidden
phrases − length penalties), and compares the score distribution against the
checked-in healthy baseline using PSI + mean-delta drift detection.
""")
