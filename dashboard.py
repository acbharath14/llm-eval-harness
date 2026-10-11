"""Streamlit dashboard for the LLM eval harness.

Run:  streamlit run dashboard.py

Three modes:
  Run live eval    — run the golden set through a responder right now
  View saved report — upload a --output JSON report and visualize it
  Compare models   — upload 2+ reports for side-by-side model selection
"""
from __future__ import annotations

import datetime
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


def _model_label(report: dict, fallback: str) -> str:
    """Human-readable label from provenance: model name or responder key."""
    config = report.get("config", {})
    return (config.get("responder_model")
            or config.get("judge_model")
            or report.get("responder", fallback))


def _render_report(report: dict, use_baseline: bool = True) -> None:
    """Render metrics, charts, and tables from a report dict."""
    agg = report["aggregate"]
    scored = report.get("items", [])

    config = report.get("config", {})
    if config:
        with st.expander("🔍 Evaluation provenance", expanded=True):
            cols = st.columns(4)
            cols[0].write(f"**Responder:** {config.get('responder', '?')}")
            if config.get("responder_model"):
                cols[1].write(f"**Model:** {config['responder_model']}")
            cols[2].write(f"**Judge:** {config.get('judge', '?')}")
            if config.get("judge_model"):
                cols[3].write(f"**Judge model:** {config['judge_model']}")
            st.caption(f"Ran {config.get('timestamp', '?')} · "
                       f"{config.get('n_items', '?')} items")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean score", round(agg["mean_score"], 4))
    c2.metric("Items scored", f"{agg['n_scored']}/{agg['n']}")
    c3.metric("Min / Max", f"{agg['min_score']} / {agg['max_score']}")
    drift_info = report.get("drift")
    if drift_info is None and use_baseline and scored:
        try:
            baseline = json.loads(
                (ROOT / "baselines" / "baseline_results.json").read_text())
            base_scores = baseline["aggregate"]["per_item_scores"]
            cur_scores = [e["score"] for e in scored if e.get("score") is not None]
            drift_info = drift_mod.detect_drift(base_scores, cur_scores)
        except (FileNotFoundError, KeyError, ValueError):
            drift_info = None
    if drift_info:
        c4.metric("Drift", "YES 🔴" if drift_info["drift_detected"] else "no 🟢",
                  f"PSI {drift_info['psi']:.3f}")
    else:
        c4.metric("Drift", "n/a")

    st.subheader("Score distribution")
    scores = [e["score"] for e in scored if e.get("score") is not None]
    if scores:
        st.bar_chart({
            f"{i/10:.1f}-{(i+1)/10:.1f}":
            sum(1 for s in scores
                if i/10 <= s < (i+1)/10 or (i == 9 and s == 1.0))
            for i in range(10)})

    st.subheader("By severity")
    st.table({sev: {"n": v["n"], "mean": round(v["mean"], 4),
                    "min": v["min"], "max": v["max"]}
              for sev, v in agg["by_severity"].items()})

    if drift_info:
        st.subheader("Drift vs baseline")
        st.json(drift_info)

    st.subheader("Per-item results")
    rows = [{
        "id": e.get("item_id", e.get("id")),
        "category": e.get("category"),
        "severity": e.get("severity"),
        "score": e.get("score"),
        "keyword_coverage": e.get("keyword_coverage"),
        "forbidden_hits": ", ".join(e.get("forbidden_hits", [])),
        "grounded_recall": e.get("grounded_recall"),
        "novel_keywords": ", ".join(e.get("augmented_keywords", [])[:5]),
        "response": e.get("response_preview", ""),
    } for e in scored]
    st.dataframe(rows, use_container_width=True)


def _render_comparison(reports: list[dict], names: list[str]) -> None:
    """Side-by-side columns: one per model run."""
    labels = [_model_label(r, n) for r, n in zip(reports, names)]
    cols = st.columns(len(reports))

    for col, report, label in zip(cols, reports, labels):
        agg = report["aggregate"]
        drift = report.get("drift") or {}
        with col:
            st.subheader(label)
            st.metric("Mean", round(agg["mean_score"], 4))
            st.metric("Scored", f"{agg['n_scored']}/{agg['n']}")
            if drift:
                st.metric("Drift",
                          "🔴 YES" if drift.get("drift_detected") else "🟢 no",
                          f"PSI {drift.get('psi', 0):.3f}")

    st.subheader("Mean score by severity")
    sev_rows: dict = {}
    for label, report in zip(labels, reports):
        for sev, v in report["aggregate"]["by_severity"].items():
            sev_rows.setdefault(sev, {})[label] = round(v["mean"], 4)
    st.table(sev_rows)

    st.subheader("Mean score by category")
    cat_rows: dict = {}
    for label, report in zip(labels, reports):
        by_cat: dict = {}
        for e in report.get("items", []):
            cat = e.get("category", "?")
            by_cat.setdefault(cat, []).append(e.get("score") or 0)
        for cat, sc in by_cat.items():
            cat_rows.setdefault(cat, {})[label] = round(sum(sc) / len(sc), 4)
    st.table(cat_rows)

    st.subheader("Selection guidance")
    for sev, per_model in sev_rows.items():
        best = max(per_model, key=per_model.get)
        worst = min(per_model, key=per_model.get)
        gap = per_model[best] - per_model[worst]
        flag = " ⚠️ CHECK CRITICAL" if sev == "critical" and gap > 0.15 else ""
        st.write(f"**{sev.title()}:** {best} leads "
                 f"({per_model[best]:.3f} vs {per_model[worst]:.3f}){flag}")


# --- Sidebar & mode dispatch ---

with st.sidebar:
    st.header("Run configuration")
    mode = st.radio("Mode", ["Run live eval", "View saved report",
                             "Compare models"])
    responder_name = st.selectbox("Responder", sorted(RESPONDERS))
    judge_name = st.selectbox("Judge", ["heuristic", "llm"])
    use_baseline = st.checkbox("Compare against baseline", value=True)
    run_btn = st.button("Run evaluation", type="primary")

if mode == "Compare models":
    st.subheader("Side-by-side model comparison")
    st.caption("Upload 2+ `--output` report JSONs. Each column is one model run — "
               "provenance is read from the report's config block.")
    uploads = st.file_uploader("Choose report files", type=["json"],
                               accept_multiple_files=True)
    if uploads and len(uploads) >= 2:
        reports = []
        for f in uploads:
            try:
                reports.append((f.name, json.load(f)))
            except (json.JSONDecodeError, KeyError):
                st.warning(f"Skipped {f.name}: not a valid report")
        if len(reports) >= 2:
            _render_comparison([r for _, r in reports],
                               [n for n, _ in reports])
    elif uploads:
        st.info("Upload at least 2 reports to compare.")
    else:
        st.info("Upload report JSONs to compare models side by side.")
elif mode == "View saved report":
    st.subheader("Upload a report JSON")
    st.caption("Reports are created with `--output report.json`. The config block "
               "shows exactly which model was evaluated.")
    uploaded = st.file_uploader("Choose a report file", type=["json"])
    if uploaded:
        try:
            report = json.load(uploaded)
            _render_report(report, use_baseline)
        except (json.JSONDecodeError, KeyError) as e:
            st.error(f"Could not parse report: {e}")
    else:
        st.info("Upload a `--output` JSON report to visualize it.")
elif run_btn:
    with st.spinner("Running eval over 50 golden items..."):
        items = load_golden_set()
        judge = LLMJudge() if judge_name == "llm" else HeuristicJudge()
        scored = run_eval(items, RESPONDERS[responder_name], judge)

    if judge_name == "llm" and not LLMJudge().available:
        st.warning("JUDGE_API_KEY not set — LLM judge scores were skipped.")

    config = {
        "responder": responder_name, "judge": judge_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_items": len(items),
    }
    report = {"config": config, "responder": responder_name,
              "judge": judge_name, "aggregate": aggregate(scored),
              "items": scored}
    _render_report(report, use_baseline)
else:
    st.info("Choose a responder and judge on the left, then click **Run evaluation**.")
    st.markdown("""
**What this does:** runs 50 golden Q/A items through a demo responder,
scores each response with the heuristic judge (keyword coverage − forbidden
phrases − length penalties), and compares the score distribution against the
checked-in healthy baseline using PSI + mean-delta drift detection.
""")
