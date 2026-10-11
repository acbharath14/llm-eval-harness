"""Tests for the runner CLI: exit codes, drift wiring, error handling."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval_harness.runner import main, run_eval
from eval_harness.golden_sets import load_golden_set
from eval_harness.judges import HeuristicJudge
from eval_harness.responders import healthy_responder, regressed_responder

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "baselines" / "baseline_results.json"


def test_run_eval_returns_per_item_scores():
    items = load_golden_set()[:3]
    scored = run_eval(items, healthy_responder, HeuristicJudge())
    assert len(scored) == 3
    assert all("score" in s and "id" in s for s in scored)


def test_healthy_no_drift_exit_0(capsys):
    code = main(["--responder", "healthy", "--judge", "heuristic",
                 "--baseline", str(BASELINE)])
    assert code == 0


def test_regressed_drift_exit_1(capsys):
    code = main(["--responder", "regressed", "--judge", "heuristic",
                 "--baseline", str(BASELINE)])
    assert code == 1


def test_missing_baseline_warns_not_crashes(capsys):
    code = main(["--responder", "healthy", "--baseline", "/nope/missing.json"])
    assert code == 0  # baseline skipped, eval still succeeds


def test_output_flag_writes_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    code = main(["--responder", "healthy", "--output", str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["aggregate"]["n"] == 50
    assert len(report["items"]) == 50


def test_responder_exception_does_not_kill_run():
    def boom(item):
        raise RuntimeError("kaboom")
    scored = run_eval(load_golden_set()[:2], boom, HeuristicJudge())
    assert len(scored) == 2
    assert scored[0]["responder_error"] == "kaboom"


def test_limit_flag_restricts_items(tmp_path, capsys):
    out = tmp_path / "report.json"
    code = main(["--responder", "healthy", "--limit", "5",
                 "--output", str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["aggregate"]["n"] == 5
    assert len(report["items"]) == 5


def test_limit_flag_rejects_zero(capsys):
    code = main(["--responder", "healthy", "--limit", "0"])
    assert code == 1
    assert "must be >= 1" in capsys.readouterr().err
