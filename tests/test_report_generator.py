import json
import os

from approval.store import ApprovalStore
from memory.db import ExperimentDB
from reporting.report_generator import compute_kpis, generate_report


def test_kpis_computed_from_actual_stored_data(tmp_dir):
    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    store = ApprovalStore(os.path.join(tmp_dir, "approvals.db"))

    db.store_experiment(hypothesis="a", diff="d1", rationale="r", metrics={"execution_time": 2.0}, outcome="success")
    db.store_experiment(hypothesis="b", diff="d2", rationale="r", metrics={"execution_time": 1.0}, outcome="failure")
    req_id = store.create_request("cand-1", "goal", "diff", 0.9, {})
    store.decide(req_id, "approved")

    kpis = compute_kpis(db, store, evolution_report_path=os.path.join(tmp_dir, "no_such_file.jsonl"))

    assert kpis["total_experiments"] == 2
    assert kpis["success_count"] == 1
    assert kpis["failure_count"] == 1
    assert kpis["approvals"]["approved"] == 1
    # no evolution_report.jsonl -> falls back to summing experiment execution_time
    assert kpis["total_compute_seconds"] == 3.0
    assert kpis["compute_cost_per_improvement_seconds"] == 3.0  # 3.0s / 1 success


def test_generate_report_file_matches_compute_kpis(tmp_dir):
    """The written report file's numbers must be exactly what compute_kpis produces - not a separate recomputation."""
    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    store = ApprovalStore(os.path.join(tmp_dir, "approvals.db"))
    db.store_experiment(hypothesis="a", diff="d1", rationale="r", metrics={}, outcome="success")

    output_dir = os.path.join(tmp_dir, "reports")
    report_path = generate_report(db, store, output_dir=output_dir)

    assert os.path.exists(report_path)
    with open(os.path.join(output_dir, "latest_report.json")) as f:
        written_kpis = json.load(f)

    fresh_kpis = compute_kpis(db, store)
    # generated_at will legitimately differ by a few ms; compare everything else
    written_kpis.pop("generated_at")
    fresh_kpis.pop("generated_at")
    assert written_kpis == fresh_kpis


def test_duplicate_avoidance_rate_reads_from_evolution_report(tmp_dir):
    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    report_path = os.path.join(tmp_dir, "evolution_report.jsonl")
    with open(report_path, "w") as f:
        f.write(json.dumps({"population_size": 5, "duplicate_avoidance_count": 2, "compute_time_spent": 10.0}) + "\n")
        f.write(json.dumps({"population_size": 5, "duplicate_avoidance_count": 3, "compute_time_spent": 8.0}) + "\n")

    kpis = compute_kpis(db, None, evolution_report_path=report_path)

    assert kpis["duplicates_avoided"] == 5
    assert kpis["candidates_scheduled"] == 10
    assert kpis["duplicate_avoidance_rate"] == 5 / 15
    assert kpis["total_compute_seconds"] == 18.0
