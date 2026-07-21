import os
import sys

import pytest


@pytest.fixture
def client(tmp_dir):
    os.environ["CHROMA_DB_PATH"] = os.path.join(tmp_dir, "chroma")
    os.environ["APPROVAL_DB_PATH"] = os.path.join(tmp_dir, "approvals.db")
    os.environ["EVOLUTION_REPORT_PATH"] = os.path.join(tmp_dir, "evolution_report.jsonl")

    # api.main builds its db/store at import time from those env vars,
    # so force a fresh import per test rather than reusing a cached module.
    for mod in ("api.main",):
        if mod in sys.modules:
            del sys.modules[mod]
    import api.main as api_main
    from fastapi.testclient import TestClient

    yield TestClient(api_main.app), api_main.db, api_main.store


def test_dashboard_renders_pending_and_history(client):
    test_client, db, store = client
    db.store_experiment(hypothesis="improve x", diff="diff1", rationale="r", metrics={}, outcome="success")
    store.create_request("cand-1", "improve z", "diff3", 0.85, {})

    r = test_client.get("/")
    assert r.status_code == 200
    assert "cand-1" in r.text
    assert "improve x" in r.text


def test_pending_endpoint_lists_only_pending(client):
    test_client, db, store = client
    req_id = store.create_request("cand-1", "goal", "diff", 0.9, {})
    store.decide(req_id, "approved")
    store.create_request("cand-2", "goal", "diff", 0.5, {})

    pending = test_client.get("/api/pending").json()
    assert len(pending) == 1
    assert pending[0]["candidate_id"] == "cand-2"


def test_approve_endpoint_persists_decision(client):
    test_client, db, store = client
    req_id = store.create_request("cand-1", "goal", "diff", 0.9, {})

    r = test_client.post(f"/approvals/{req_id}/approve", data={"note": "ship it"}, follow_redirects=False)
    assert r.status_code == 303

    stored = store.get_request(req_id)
    assert stored["status"] == "approved"
    assert stored["decision_note"] == "ship it"


def test_reject_endpoint_persists_decision(client):
    test_client, db, store = client
    req_id = store.create_request("cand-1", "goal", "diff", 0.9, {})

    test_client.post(f"/approvals/{req_id}/reject", follow_redirects=False)

    stored = store.get_request(req_id)
    assert stored["status"] == "rejected"


def test_report_endpoint_matches_raw_data(client):
    """The API's /api/report numbers must match what's actually in the underlying stores - not a separately maintained count."""
    test_client, db, store = client
    db.store_experiment(hypothesis="a", diff="d1", rationale="r", metrics={}, outcome="success")
    db.store_experiment(hypothesis="b", diff="d2", rationale="r", metrics={}, outcome="failure")
    db.store_experiment(hypothesis="c", diff="d3", rationale="r", metrics={}, outcome="held")

    report = test_client.get("/api/report").json()
    raw_experiments = db.list_all_experiments()

    assert report["total_experiments"] == len(raw_experiments)
    assert report["success_count"] == sum(1 for e in raw_experiments if e["outcome"] == "success")
    assert report["held_count"] == sum(1 for e in raw_experiments if e["outcome"] == "held")
    assert report["failure_count"] == sum(1 for e in raw_experiments if e["outcome"] == "failure")


def test_history_endpoint_matches_db_count(client):
    test_client, db, store = client
    for i in range(5):
        db.store_experiment(hypothesis=f"h{i}", diff="d", rationale="r", metrics={}, outcome="success")

    history = test_client.get("/api/history").json()
    assert len(history) == len(db.list_all_experiments())
