import os
import tempfile

from approval.store import ApprovalStore


def test_create_and_get_request():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff text", 0.9, {"k": 1})

        req = store.get_request(request_id)
        assert req["status"] == "pending"
        assert req["candidate_id"] == "cand-1"
        assert req["final_score"] == 0.9
        assert req["metrics"] == {"k": 1}


def test_decisions_are_persisted_across_a_fresh_store_instance():
    """Regression test: decisions must survive a process restart, not live only in memory."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "approvals.db")
        store1 = ApprovalStore(db_path)
        request_id = store1.create_request("cand-1", "goal", "diff", 0.9, {})
        store1.decide(request_id, "approved", note="fine")

        store2 = ApprovalStore(db_path)  # simulates a separate/restarted process
        req = store2.get_request(request_id)
        assert req["status"] == "approved"
        assert req["decision_note"] == "fine"


def test_list_pending_excludes_decided_requests():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        pending_id = store.create_request("cand-1", "goal", "diff", 0.9, {})
        decided_id = store.create_request("cand-2", "goal", "diff", 0.5, {})
        store.decide(decided_id, "rejected")

        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == pending_id


def test_decide_is_a_noop_once_already_decided():
    """A late human click must never override an automatic timeout hold, or vice versa."""
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff", 0.9, {})

        first = store.decide(request_id, "timed_out")
        second = store.decide(request_id, "approved")  # arrives too late

        assert first is True
        assert second is False
        assert store.get_request(request_id)["status"] == "timed_out"


def test_decide_rejects_invalid_status():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff", 0.9, {})
        try:
            store.decide(request_id, "maybe")
            assert False, "expected ValueError"
        except ValueError:
            pass
