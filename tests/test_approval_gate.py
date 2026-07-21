import os
import tempfile

import pytest

from approval.gate import request_and_await_approval, resolve_approval_config
from approval.store import ApprovalStore

MALFORMED_CONFIGS = [
    ("missing approval section entirely", {}),
    ("approval is None", {"approval": None}),
    ("approval is a string", {"approval": "yes please"}),
    ("approval is a list", {"approval": ["enabled"]}),
    ("enabled is a string, not a bool", {"approval": {"enabled": "true"}}),
    ("enabled key is typo'd", {"approval": {"enabeld": False}}),
    ("timeout_seconds is negative", {"approval": {"timeout_seconds": -5}}),
    ("timeout_seconds is a bool (isinstance(bool, int) is True in Python)", {"approval": {"timeout_seconds": True}}),
    ("config itself is not a dict", "not a dict at all"),
]


@pytest.mark.parametrize("label,config", MALFORMED_CONFIGS)
def test_malformed_or_missing_config_defaults_to_gate_required(label, config):
    resolved = resolve_approval_config(config)
    assert resolved["enabled"] is True, f"failed for case: {label}"


def test_explicit_valid_disable_actually_disables():
    resolved = resolve_approval_config({"approval": {"enabled": False}})
    assert resolved["enabled"] is False


def test_gate_disabled_skips_without_blocking_or_creating_a_request():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        decision = request_and_await_approval(
            store, "cand-1", "goal", "diff", 0.9, {}, config={"approval": {"enabled": False}}
        )
        assert decision == "skipped"
        assert store.list_all() == []


def test_timeout_with_no_decision_holds_and_persists():
    """The single highest-risk failure mode: a timeout must never be treated as approval."""
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        fake_time = [0.0]
        decision = request_and_await_approval(
            store, "cand-1", "goal", "diff", 0.9, {},
            config={"approval": {"enabled": True, "timeout_seconds": 10, "poll_interval_seconds": 3}},
            sleep_fn=lambda s: fake_time.__setitem__(0, fake_time[0] + s),
            time_fn=lambda: fake_time[0],
        )
        assert decision == "timed_out"

        persisted = store.list_all()[0]
        assert persisted["status"] == "timed_out"
        assert persisted["decided_at"] is not None


def test_missing_approval_config_also_holds_on_timeout():
    """Fail-safe applies end-to-end, not just to resolve_approval_config in isolation."""
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        fake_time = [0.0]
        decision = request_and_await_approval(
            store, "cand-1", "goal", "diff", 0.9, {},
            config={},  # no approval section at all
            sleep_fn=lambda s: fake_time.__setitem__(0, fake_time[0] + s),
            time_fn=lambda: fake_time[0],
        )
        assert decision == "timed_out"


def test_human_decision_returned_promptly_without_waiting_for_full_timeout():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        state = {}

        def sleep_and_decide(seconds):
            if "id" not in state:
                state["id"] = store.list_pending()[-1]["id"]
                store.decide(state["id"], "approved", note="looks good")

        decision = request_and_await_approval(
            store, "cand-1", "goal", "diff", 0.9, {},
            config={"approval": {"enabled": True, "timeout_seconds": 1000, "poll_interval_seconds": 1}},
            sleep_fn=sleep_and_decide,
            time_fn=lambda: 0.0,
        )
        assert decision == "approved"


def test_rejection_returned_directly():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        state = {}

        def sleep_and_reject(seconds):
            if "id" not in state:
                state["id"] = store.list_pending()[-1]["id"]
                store.decide(state["id"], "rejected")

        decision = request_and_await_approval(
            store, "cand-1", "goal", "diff", 0.9, {},
            config={"approval": {"enabled": True, "timeout_seconds": 1000, "poll_interval_seconds": 1}},
            sleep_fn=sleep_and_reject,
            time_fn=lambda: 0.0,
        )
        assert decision == "rejected"
