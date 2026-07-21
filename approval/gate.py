import time
from typing import Any, Dict

from approval.store import ApprovalStore

DEFAULT_TIMEOUT_SECONDS = 1800  # 30 minutes
DEFAULT_POLL_INTERVAL_SECONDS = 2


def resolve_approval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fail-safe config resolution: a missing 'approval' section, one that
    isn't even a dict, or one with the wrong type for any of its keys must
    never be interpreted as "no gate needed." The gate is required unless
    a config explicitly and validly sets approval.enabled: false.
    """
    section = config.get("approval") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        section = {}

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        enabled = True  # malformed type -> fail safe to required

    timeout_seconds = section.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    poll_interval = section.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
    if not isinstance(poll_interval, (int, float)) or isinstance(poll_interval, bool) or poll_interval <= 0:
        poll_interval = DEFAULT_POLL_INTERVAL_SECONDS

    return {"enabled": enabled, "timeout_seconds": timeout_seconds, "poll_interval_seconds": poll_interval}


def request_and_await_approval(
    store: ApprovalStore,
    candidate_id: str,
    goal: str,
    diff: str,
    final_score: float,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    sleep_fn=time.sleep,
    time_fn=time.monotonic,
) -> str:
    """
    Creates an approval request and blocks (polling the persisted store)
    until a human decides, or the configured timeout elapses.

    Returns one of:
      "approved"  - a human approved it; the caller may merge.
      "rejected"  - a human rejected it; the caller must roll back.
      "timed_out" - no decision arrived in time; the caller must roll back.
                    This is a real, persisted decision, not a silent
                    fallback - it is never treated as approval.
      "skipped"   - the gate is disabled via an explicit, valid config.

    Callers must only merge when the return value is exactly "approved".
    """
    approval_cfg = resolve_approval_config(config)
    if not approval_cfg["enabled"]:
        return "skipped"

    request_id = store.create_request(candidate_id, goal, diff, final_score, metrics)
    deadline = time_fn() + approval_cfg["timeout_seconds"]

    while True:
        try:
            request = store.get_request(request_id)
        except Exception:
            # store unreachable/corrupted mid-wait - fail safe, do not merge
            return "timed_out"

        if request is None or request["status"] != "pending":
            return request["status"] if request else "timed_out"

        if time_fn() >= deadline:
            store.decide(request_id, "timed_out", note="No decision within timeout window.")
            return "timed_out"

        sleep_fn(approval_cfg["poll_interval_seconds"])
