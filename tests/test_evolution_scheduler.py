import os
import subprocess
from unittest.mock import MagicMock

from approval.store import ApprovalStore
from evolution.scheduler import ConcurrentScheduler
from vcs.git_controller import GitController


def _init_repo(tmp_dir):
    repo_dir = os.path.join(tmp_dir, "repo")
    os.makedirs(repo_dir)
    with open(os.path.join(repo_dir, "candidate_script.py"), "w") as f:
        f.write("\n")
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)
    return repo_dir


def test_all_sandbox_evaluation_completes_before_any_approval_request_blocks(tmp_dir):
    """
    Wave 3 acceptance: sandbox execution for a whole generation must
    complete before any approval request blocks compute. With a single
    worker, the old single-phase design would create and await candidate
    1's approval - blocking - before candidate 2's sandbox ever ran; the
    two-phase design runs every candidate's sandbox stages first and only
    then creates and awaits approval requests.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    events = []

    class RecordingSandbox:
        def run_candidate(self, script_path, env_vars=None, out_dir=None):
            events.append(("sandbox", script_path))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
                f.write('{"id": "0", "pred": 1}\n')
            return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}

    class RecordingStore(ApprovalStore):
        def create_request(self, candidate_id, *a, **kw):
            events.append(("approval_request", candidate_id))
            return super().create_request(candidate_id, *a, **kw)

    store = RecordingStore(os.path.join(tmp_dir, "approvals.db"))
    evaluator = MagicMock()
    evaluator.evaluate_stage.return_value = (True, 1.0, False)

    candidates = [
        {"id": "aaa", "diff": "", "goal": "g"},
        {"id": "bbb", "diff": "", "goal": "g"},
    ]

    # max_workers=1: a single sandbox slot, so the two candidates run their
    # sandbox stages strictly one after another - the ordering assertion
    # below is meaningless with more workers, where sandbox calls could
    # interleave with approval creation for unrelated reasons.
    scheduler = ConcurrentScheduler(max_workers=1)
    gate_config = {"approval": {"enabled": True, "timeout_seconds": 0.05, "poll_interval_seconds": 0.01}}

    scheduler.execute_generation(
        candidates,
        eval_stages=[{"subset_percentage": 100, "threshold": 0.5}],
        git_controller=git_controller,
        sandbox=RecordingSandbox(),
        evaluator=evaluator,
        metrics_calculator=lambda r: {},
        failure_analyzer=lambda r, passed: ("failure", ""),
        approval_store=store,
        approval_config=gate_config,
        truth={"0": 1},
    )

    sandbox_indices = [i for i, e in enumerate(events) if e[0] == "sandbox"]
    approval_indices = [i for i, e in enumerate(events) if e[0] == "approval_request"]

    assert len(sandbox_indices) == 2
    assert len(approval_indices) == 2
    assert max(sandbox_indices) < min(approval_indices)


def test_syntax_error_is_rejected_before_ever_reaching_the_sandbox(tmp_dir):
    """
    Council audit finding: evolutionary mode never ran the static
    syntax-check pass that orchestrator/run.py's sequential path always
    does - README claims a bad-syntax candidate is rejected before it ever
    reaches the sandbox, but evolutionary mode burned a full sandbox run on
    it regardless. A candidate whose diff produces invalid Python must be
    rejected as a syntax_error without ever calling sandbox.run_candidate.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    sandbox_calls = []

    class RecordingSandbox:
        def run_candidate(self, script_path, env_vars=None, out_dir=None):
            sandbox_calls.append(script_path)
            return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}

    store = ApprovalStore(os.path.join(tmp_dir, "approvals.db"))
    evaluator = MagicMock()
    evaluator.evaluate_stage.return_value = (True, 1.0, False)

    broken_diff = (
        "--- a/candidate_script.py\n"
        "+++ b/candidate_script.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-\n"
        "+def broken(:\n"
    )
    candidates = [{"id": "aaa", "diff": broken_diff, "goal": "g"}]
    scheduler = ConcurrentScheduler(max_workers=1)

    evaluated = scheduler.execute_generation(
        candidates,
        eval_stages=[{"subset_percentage": 100, "threshold": 0.5}],
        git_controller=git_controller,
        sandbox=RecordingSandbox(),
        evaluator=evaluator,
        metrics_calculator=lambda r: {},
        failure_analyzer=lambda r, passed: ("failure", ""),
        approval_store=store,
        approval_config={"approval": {"enabled": False}},
        truth={"0": 1},
    )

    assert sandbox_calls == []
    assert evaluated[0]["eval_passed"] is False
    assert evaluated[0]["failure_category"] == "syntax_error"


def test_progressive_stages_mount_a_host_selected_subset_not_the_full_train_file(tmp_dir):
    """
    Council audit finding: progressive-scaling stages used to mount the
    SAME full train.jsonl at every stage - SUBSET_PERCENTAGE was only an
    environment variable a candidate's own code could ignore entirely, and
    still see (and train on) 100% of the data at every stage. When
    train_path is supplied, each stage must instead receive a
    train_path_override pointing at a file containing only that stage's
    subset - and different stages must get DIFFERENT subset files, not the
    same one reused throughout.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    full_train_path = os.path.join(tmp_dir, "train.jsonl")
    with open(full_train_path, "w") as f:
        for i in range(100):
            f.write(f'{{"x1": {i}.0, "x2": 0.0, "label": {i % 2}}}\n')

    # The override path is reused (overwritten) across stages and its
    # directory is cleaned up once execute_generation returns, so the
    # ROW COUNT must be captured immediately, at call time, not read back
    # from the path afterwards.
    seen_row_counts = []
    seen_overrides = []

    class RecordingSandbox:
        def run_candidate(self, script_path, env_vars=None, out_dir=None, train_path_override=None):
            seen_overrides.append(train_path_override)
            with open(train_path_override) as f:
                seen_row_counts.append(len(f.read().strip().splitlines()))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
                f.write('{"id": "0", "pred": 1}\n')
            return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}

    store = ApprovalStore(os.path.join(tmp_dir, "approvals.db"))
    evaluator = MagicMock()
    evaluator.evaluate_stage.return_value = (True, 0.9, False)

    diff = (
        "--- a/candidate_script.py\n"
        "+++ b/candidate_script.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-\n"
        "+print('hi')\n"
    )
    candidates = [{"id": "aaa", "diff": diff, "goal": "g"}]
    scheduler = ConcurrentScheduler(max_workers=1)

    scheduler.execute_generation(
        candidates,
        eval_stages=[{"subset_percentage": 5, "threshold": 0.5}, {"subset_percentage": 50, "threshold": 0.5}],
        git_controller=git_controller,
        sandbox=RecordingSandbox(),
        evaluator=evaluator,
        metrics_calculator=lambda r: {},
        failure_analyzer=lambda r, passed: ("failure", ""),
        approval_store=store,
        approval_config={"approval": {"enabled": False}},
        truth={"0": 1},
        train_path=full_train_path,
    )

    assert len(seen_overrides) == 2
    assert all(seen_overrides)  # never None/empty - always a real path
    assert all(p != full_train_path for p in seen_overrides)  # never the full file itself
    assert seen_row_counts[0] < seen_row_counts[1] < 100  # 5% stage < 50% stage < the full 100 rows


class _FlippingBaselineStore:
    """
    A baseline whose .passes() answer changes between calls, simulating
    another candidate raising the real baseline_store in between phase 1's
    read and phase 2's finalize. Phase 1 sees "clears it" (1st call); the
    re-check immediately before finalize_merge sees "no longer clears it"
    (2nd call) - a real BaselineStore.passes() would behave exactly this
    way if update_if_better() were called on it between those two reads.
    """
    def __init__(self):
        self.passes_calls = 0
        self.updated = []

    def get(self, stage_key):
        return None

    def passes(self, stage_key, score, min_improvement=0.001):
        self.passes_calls += 1
        return self.passes_calls == 1

    def update_if_better(self, stage_key, score):
        self.updated.append((stage_key, score))


def test_phase2_rechecks_the_baseline_immediately_before_finalizing(tmp_dir):
    """
    Council audit finding: phase 2 used to finalize every approved
    candidate using only the baseline comparison phase 1 made BEFORE any
    candidate in this generation had merged - so a candidate whose score
    cleared the baseline at evaluation time could still finalize even after
    another candidate (processed earlier in this same phase 2 loop) raised
    the baseline past it. The merge must be gated on a fresh baseline read
    taken under the same lock that serializes finalization, not the stale
    one from phase 1.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    class RecordingSandbox:
        def run_candidate(self, script_path, env_vars=None, out_dir=None):
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
                f.write('{"id": "0", "pred": 1}\n')
            return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}

    store = ApprovalStore(os.path.join(tmp_dir, "approvals.db"))
    baseline_store = _FlippingBaselineStore()

    evaluator = MagicMock()
    evaluator.evaluate_stage.return_value = (True, 0.9, False)

    # A genuine diff (not empty) - patch_generator applies it for real.
    diff = (
        "--- a/candidate_script.py\n"
        "+++ b/candidate_script.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-\n"
        "+print('hi')\n"
    )
    candidates = [{"id": "aaa", "diff": diff, "goal": "g"}]
    scheduler = ConcurrentScheduler(max_workers=1)
    gate_config = {"approval": {"enabled": False}}  # skip straight to finalize

    evaluated = scheduler.execute_generation(
        candidates,
        eval_stages=[{"subset_percentage": 100, "threshold": 0.5}],
        git_controller=git_controller,
        sandbox=RecordingSandbox(),
        evaluator=evaluator,
        metrics_calculator=lambda r: {},
        failure_analyzer=lambda r, passed: ("failure", ""),
        approval_store=store,
        approval_config=gate_config,
        truth={"0": 1},
        baseline_store=baseline_store,
    )

    assert baseline_store.passes_calls >= 2, "phase 2 must re-read the baseline, not reuse phase 1's answer"
    assert evaluated[0]["success"] is False
    assert evaluated[0]["failure_category"] == "below_baseline"
    assert baseline_store.updated == []  # never credited as a merge


def test_phase2_rejects_a_duplicate_whose_rebase_collapses_to_the_current_base(tmp_dir):
    """
    Council audit finding: a candidate whose diff is byte-for-byte
    identical to one that already finalized earlier in the same generation
    rebases to nothing (git silently drops the duplicate commit rather than
    conflicting) - the old code counted this as a "success" merge even
    though it published nothing new. Two real candidates with the exact
    same diff: the first must merge, the second must be rejected as a
    no-op, not recorded as success.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    class RecordingSandbox:
        def run_candidate(self, script_path, env_vars=None, out_dir=None):
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
                f.write('{"id": "0", "pred": 1}\n')
            return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}

    store = ApprovalStore(os.path.join(tmp_dir, "approvals.db"))

    evaluator = MagicMock()
    evaluator.evaluate_stage.return_value = (True, 0.9, False)

    identical_diff = (
        "--- a/candidate_script.py\n"
        "+++ b/candidate_script.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-\n"
        "+print('duplicate')\n"
    )
    candidates = [
        {"id": "aaa", "diff": identical_diff, "goal": "g"},
        {"id": "bbb", "diff": identical_diff, "goal": "g"},
    ]
    # max_workers=1 makes phase 1's completion order deterministic (aaa then
    # bbb), so phase 2 processes aaa first - the one that actually merges.
    scheduler = ConcurrentScheduler(max_workers=1)
    gate_config = {"approval": {"enabled": False}}

    evaluated = scheduler.execute_generation(
        candidates,
        eval_stages=[{"subset_percentage": 100, "threshold": 0.5}],
        git_controller=git_controller,
        sandbox=RecordingSandbox(),
        evaluator=evaluator,
        metrics_calculator=lambda r: {},
        failure_analyzer=lambda r, passed: ("failure", ""),
        approval_store=store,
        approval_config=gate_config,
        truth={"0": 1},
    )

    by_id = {c["id"]: c for c in evaluated}
    assert by_id["aaa"]["success"] is True
    assert by_id["bbb"]["success"] is False
    assert by_id["bbb"]["failure_category"] == "no_op_after_rebase"


def test_phase2_reevaluates_when_the_base_moved_since_phase1_scored_it(tmp_dir):
    """
    Council audit finding: phase 2 used to finalize a rebased candidate
    using its phase-1 score even when an earlier candidate's merge moved
    the base underneath it - so genuinely never-evaluated code (the
    rebased result) could be published on the strength of a score that
    described different code. Two candidates touching DIFFERENT lines (so
    the second's rebase applies cleanly, producing real - not duplicate -
    content) must trigger a second, real sandbox run for the second
    candidate once its base has moved.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    run_count = {"n": 0}

    class RecordingSandbox:
        def run_candidate(self, script_path, env_vars=None, out_dir=None):
            run_count["n"] += 1
            os.makedirs(out_dir, exist_ok=True)
            with open(out_dir + "/predictions.jsonl", "w") as f:
                f.write('{"id": "0", "pred": 1}\n')
            return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}

    store = ApprovalStore(os.path.join(tmp_dir, "approvals.db"))

    evaluator = MagicMock()
    evaluator.evaluate_stage.return_value = (True, 0.9, False)

    # aaa adds a line at the top; bbb adds a DIFFERENT line at the bottom -
    # bbb's rebase onto aaa's already-merged content applies cleanly
    # (disjoint context) and produces genuinely new content, not a
    # duplicate collapse.
    diff_a = (
        "--- a/candidate_script.py\n"
        "+++ b/candidate_script.py\n"
        "@@ -1,1 +1,2 @@\n"
        "-\n"
        "+print('from a')\n"
        "+\n"
    )
    diff_b = (
        "--- a/candidate_script.py\n"
        "+++ b/candidate_script.py\n"
        "@@ -1,1 +1,2 @@\n"
        " \n"
        "+print('from b')\n"
    )
    candidates = [
        {"id": "aaa", "diff": diff_a, "goal": "g"},
        {"id": "bbb", "diff": diff_b, "goal": "g"},
    ]
    scheduler = ConcurrentScheduler(max_workers=1)
    gate_config = {"approval": {"enabled": False}}

    evaluated = scheduler.execute_generation(
        candidates,
        eval_stages=[{"subset_percentage": 100, "threshold": 0.5}],
        git_controller=git_controller,
        sandbox=RecordingSandbox(),
        evaluator=evaluator,
        metrics_calculator=lambda r: {},
        failure_analyzer=lambda r, passed: ("failure", ""),
        approval_store=store,
        approval_config=gate_config,
        truth={"0": 1},
    )

    by_id = {c["id"]: c for c in evaluated}
    assert by_id["aaa"]["success"] is True
    assert by_id["bbb"]["success"] is True
    # aaa: 1 phase-1 run. bbb: 1 phase-1 run + 1 phase-2 re-evaluation run
    # (its base moved once aaa finalized first).
    assert run_count["n"] == 3
