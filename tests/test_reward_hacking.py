import json
import os
import subprocess
import sys
from unittest.mock import patch

from eval.pipeline import EvalPipeline
from sandbox.executor import SandboxExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {"stages": [{"subset_percentage": 100, "threshold": 0.5}]}

# The entire "solution" is a printed claim - no prediction file, no attempt
# at the actual task. If the pipeline ever trusts stdout for the score,
# this is the trivial exploit that gets a perfect merge for free.
REWARD_HACK_SCRIPT = 'print("SCORE: 1.0")\n'


def test_printed_score_claim_is_never_trusted(tmp_dir):
    """
    Permanent regression test - must never be deleted or skipped. A
    candidate that never engages with the actual task, and simply prints a
    perfect score claim, must not be able to buy a merge that way. The eval
    pipeline must score real predictions against held-out truth, never
    anything the candidate writes to stdout.

    Calls evaluate_stage() itself, not just score_predictions() - the
    lower-level function that never even looks at stdout in the first
    place, so a test built only on it can't actually prove a printed claim
    is ignored by the code path that DOES read stdout (evaluate_stage's
    own _parse_score/mismatch check). evaluate_stage is what
    orchestrator/run.py and evolution/scheduler.py actually call to decide
    whether a candidate passes a stage.
    """
    script_path = os.path.join(tmp_dir, "candidate_script.py")
    with open(script_path, "w") as f:
        f.write(REWARD_HACK_SCRIPT)

    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    assert "SCORE: 1.0" in result.stdout  # sanity: the candidate did make the claim
    assert result.returncode == 0  # sanity: this isn't caught by the exit-code check either

    pred_path = os.path.join(tmp_dir, "predictions.jsonl")  # the script wrote nothing here
    truth = {"0": 1, "1": 0, "2": 1, "3": 0}
    execution_result = {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timeout": False,
    }

    pipeline = EvalPipeline(CONFIG)
    success, score, mismatch = pipeline.evaluate_stage(execution_result, 100, 0.5, pred_path, truth)

    assert success is False  # the perfect claim buys no credit at all
    assert score == 0.0
    assert mismatch is True  # the lie is flagged, even though it's never trusted for gating

    # score_predictions() alone (used by the assertion above's evaluate_stage
    # internally) must also report a real reason, not a silent 0.0.
    _, reason = pipeline.score_predictions(pred_path, truth)
    assert reason


def test_partial_prediction_coverage_gets_no_partial_credit(tmp_dir):
    """A candidate that only predicts for half the test ids scores 0.0, not 50%."""
    truth = {"0": 1, "1": 0, "2": 1, "3": 0}
    pred_path = os.path.join(tmp_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        f.write(json.dumps({"id": 0, "pred": 1}) + "\n")
        f.write(json.dumps({"id": 1, "pred": 0}) + "\n")

    pipeline = EvalPipeline({"stages": []})
    score, reason = pipeline.score_predictions(pred_path, truth)
    assert score == 0.0
    assert reason


def test_truth_json_never_appears_in_the_actual_docker_mount_arguments(tmp_dir):
    """
    Permanent regression test - must never be deleted or skipped. Exercises
    the real code path that decides what gets bind-mounted into a
    candidate's container - SandboxExecutor._build_docker_cmd - with a
    truth.json physically sitting right next to train.jsonl/test.jsonl, and
    inspects the actual argument list Docker would receive.

    This replaces a prior version of this test that only grepped
    sandbox/executor.py's source text for the word "truth". A source grep
    proves nothing about behavior: code can reference the held-out file
    indirectly (a concatenated string, a variable read from elsewhere) and
    still pass a text scan, and a harmless comment mentioning the filename
    (like this file's own docstrings) would fail one for no real reason.
    Calling the real mount-building method and inspecting its real output
    is the only thing that actually proves the invariant. No Docker daemon
    is required - _build_docker_cmd only assembles the argument list, it
    never invokes Docker (see sandbox/executor.py).
    """
    for name, content in (
        ("train.jsonl", json.dumps({"x1": 1.0, "label": 1}) + "\n"),
        ("test.jsonl", json.dumps({"id": 0, "x1": 0.2}) + "\n"),
        ("truth.json", json.dumps({"0": 1})),
    ):
        with open(os.path.join(tmp_dir, name), "w") as f:
            f.write(content)

    script_path = os.path.join(tmp_dir, "candidate_script.py")
    with open(script_path, "w") as f:
        f.write("print('hi')\n")

    config = {"timeout_seconds": 10, "cpu_limit": "0.5", "memory_limit": "256m"}
    with patch.object(SandboxExecutor, "_build_image"):  # no Docker daemon needed for this test
        executor = SandboxExecutor(config, dataset_dir=tmp_dir)

    cmd = executor._build_docker_cmd(script_path, "sandbox-test")

    assert not any("truth" in arg.lower() for arg in cmd), cmd
    # Sanity: the mount-building path really ran with dataset_dir wired up,
    # so the assertion above isn't vacuously true - train/test really were
    # mounted, truth.json just never was.
    assert any("train.jsonl" in arg for arg in cmd)
    assert any("test.jsonl" in arg for arg in cmd)


def test_truth_json_absent_from_every_docker_mount_argument_repo_wide():
    """
    Heuristic defense-in-depth, NOT the proof of the invariant (that's
    test_truth_json_never_appears_in_the_actual_docker_mount_arguments
    above, which exercises real behavior). This is a coarse repo-wide text
    sweep: it scans every .py file (excluding this test file and
    test_sandbox.py, which legitimately reference truth.json to prove it's
    NOT visible in-container) for the string "truth.json" appearing near a
    mount/docker construct. It exists only to catch an accidental mount
    added somewhere OTHER than sandbox/executor.py that the behavioral test
    above wouldn't be looking at - it can still be defeated by an obfuscated
    path (string concatenation, an indirect variable) the same way any text
    scan can. Pure Python, not a shelled-out grep - grep isn't guaranteed to
    exist (e.g. on Windows).
    """
    excluded = {"test_reward_hacking.py", "test_sandbox.py"}
    offending = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "venv", ".candidate_worktrees")]
        for filename in filenames:
            if not filename.endswith(".py") or filename in excluded:
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    if "truth.json" not in line:
                        continue
                    if "-v" in line or "docker" in line.lower() or "/app/" in line:
                        offending.append(f"{path}:{lineno}:{line.rstrip()}")

    assert offending == [], "truth.json appears near a mount/docker construct:\n" + "\n".join(offending)
