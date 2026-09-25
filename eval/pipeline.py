# =============================================================================
# HARD INVARIANT - read this in full before changing anything in this file,
# eval/dataset.py, or sandbox/executor.py:
#
#   1. truth.json (the held-out labels) is NEVER mounted into the sandbox.
#      It is loaded host-side only, via eval/dataset.py:load_truth(), and
#      passed into this module as an in-memory dict. No code path here (or
#      in sandbox/executor.py) may ever construct a docker mount argument
#      referencing truth.json, or make it readable from inside a container.
#      truth.json lives on disk in the SAME directory as train.jsonl/
#      test.jsonl (see eval/dataset.py:generate_split) - the only reason it
#      stays hidden is that sandbox/executor.py mounts individual files
#      (`-v host_path:container_path:ro`), never the whole directory. If
#      that ever changes to a directory-level mount, this invariant breaks
#      silently.
#
#   2. score_predictions() is the ONLY source of truth for a candidate's
#      score. Nothing a candidate writes to stdout/stderr is ever trusted
#      for gating - see _parse_score(), which exists purely to detect a
#      mismatch (a candidate lying about its own performance), never to
#      substitute for a real score.
#
# Both properties are enforced by permanent regression tests that must
# never be deleted, skipped, or weakened:
#   - tests/test_reward_hacking.py::test_printed_score_claim_is_never_trusted
#   - tests/test_reward_hacking.py::test_truth_json_never_appears_in_the_actual_docker_mount_arguments
#   - tests/test_reward_hacking.py::test_truth_json_absent_from_every_docker_mount_argument_repo_wide
#   - tests/test_sandbox.py::test_truth_json_unreachable_by_any_path_inside_the_sandbox
#     (walks the ENTIRE container filesystem for a file literally named
#     truth.json - proof that no path construction trick, not just the one
#     obvious path, can ever reach it)
#
# If you're about to mount a whole directory (rather than individual
# files) into the sandbox, or change how/where truth is loaded, stop and
# re-run every test above first.
# =============================================================================

import importlib
import json
import os
import re
from typing import Any, Callable, Dict, Optional, Tuple

from observability.logging_config import get_logger

SCORE_PATTERN = re.compile(r"SCORE:\s*([-+]?\d*\.?\d+)")
_module_logger = get_logger(__name__)
SCORE_CLAIM_MISMATCH_THRESHOLD = 0.05
# predictions.jsonl is written by the candidate's own code running inside
# the sandbox - a legitimate prediction file is a few dozen bytes per test
# row, so this is generous headroom (tens of thousands of rows), not a tight
# fit. It exists to stop a candidate from turning the host read into a DoS:
# without a cap, a predictions.jsonl that is either a symlink to /dev/zero
# or just a very large regular file gets read here (host-side, outside the
# sandbox's own memory limits) in one unbounded `for line in f` pass.
MAX_PREDICTIONS_FILE_BYTES = 64 * 1024 * 1024

Scorer = Callable[[Dict[str, Any], Dict[str, Any]], Tuple[float, str]]


def load_predictions(pred_path: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Loads predictions.jsonl into an id -> raw prediction value dict,
    enforcing the security checks that must hold regardless of task or
    scorer: never a symlink, never over the size cap, must actually parse.
    Returns (preds, reason) - preds is None on any failure, with reason
    explaining why (the caller scores 0.0 in that case). Deliberately does
    NOT constrain what a prediction value IS (int/float/str) or check it
    against truth's id set - that's the scorer's job, since it varies by
    task (see binary_accuracy_scorer below for the built-in one).
    """
    if not os.path.exists(pred_path):
        return None, "no predictions written"

    # A candidate script runs inside the sandbox but still controls what
    # ends up at this host path - a symlink to /dev/zero (or any other
    # infinite/huge source) turns an ordinary host-side read into a
    # denial of service, since nothing here is bound by the sandbox's
    # own memory limits. os.path.islink is a TOCTOU check, not airtight
    # against a concurrent swap, but the writer (the candidate's
    # process) has already exited by the time this runs - there is no
    # legitimate reason for that race to ever occur.
    if os.path.islink(pred_path):
        return None, "predictions file is a symlink, refusing to read it"
    try:
        file_size = os.path.getsize(pred_path)
    except OSError as e:
        return None, f"could not stat predictions file: {e}"
    if file_size > MAX_PREDICTIONS_FILE_BYTES:
        return None, f"predictions file too large ({file_size} bytes > {MAX_PREDICTIONS_FILE_BYTES} cap)"

    preds: Dict[str, Any] = {}
    try:
        with open(pred_path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                preds[str(row["id"])] = row["pred"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None, f"malformed predictions file: {e}"
    except OSError as e:
        return None, f"could not read predictions file: {e}"

    return preds, ""


def binary_accuracy_scorer(preds: Dict[str, Any], truth: Dict[str, int]) -> Tuple[float, str]:
    """
    Default scorer, unchanged from the original hardcoded behavior:
    predictions must exactly cover truth's id set and be 0/1, scored as
    plain accuracy. A project with a different task (regression, ranking,
    multi-class, ...) sets eval.scorer to its own "module:function"
    implementing this same (preds, truth) -> (score, reason) contract.
    """
    if set(preds) != set(truth):
        return 0.0, f"prediction id set mismatch ({len(preds)} vs {len(truth)})"

    try:
        preds_int = {k: int(v) for k, v in preds.items()}
    except (TypeError, ValueError):
        return 0.0, "predictions must be 0 or 1"

    if not all(v in (0, 1) for v in preds_int.values()):
        return 0.0, "predictions must be 0 or 1"

    correct = sum(1 for i, y in truth.items() if preds_int[i] == y)
    return correct / len(truth), ""


def _load_scorer(dotted_path: str) -> Scorer:
    """Resolves eval.scorer ("module.path:function_name") to a callable - fails at construction time, not mid-run, on a bad path."""
    module_name, sep, func_name = dotted_path.partition(":")
    if not sep:
        raise ValueError(f"eval.scorer must be 'module.path:function_name', got {dotted_path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(f"eval.scorer: could not import {module_name!r}: {e}")
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise ValueError(f"eval.scorer: {module_name!r} has no attribute {func_name!r}")


class EvalPipeline:
    """
    Progressive-scaling evaluator. The candidate's own stdout is never
    trusted for the score that gates a merge - score_predictions() scores
    real predictions against held-out truth the candidate never saw. A
    printed "SCORE:" line (if any) is parsed only as a diagnostic, to
    detect a candidate lying about its own performance.
    """
    def __init__(self, config: Dict[str, Any]):
        self.stages = config.get('stages', [])
        self.correlation_log = []
        # Pluggable per-task scoring (see eval.scorer in configs/example.yaml) -
        # defaults to the built-in binary-accuracy behavior, unchanged from
        # before this was pluggable. The file-safety checks in
        # load_predictions() are NOT part of the plugin surface - every
        # scorer gets predictions that already passed them.
        self.scorer: Scorer = _load_scorer(config.get('scorer', 'eval.pipeline:binary_accuracy_scorer'))

    def score_predictions(self, pred_path: str, truth: Dict[str, int]) -> Tuple[float, str]:
        """
        Scores predictions written by the candidate against held-out
        truth, via self.scorer. Returns (score, reason) - reason is "" on
        a clean score, otherwise a human-readable explanation of why the
        score is 0.0. Never raises: a missing file, malformed JSON, or
        anything the scorer itself rejects all score 0.0 with a reason
        instead of an exception escaping to the caller.
        """
        preds, reason = load_predictions(pred_path)
        if preds is None:
            return 0.0, reason
        return self.scorer(preds, truth)

    def evaluate_stage(self, execution_result: Dict[str, Any], subset_percentage: int, threshold: float,
                        pred_path: str, truth: Dict[str, int], logger=None) -> Tuple[bool, float, bool]:
        """
        Evaluates a single stage's real execution result. Returns (success,
        score, score_claim_mismatch) - all three are returned directly,
        never stashed on `self`, because this EvalPipeline instance is
        shared across every candidate and stage (see evolution/scheduler.py,
        which evaluates several candidates concurrently across threads): an
        instance attribute set here and read by the caller "immediately
        after" the call is exactly the kind of state that isn't actually
        safe under concurrent calls - a second thread's call can overwrite
        it before the first thread reads its own result back.

        logger defaults to a module-level logger with no run/candidate context -
        callers that have it (orchestrator/run.py, evolution/scheduler.py) should
        pass a logger already bound with candidate_id so these records can be
        correlated back to the candidate they're about.
        """
        log = logger or _module_logger
        if execution_result['exit_code'] != 0 or execution_result.get('timeout', False):
            return False, 0.0, False

        score, reason = self.score_predictions(pred_path, truth)
        if reason:
            log.info(f"Stage {subset_percentage}%: prediction scoring failed: {reason}")

        claimed = self._parse_score(execution_result)
        mismatch = claimed is not None and abs(claimed - score) > SCORE_CLAIM_MISMATCH_THRESHOLD
        if mismatch:
            log.warning(f"score_claim_mismatch: candidate claimed SCORE={claimed:.4f}, real score={score:.4f}")

        success = score >= threshold

        log.info(f"Stage {subset_percentage}%: Score={score:.4f}, Threshold={threshold}")
        if not success:
            log.info(f"Candidate failed at {subset_percentage}% subset.")

        return success, score, mismatch

    @staticmethod
    def _parse_score(execution_result: Dict[str, Any]) -> Optional[float]:
        """Diagnostic only - the candidate's own stdout claim, never trusted for gating."""
        match = SCORE_PATTERN.search(execution_result.get('stdout', ''))
        if not match:
            return None
        return float(match.group(1))
