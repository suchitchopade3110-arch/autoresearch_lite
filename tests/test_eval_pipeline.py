import json
import os

import pytest

from eval.pipeline import EvalPipeline, run_holdout_evaluation

CONFIG = {"stages": [
    {"subset_percentage": 1, "threshold": 0.5},
    {"subset_percentage": 100, "threshold": 0.8},
]}

TRUTH = {"0": 1, "1": 0, "2": 1, "3": 0}


def _always_score_point_five(preds, truth):
    """A trivial stand-in for a project's own metric, used to prove eval.scorer is actually invoked."""
    return 0.5, ""


def make_result(stdout="", exit_code=0, timeout=False):
    return {"exit_code": exit_code, "stdout": stdout, "stderr": "", "execution_time": 0.1, "timeout": timeout}


def write_predictions(tmp_dir, preds):
    path = os.path.join(tmp_dir, "predictions.jsonl")
    with open(path, "w") as f:
        for row in preds:
            f.write(json.dumps(row) + "\n")
    return path


def test_evaluate_stage_scores_real_predictions_not_stdout(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 1}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score, _ = pipeline.evaluate_stage(make_result(), 100, 0.8, pred_path, TRUTH)
    assert success
    assert score == 1.0


def test_evaluate_stage_ignores_a_perfect_stdout_claim(tmp_dir):
    """The exact reward-hacking exploit: a candidate claims SCORE: 1.0 but writes nothing useful."""
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 0}, {"id": 1, "pred": 0}, {"id": 2, "pred": 0}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score, mismatch = pipeline.evaluate_stage(make_result(stdout="SCORE: 1.0\n"), 100, 0.8, pred_path, TRUTH)
    assert not success
    assert score == 0.5  # 2 of 4 correct, regardless of the claimed 1.0
    assert mismatch is True  # the lie itself is still flagged, just never trusted for gating


def test_evaluate_stage_returns_the_mismatch_flag_directly_not_via_shared_state():
    """
    Council audit finding: the mismatch flag used to live on
    self.last_stage_flags, a single shared EvalPipeline instance's mutable
    attribute - safe only if callers read it immediately after the call,
    which concurrent callers (evolution/scheduler.py evaluates several
    candidates across threads) cannot guarantee: a second thread's call can
    overwrite it before the first thread reads its own result back. It must
    now be returned directly, and EvalPipeline must no longer even expose
    that attribute.
    """
    assert not hasattr(EvalPipeline({"stages": []}), "last_stage_flags")


def test_evaluate_stage_mismatch_flag_is_false_not_stale_on_a_crashed_stage(tmp_dir):
    """
    Council audit finding: the old shared last_stage_flags was never reset
    on the early "exit_code != 0" return, so a crashed stage's read could
    see a PREVIOUS candidate's leftover mismatch flag. The mismatch flag
    returned for a crashed/timed-out stage must always be False - there is
    no valid execution to have made a false claim about.
    """
    pipeline = EvalPipeline(CONFIG)
    pred_path = os.path.join(tmp_dir, "never_written.jsonl")
    success, score, mismatch = pipeline.evaluate_stage(make_result(exit_code=1), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert mismatch is False


def test_evaluate_stage_fails_below_threshold(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 0}, {"id": 1, "pred": 0}, {"id": 2, "pred": 0}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score, _ = pipeline.evaluate_stage(make_result(), 1, 0.9, pred_path, TRUTH)
    assert not success
    assert score == 0.5


def test_evaluate_stage_missing_predictions_scores_zero(tmp_dir):
    pipeline = EvalPipeline(CONFIG)
    pred_path = os.path.join(tmp_dir, "never_written.jsonl")
    success, score, _ = pipeline.evaluate_stage(make_result(), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_partial_id_coverage_scores_zero(tmp_dir):
    """A candidate that only predicts for half the test ids gets 0.0, not partial credit."""
    pred_path = write_predictions(tmp_dir, [{"id": 0, "pred": 1}, {"id": 1, "pred": 0}])
    pipeline = EvalPipeline(CONFIG)
    success, score, _ = pipeline.evaluate_stage(make_result(), 100, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_malformed_predictions_scores_zero_not_exception(tmp_dir):
    path = os.path.join(tmp_dir, "predictions.jsonl")
    with open(path, "w") as f:
        f.write("not json at all\n")
    pipeline = EvalPipeline(CONFIG)
    success, score, _ = pipeline.evaluate_stage(make_result(), 1, 0.5, path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_non_binary_prediction_scores_zero(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 7}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score, _ = pipeline.evaluate_stage(make_result(), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_nonzero_exit_fails(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 1}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score, _ = pipeline.evaluate_stage(make_result(exit_code=1), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_timeout_fails(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 1}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score, _ = pipeline.evaluate_stage(make_result(timeout=True), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_score_predictions_refuses_a_symlinked_predictions_file(tmp_dir):
    """
    Council audit finding: a candidate that makes predictions.jsonl a
    symlink to /dev/zero (or anywhere else) turns an unbounded host-side
    read into a denial of service, since this runs outside the sandbox's
    own memory limits. Must be refused before any read is attempted.
    """
    real_target = os.path.join(tmp_dir, "not_predictions_at_all.txt")
    with open(real_target, "w") as f:
        f.write("irrelevant")
    pred_path = os.path.join(tmp_dir, "predictions.jsonl")
    try:
        os.symlink(real_target, pred_path)
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("symlink creation not permitted in this environment")

    pipeline = EvalPipeline(CONFIG)
    score, reason = pipeline.score_predictions(pred_path, TRUTH)
    assert score == 0.0
    assert "symlink" in reason


def test_score_predictions_refuses_a_named_pipe_predictions_file(tmp_dir):
    """
    Council-audit finding: a FIFO (named pipe) at predictions.jsonl has no
    fixed size - os.stat().st_size on one is always 0, so it defeats the
    size cap the same way a symlink to /dev/zero would defeat it, and a
    candidate holding a writer open on the other end turns the host-side
    `for line in f` read into an indefinite hang rather than a fast error.
    Must be refused before any read is attempted, same as a symlink.
    """
    pred_path = os.path.join(tmp_dir, "predictions.jsonl")
    try:
        os.mkfifo(pred_path)
    except (AttributeError, OSError, NotImplementedError):
        pytest.skip("os.mkfifo not available in this environment")

    pipeline = EvalPipeline(CONFIG)
    score, reason = pipeline.score_predictions(pred_path, TRUTH)
    assert score == 0.0
    assert "pipe" in reason or "fifo" in reason.lower()


def test_score_predictions_refuses_an_oversized_predictions_file(tmp_dir, monkeypatch):
    """A regular (non-symlink) file that is simply huge must also be refused, not read in full."""
    from eval import pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "MAX_PREDICTIONS_FILE_BYTES", 10)

    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 1}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    assert os.path.getsize(pred_path) > 10

    pipeline = EvalPipeline(CONFIG)
    score, reason = pipeline.score_predictions(pred_path, TRUTH)
    assert score == 0.0
    assert "too large" in reason


def test_score_predictions_reports_a_reason_on_every_zero(tmp_dir):
    pipeline = EvalPipeline(CONFIG)

    score, reason = pipeline.score_predictions(os.path.join(tmp_dir, "missing.jsonl"), TRUTH)
    assert score == 0.0 and reason

    mismatched_path = write_predictions(tmp_dir, [{"id": 0, "pred": 1}])
    score, reason = pipeline.score_predictions(mismatched_path, TRUTH)
    assert score == 0.0 and reason


def test_eval_pipeline_defaults_to_binary_accuracy_scorer():
    pipeline = EvalPipeline(CONFIG)
    assert pipeline.scorer.__name__ == "binary_accuracy_scorer"


def test_eval_pipeline_accepts_a_custom_scorer_via_config(tmp_dir):
    """
    A project with a non-classification task points eval.scorer at its
    own "module:function" - the file-safety checks (symlink/size cap)
    still run first regardless, only the actual scoring is pluggable.
    """
    pred_path = write_predictions(tmp_dir, [{"id": 0, "pred": "anything"}])
    pipeline = EvalPipeline({**CONFIG, "scorer": "tests.test_eval_pipeline:_always_score_point_five"})

    score, reason = pipeline.score_predictions(pred_path, {"0": "irrelevant-to-this-scorer"})

    assert score == 0.5
    assert reason == ""


def test_eval_pipeline_custom_scorer_never_sees_a_symlinked_predictions_file(tmp_dir):
    """The symlink/size-cap safety checks apply to every scorer, not just the default - a custom scorer never even gets called."""
    real_path = write_predictions(tmp_dir, [{"id": 0, "pred": 1}])
    symlink_path = os.path.join(tmp_dir, "symlinked_predictions.jsonl")
    try:
        os.symlink(real_path, symlink_path)
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("symlink creation not permitted in this environment")

    pipeline = EvalPipeline({**CONFIG, "scorer": "tests.test_eval_pipeline:_always_score_point_five"})
    score, reason = pipeline.score_predictions(symlink_path, TRUTH)

    assert score == 0.0
    assert "symlink" in reason


def test_eval_pipeline_rejects_a_malformed_scorer_path_at_construction_time():
    """Fails fast when EvalPipeline is built, not lazily on the first scored candidate."""
    with pytest.raises(ValueError, match="module.path:function_name"):
        EvalPipeline({**CONFIG, "scorer": "not_a_dotted_path"})


def test_eval_pipeline_rejects_a_scorer_module_that_does_not_exist():
    with pytest.raises(ValueError, match="could not import"):
        EvalPipeline({**CONFIG, "scorer": "no_such_module_at_all:some_func"})


def test_eval_pipeline_rejects_a_scorer_function_that_does_not_exist():
    with pytest.raises(ValueError, match="no attribute"):
        EvalPipeline({**CONFIG, "scorer": "tests.test_eval_pipeline:no_such_function"})


class _FakeHoldoutSandbox:
    """Writes a fixed set of predictions to out_dir, as if a real container had run and exited 0."""
    def __init__(self, preds):
        self.preds = preds
        self.calls = []

    def run_candidate(self, script_path, env_vars=None, out_dir=None, extra_files=None,
                       train_path_override=None, test_path_override=None):
        self.calls.append({
            "env_vars": env_vars, "extra_files": extra_files,
            "train_path_override": train_path_override, "test_path_override": test_path_override,
        })
        with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
            for row in self.preds:
                f.write(json.dumps(row) + "\n")
        return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}


def test_run_holdout_evaluation_scores_against_holdout_truth_using_full_training_data():
    """
    Council-audit finding: the baseline gate re-uses the same selection set
    (test.jsonl) on every merge decision, letting it ratchet upward on that
    set's own sampling noise. run_holdout_evaluation scores a candidate
    exactly once against a SEPARATE sealed holdout, using the full
    (unsubsetted) training file rather than any progressive-scaling
    subset - this asserts both that the real score comes back correctly
    AND that the sandbox call was actually wired the right way (100% train,
    holdout mounted in place of test.jsonl).
    """
    pipeline = EvalPipeline(CONFIG)
    holdout_truth = {"10": 1, "11": 0}
    sandbox = _FakeHoldoutSandbox(preds=[{"id": 10, "pred": 1}, {"id": 11, "pred": 1}])

    result = run_holdout_evaluation(
        pipeline, sandbox, script_path="/fake/candidate_script.py",
        train_path="/fake/train.jsonl", holdout_path="/fake/holdout.jsonl", holdout_truth=holdout_truth,
    )

    assert result["holdout_skipped"] is False
    assert result["holdout_score"] == 0.5  # 1 of 2 correct
    assert sandbox.calls[0]["train_path_override"] == "/fake/train.jsonl"
    assert sandbox.calls[0]["test_path_override"] == "/fake/holdout.jsonl"
    assert sandbox.calls[0]["env_vars"]["SUBSET_PERCENTAGE"] == "100"


def test_run_holdout_evaluation_is_skipped_gracefully_when_holdout_is_unavailable():
    """
    A bring-your-own custom dataset that predates the sealed holdout (or
    was set up without one) must not crash or fail the candidate - it
    simply reports that holdout scoring was skipped, never touching the
    sandbox at all.
    """
    pipeline = EvalPipeline(CONFIG)
    sandbox = _FakeHoldoutSandbox(preds=[])

    result = run_holdout_evaluation(
        pipeline, sandbox, script_path="/fake/candidate_script.py",
        train_path="/fake/train.jsonl", holdout_path=None, holdout_truth=None,
    )

    assert result == {"holdout_score": 0.0, "holdout_error": "", "holdout_skipped": True}
    assert sandbox.calls == []


def test_run_holdout_evaluation_never_gates_on_a_failed_holdout_execution():
    """A holdout execution that crashes reports holdout_score=0.0 with a reason - it must never raise."""
    pipeline = EvalPipeline(CONFIG)

    class _FailingSandbox:
        def run_candidate(self, script_path, **kwargs):
            return {"exit_code": 1, "stdout": "", "stderr": "boom", "execution_time": 0.01, "timeout": False}

    result = run_holdout_evaluation(
        pipeline, _FailingSandbox(), script_path="/fake/candidate_script.py",
        train_path="/fake/train.jsonl", holdout_path="/fake/holdout.jsonl", holdout_truth={"0": 1},
    )

    assert result["holdout_skipped"] is False
    assert result["holdout_score"] == 0.0
    assert result["holdout_error"]
