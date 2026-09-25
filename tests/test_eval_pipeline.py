import json
import os

import pytest

from eval.pipeline import EvalPipeline

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
    os.symlink(real_path, symlink_path)

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
