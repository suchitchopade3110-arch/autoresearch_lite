from eval.pipeline import EvalPipeline

CONFIG = {"stages": [
    {"subset_percentage": 1, "threshold": 0.5},
    {"subset_percentage": 100, "threshold": 0.8},
]}


def make_result(score=None, exit_code=0, timeout=False):
    stdout = f"SCORE: {score}\n" if score is not None else "no score printed\n"
    return {"exit_code": exit_code, "stdout": stdout, "stderr": "", "execution_time": 0.1, "timeout": timeout}


def test_evaluate_stage_parses_real_score():
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(score=0.9), 100, 0.8)
    assert success
    assert score == 0.9


def test_evaluate_stage_fails_below_threshold():
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(score=0.3), 1, 0.5)
    assert not success
    assert score == 0.3


def test_evaluate_stage_missing_score_line_scores_zero():
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(score=None), 1, 0.5)
    assert not success
    assert score == 0.0


def test_evaluate_stage_nonzero_exit_fails():
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(score=0.99, exit_code=1), 1, 0.5)
    assert not success
    assert score == 0.0


def test_evaluate_stage_timeout_fails():
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(score=0.99, timeout=True), 1, 0.5)
    assert not success
    assert score == 0.0


def test_evaluate_reuses_single_result_across_all_stages():
    """evaluate() is the single-execution-result fallback; real per-stage runs should call evaluate_stage directly."""
    pipeline = EvalPipeline(CONFIG)
    success, final_score = pipeline.evaluate(make_result(score=0.9))
    assert success
    assert final_score == 0.9
