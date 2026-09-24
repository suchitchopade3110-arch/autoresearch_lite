from evolution.scoring import score_candidates


def make_candidate(id_, success, final_score, execution_speed, execution_time, eval_passed=None):
    # eval_passed defaults to mirroring `success` when not given explicitly -
    # convenient for tests that don't care about the merged/held distinction,
    # but test_clamp_keys_on_eval_passed_not_merged below sets it explicitly
    # to prove the two are no longer conflated.
    return {
        'id': id_,
        'success': success,
        'eval_passed': success if eval_passed is None else eval_passed,
        'final_score': final_score,
        'metrics': {'execution_speed': execution_speed, 'execution_time': execution_time},
    }


def test_pareto_never_ranks_a_failure_above_a_success():
    """
    Regression test for the bug where a candidate that crashed almost
    instantly (very low execution_time -> very high execution_speed) could
    outrank a candidate that actually succeeded, because validation_accuracy
    was never wired to the real eval result and defaulted to 0.0 for
    everyone.
    """
    success = make_candidate('success', True, final_score=0.95, execution_speed=0.5, execution_time=2.0)
    fast_failure = make_candidate('fast_failure', False, final_score=0.0, execution_speed=20.0, execution_time=0.05)

    scored = score_candidates([success, fast_failure], {'scoring_strategy': 'pareto'})
    winner = max(scored, key=lambda c: c['composite_score'])
    assert winner['id'] == 'success'


def test_weighted_never_ranks_a_failure_above_a_success():
    success = make_candidate('success', True, final_score=0.95, execution_speed=0.5, execution_time=2.0)
    fast_failure = make_candidate('fast_failure', False, final_score=0.0, execution_speed=20.0, execution_time=0.05)

    scored = score_candidates([success, fast_failure], {'scoring_strategy': 'weighted'})
    winner = max(scored, key=lambda c: c['composite_score'])
    assert winner['id'] == 'success'


def test_weighted_ranks_successes_by_actual_accuracy():
    good = make_candidate('good', True, final_score=0.95, execution_speed=1.0, execution_time=1.0)
    mediocre = make_candidate('mediocre', True, final_score=0.6, execution_speed=1.0, execution_time=1.0)

    scored = score_candidates([good, mediocre], {'scoring_strategy': 'weighted'})
    winner = max(scored, key=lambda c: c['composite_score'])
    assert winner['id'] == 'good'


def test_relative_order_preserved_within_an_all_failure_generation():
    less_bad = make_candidate('less_bad', False, final_score=0.3, execution_speed=1.0, execution_time=1.0)
    worse = make_candidate('worse', False, final_score=0.1, execution_speed=1.0, execution_time=1.0)

    scored = score_candidates([less_bad, worse], {'scoring_strategy': 'weighted'})
    winner = max(scored, key=lambda c: c['composite_score'])
    assert winner['id'] == 'less_bad'


def test_unknown_strategy_raises():
    candidate = make_candidate('a', True, final_score=0.5, execution_speed=1.0, execution_time=1.0)
    try:
        score_candidates([candidate], {'scoring_strategy': 'nonsense'})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_clamp_keys_on_eval_passed_not_merged():
    """
    Council audit finding: "failed never outranks successful" used to key
    on `success` (== actually merged), which is False for EVERY candidate
    in a generation where nothing merges - common with a human approval
    gate, since candidates sit "held" pending review rather than merging
    immediately. With nothing to compare against `success=True`, the clamp
    was a no-op: a near-instant crash (very low execution_time -> very high
    execution_speed) could outrank a candidate that scored 0.95+ on real
    held-out data and was simply awaiting human approval, and go on to be
    selected as a parent/elite. Keying on eval_passed instead means the
    clamp still separates them even though NEITHER has merged yet.
    """
    held_but_passed = make_candidate(
        'held', success=False, final_score=0.95, execution_speed=0.5, execution_time=2.0, eval_passed=True,
    )
    crashed = make_candidate(
        'crashed', success=False, final_score=0.0, execution_speed=20.0, execution_time=0.05, eval_passed=False,
    )

    scored = score_candidates([held_but_passed, crashed], {'scoring_strategy': 'weighted'})
    winner = max(scored, key=lambda c: c['composite_score'])
    assert winner['id'] == 'held'
