from evolution.scoring import score_candidates


def make_candidate(id_, success, final_score, execution_speed, execution_time):
    return {
        'id': id_,
        'success': success,
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
