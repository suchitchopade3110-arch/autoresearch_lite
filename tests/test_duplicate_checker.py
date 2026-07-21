from evolution.duplicate_checker import is_duplicate
from memory.db import ExperimentDB

DIFF_A = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-foo()\n+bar()"
NEAR_DUP_OF_A = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-foo()\n+bar( )"
UNRELATED_DIFF = "--- a/x\n+++ b/x\n@@ -1 +1,50 @@\n-completely different multi line change unrelated to the above"


def test_empty_diff_is_never_a_duplicate(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    assert is_duplicate("", db) is False
    assert is_duplicate("   ", db) is False


def test_no_prior_experiments_is_never_a_duplicate(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    assert is_duplicate(DIFF_A, db) is False


def test_exact_diff_match_is_a_duplicate(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="goal", diff=DIFF_A, rationale="r", metrics={}, outcome="success")
    assert is_duplicate(DIFF_A, db, hypothesis="goal") is True


def test_near_identical_diff_is_flagged_as_a_duplicate(tmp_dir):
    """
    Regression test: retrieve_experiments never returned a 'distance' field
    (duplicate_checker read closest.get('distance', inf), which was always
    inf), so this never fired regardless of threshold - only an exact
    string match ever counted as a duplicate. A one-character variant of an
    already-tried diff must be caught by the semantic-similarity check.
    """
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="goal", diff=DIFF_A, rationale="r", metrics={}, outcome="success")
    assert is_duplicate(NEAR_DUP_OF_A, db, threshold=0.25, hypothesis="goal") is True


def test_genuinely_different_diff_is_not_flagged(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="goal", diff=DIFF_A, rationale="r", metrics={}, outcome="success")
    assert is_duplicate(UNRELATED_DIFF, db, threshold=0.25, hypothesis="goal") is False


def test_retrieve_experiments_includes_distance(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="goal", diff=DIFF_A, rationale="r", metrics={}, outcome="success")
    results = db.retrieve_experiments(query=DIFF_A, k=1)
    assert 'distance' in results[0]
    assert isinstance(results[0]['distance'], float)
