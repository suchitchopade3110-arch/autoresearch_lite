import pytest
import os
import tempfile
from memory.db import ExperimentDB

@pytest.fixture
def test_db():
    with tempfile.TemporaryDirectory() as d:
        db = ExperimentDB(db_path=d)
        yield db

def test_db_roundtrip(test_db):
    rec_id = test_db.store_experiment(
        hypothesis="Make it faster",
        diff="--- a\n+++ b\n@@ -1 +1 @@\n-slow()\n+fast()",
        rationale="fast() is better",
        metrics={"speed": 10.0},
        outcome="success"
    )

    results = test_db.retrieve_experiments("faster", k=1)

    assert len(results) == 1
    res = results[0]
    assert res['hypothesis'] == "Make it faster"
    assert res['outcome'] == "success"
    assert res['metrics'] == {"speed": 10.0}

def test_db_semantic_relevance(test_db):
    test_db.store_experiment(
        hypothesis="Increase learning rate",
        diff="diff1",
        rationale="lr=0.01",
        metrics={},
        outcome="failure"
    )
    test_db.store_experiment(
        hypothesis="Use batch normalization",
        diff="diff2",
        rationale="reduce covariance shift",
        metrics={},
        outcome="success"
    )

    results = test_db.retrieve_experiments("learning rate scheduling", k=1)
    assert len(results) == 1
    assert results[0]['hypothesis'] == "Increase learning rate"
