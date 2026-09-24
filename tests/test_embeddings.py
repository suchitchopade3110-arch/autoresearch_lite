import os
from unittest.mock import patch

from memory.embeddings import HashingEmbeddingFunction


def _as_plain_lists(embeddings):
    # chromadb's EmbeddingFunction.__init_subclass__ wraps __call__ to run
    # the result through normalize_embeddings(), which returns numpy
    # float32 arrays - convert back to plain Python floats/lists so
    # ordinary list/float comparisons work in these tests.
    return [[float(x) for x in row] for row in embeddings]


def test_hashing_embedding_is_deterministic_across_instances():
    """Same text must embed identically every time - required for reproducible tests."""
    a = _as_plain_lists(HashingEmbeddingFunction()(["Hypothesis: goal\nDiff:\n-foo()\n+bar()"]))
    b = _as_plain_lists(HashingEmbeddingFunction()(["Hypothesis: goal\nDiff:\n-foo()\n+bar()"]))
    assert a == b


def test_hashing_embedding_is_a_unit_vector():
    [vec] = _as_plain_lists(HashingEmbeddingFunction()(["some arbitrary text with several words in it"]))
    norm_sq = sum(v * v for v in vec)
    # float32 round-trip precision (chromadb's wrapper casts to float32),
    # not the tighter tolerance a pure float64 computation would allow.
    assert abs(norm_sq - 1.0) < 1e-5


def test_hashing_embedding_empty_text_does_not_divide_by_zero():
    [vec] = _as_plain_lists(HashingEmbeddingFunction()([""]))
    assert all(v == 0.0 for v in vec)


def test_hashing_embedding_near_identical_text_is_closer_than_vocabulary_disjoint_text():
    """
    The one property every test in this suite that depends on embedding
    behavior actually needs: near-identical text must cosine-score
    distinctly closer than text sharing no vocabulary at all (see
    evolution/duplicate_checker.py's threshold-based near-duplicate check).
    """
    ef = HashingEmbeddingFunction()
    base = "Hypothesis: goal\nDiff:\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-foo()\n+bar()"
    near = "Hypothesis: goal\nDiff:\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-foo()\n+bar( )"
    disjoint = "zzyzx quux wobble frobnicate xyzzy plugh"

    e_base, e_near, e_disjoint = _as_plain_lists(ef([base, near, disjoint]))

    def cosine_distance(a, b):
        return 1 - sum(x * y for x, y in zip(a, b))

    near_distance = cosine_distance(e_base, e_near)
    disjoint_distance = cosine_distance(e_base, e_disjoint)
    assert near_distance < 0.05
    assert disjoint_distance > 0.9
    assert near_distance < disjoint_distance


def test_experiment_db_uses_the_hashing_stub_when_hermetic_env_var_is_set(tmp_dir):
    """
    tests/conftest.py sets AUTORESEARCH_HERMETIC_EMBEDDINGS=1 for the whole
    suite - this is the mechanism itself, verified in isolation.
    """
    assert os.environ.get("AUTORESEARCH_HERMETIC_EMBEDDINGS") == "1"
    from memory.db import ExperimentDB

    db = ExperimentDB(db_path=tmp_dir)
    assert isinstance(db.ef, HashingEmbeddingFunction)


def test_experiment_db_uses_the_real_default_embedder_when_hermetic_env_var_is_unset(tmp_dir):
    """
    A real run (the env var never set outside tests/conftest.py) must still
    get chromadb's real DefaultEmbeddingFunction, not silently downgrade to
    the lexical-only test stub - hermeticity is a test-only concern, never
    a production behavior change. Patches DefaultEmbeddingFunction itself
    so this doesn't require network access to prove.
    """
    import memory.db as db_module

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AUTORESEARCH_HERMETIC_EMBEDDINGS", None)
        with patch.object(db_module.embedding_functions, "DefaultEmbeddingFunction") as mock_default:
            # A real (if different) embedding function stands in for "the
            # DefaultEmbeddingFunction() constructor's return value" - the
            # point is proving THAT constructor gets called and its result
            # becomes self.ef, not that the sentinel itself is inert.
            sentinel = HashingEmbeddingFunction()
            mock_default.return_value = sentinel
            db = db_module.ExperimentDB(db_path=tmp_dir)
            assert db.ef is sentinel
            mock_default.assert_called_once()


def test_explicit_embedding_function_always_wins_over_the_env_var(tmp_dir):
    from memory.db import ExperimentDB

    custom = HashingEmbeddingFunction()
    db = ExperimentDB(db_path=tmp_dir, embedding_function=custom)
    assert db.ef is custom
