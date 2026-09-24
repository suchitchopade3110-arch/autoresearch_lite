import os
import shutil
import tempfile

import pytest

# Makes every ExperimentDB() constructed anywhere in this test suite use
# memory/embeddings.py's dependency-free HashingEmbeddingFunction instead
# of chromadb's real DefaultEmbeddingFunction (which lazily downloads an
# ONNX model from HuggingFace on first use) - set here, once, before any
# test module imports memory.db, rather than threading an explicit
# embedding_function= kwarg through every one of the ~30 ExperimentDB(...)
# call sites across the suite. Never set outside tests/conftest.py: a real
# run must always get the real embedding function.
os.environ["AUTORESEARCH_HERMETIC_EMBEDDINGS"] = "1"


@pytest.fixture
def tmp_dir():
    """
    Like a plain tempfile.TemporaryDirectory(), but cleanup tolerates files
    still locked by a not-yet-released handle instead of failing the test -
    chromadb's sqlite file in particular doesn't reliably release its
    Windows file lock by the time a test tears down, even after the
    ExperimentDB object goes out of scope.
    """
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
