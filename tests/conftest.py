import shutil
import tempfile

import pytest


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
