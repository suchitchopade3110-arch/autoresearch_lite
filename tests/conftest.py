import hashlib
import math
import re
import shutil
import tempfile

import chromadb
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


_WORD_RE = re.compile(r"[a-z0-9]+")
_HASH_DIM = 256


class _HermeticHashingEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Deterministic, fully offline stand-in for chromadb's real
    DefaultEmbeddingFunction (memory/db.py's production default).

    DefaultEmbeddingFunction delegates to ONNXMiniLM_L6_V2, which downloads
    a ~90MB model from the network on first use and caches it under
    ~/.cache/chroma - on any machine without that cache already warm (a
    fresh CI runner, a fresh clone, an offline sandbox), every Chroma-backed
    test either fails outright or becomes network-speed-dependent. That
    makes the whole suite non-hermetic: a test's pass/fail should depend
    only on the code under test, not on what happens to already be sitting
    in a cache directory.

    This is a fixed-dimension bag-of-words hash: the same input text always
    produces the same vector, using nothing but the standard library. It
    preserves plain word-overlap similarity (two texts that share more
    words score closer under cosine distance), which is all the existing
    semantic-relevance and near-duplicate assertions actually rely on.
    """

    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            vec = [0.0] * _HASH_DIM
            for word in _WORD_RE.findall(text.lower()):
                idx = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % _HASH_DIM
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def name(self):
        return "hermetic-hashing-embedding"

    def get_config(self):
        return {}


@pytest.fixture(autouse=True)
def hermetic_chroma_embeddings(monkeypatch):
    """
    Every ExperimentDB constructed anywhere in the test suite must embed
    with _HermeticHashingEmbeddingFunction, not the real network-backed
    default - see its docstring. memory/db.py obtains the default via
    `embedding_functions.DefaultEmbeddingFunction()`, so patching that one
    factory here covers every test/fixture that builds an ExperimentDB,
    with no change to production code or to individual test call sites.
    """
    monkeypatch.setattr(
        "chromadb.utils.embedding_functions.DefaultEmbeddingFunction",
        _HermeticHashingEmbeddingFunction,
    )
