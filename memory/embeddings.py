import hashlib
import math
import re
from typing import Any, Dict, List

from chromadb.api.types import Documents, EmbeddingFunction

_TOKEN_RE = re.compile(r"\w+")


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A tiny, dependency-free, fully deterministic embedding function - NOT
    for production use, whose retrieval quality genuinely benefits from a
    real sentence embedding. Exists purely so the test suite never needs
    chromadb's DefaultEmbeddingFunction, which lazily downloads an ONNX
    model from HuggingFace on first use: several audits of this repo
    (see REMEDIATION.md's Wave 5) found the "full suite green (non-Docker)"
    claim didn't actually hold offline, because ~25 ExperimentDB-backed
    tests silently depended on that download having already happened (or
    network being available) in whatever environment ran them.

    Implementation: a normalized hashed bag-of-words vector (the
    "hashing trick" - each lowercased word token is bucketed via a
    deterministic SHA-256-derived index, never Python's own str hash,
    which is randomized per-process unless PYTHONHASHSEED is pinned).
    This is deliberately not a real embedding - it captures lexical
    overlap, not semantics - but every test in this suite that depends on
    embedding behavior at all (near-duplicate diffs scoring a small cosine
    distance, genuinely unrelated text scoring a larger one, topically
    overlapping text retrieving as more relevant) only ever exercises
    lexical similarity in the first place, which this reproduces exactly
    as well as it needs to.
    """

    _DIM = 256
    _NAME = "hashing-bag-of-words-test-stub"

    def __init__(self) -> None:
        pass  # no state to initialize - overridden only to skip the base class's DeprecationWarning

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self._DIM
        for token in _TOKEN_RE.findall(text.lower()):
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self._DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # The remaining methods implement chromadb's EmbeddingFunction protocol
    # fully (not just __call__), so collection metadata bookkeeping and
    # is_legacy() checks work cleanly instead of falling back to a
    # deprecated code path with its own warnings.
    @staticmethod
    def name() -> str:
        return HashingEmbeddingFunction._NAME

    def get_config(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction()
