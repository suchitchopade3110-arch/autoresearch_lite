import json
import os
import random
from typing import Dict, List

DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 1000


def generate_dataset(path: str, n_samples: int = DEFAULT_N_SAMPLES, seed: int = DEFAULT_SEED) -> None:
    """
    Writes a deterministic synthetic binary-classification dataset to `path`
    as JSONL, unless a file already exists there.

    The generative rule is a noisy linear boundary (label = 1 if
    2*x1 - x2 + noise > 0), so a candidate that recovers weights close to
    (2, -1, 0) scores well and one that doesn't scores poorly - real
    signal for the eval pipeline to gate on, without depending on any
    external dataset.
    """
    if os.path.exists(path):
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(n_samples):
            x1 = rng.uniform(-1, 1)
            x2 = rng.uniform(-1, 1)
            noise = rng.uniform(-0.3, 0.3)
            label = 1 if (2 * x1 - x2 + noise) > 0 else 0
            f.write(json.dumps({"x1": x1, "x2": x2, "label": label}) + "\n")


def load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def subset_indices(n: int, percentage: float, seed: int = DEFAULT_SEED) -> List[int]:
    """
    Deterministic index selection such that subsets at smaller percentages
    are always contained within subsets at larger percentages - so
    progressive-scaling stages see nested, not independently-sampled, data.
    """
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    k = max(1, int(n * percentage / 100))
    return order[:k]


def load_subset(path: str, percentage: float, seed: int = DEFAULT_SEED) -> List[Dict]:
    rows = load_dataset(path)
    idx = subset_indices(len(rows), percentage, seed)
    return [rows[i] for i in idx]
