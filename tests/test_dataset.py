import os
import tempfile

from eval.dataset import generate_dataset, load_dataset, load_subset


def test_generate_and_load_dataset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dataset.jsonl")
        generate_dataset(path, n_samples=200, seed=1)
        rows = load_dataset(path)
        assert len(rows) == 200
        assert set(rows[0].keys()) == {"x1", "x2", "label"}


def test_generate_dataset_is_deterministic():
    with tempfile.TemporaryDirectory() as d:
        path_a = os.path.join(d, "a.jsonl")
        path_b = os.path.join(d, "b.jsonl")
        generate_dataset(path_a, n_samples=50, seed=7)
        generate_dataset(path_b, n_samples=50, seed=7)
        assert load_dataset(path_a) == load_dataset(path_b)


def test_generate_dataset_skips_if_exists():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dataset.jsonl")
        generate_dataset(path, n_samples=10, seed=1)
        with open(path, "a") as f:
            f.write('{"sentinel": true}\n')

        generate_dataset(path, n_samples=999, seed=2)

        with open(path) as f:
            assert '"sentinel"' in f.read()


def test_subsets_are_nested():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dataset.jsonl")
        generate_dataset(path, n_samples=200, seed=3)
        small = load_subset(path, 5, seed=3)
        big = load_subset(path, 20, seed=3)

        small_keys = {(r["x1"], r["x2"]) for r in small}
        big_keys = {(r["x1"], r["x2"]) for r in big}
        assert small_keys.issubset(big_keys)
