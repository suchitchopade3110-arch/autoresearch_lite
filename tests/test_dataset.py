import os
import tempfile

from eval.dataset import generate_split, load_dataset, load_subset, load_truth, write_subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_generate_split_produces_train_test_and_truth():
    with tempfile.TemporaryDirectory() as d:
        paths = generate_split(d, n=200, seed=1)
        train_rows = load_dataset(paths["train"])
        test_rows = load_dataset(paths["test"])
        truth = load_truth(paths["truth"])

        assert set(train_rows[0].keys()) == {"x1", "x2", "label"}
        assert set(test_rows[0].keys()) == {"id", "x1", "x2"}
        assert "label" not in test_rows[0]
        assert len(test_rows) == len(truth)
        assert len(train_rows) + len(test_rows) == 200
        assert set(truth.keys()) == {str(r["id"]) for r in test_rows}


def test_generate_split_is_deterministic():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        paths_a = generate_split(a, n=50, seed=7)
        paths_b = generate_split(b, n=50, seed=7)
        assert load_dataset(paths_a["train"]) == load_dataset(paths_b["train"])
        assert load_dataset(paths_a["test"]) == load_dataset(paths_b["test"])
        assert load_truth(paths_a["truth"]) == load_truth(paths_b["truth"])


def test_generate_split_skips_if_exists():
    with tempfile.TemporaryDirectory() as d:
        generate_split(d, n=10, seed=1)
        train_path = os.path.join(d, "train.jsonl")
        with open(train_path, "a") as f:
            f.write('{"sentinel": true}\n')

        generate_split(d, n=999, seed=2)

        with open(train_path) as f:
            assert '"sentinel"' in f.read()


def test_subsets_are_nested():
    with tempfile.TemporaryDirectory() as d:
        paths = generate_split(d, n=200, seed=3)
        small = load_subset(paths["train"], 5, seed=3)
        big = load_subset(paths["train"], 20, seed=3)

        small_keys = {(r["x1"], r["x2"]) for r in small}
        big_keys = {(r["x1"], r["x2"]) for r in big}
        assert small_keys.issubset(big_keys)


def test_write_subset_writes_only_the_host_selected_rows():
    """
    Council audit finding: progressive-scaling stages used to mount the
    FULL train.jsonl at every stage regardless of SUBSET_PERCENTAGE - a
    candidate that simply ignored the env var could train on 100% of the
    data at the "1%" stage. write_subset is the host-side enforcement: the
    file it writes must physically contain only the selected rows, not the
    full dataset with a hint attached.
    """
    with tempfile.TemporaryDirectory() as d:
        paths = generate_split(d, n=200, seed=11)
        dst = os.path.join(d, "train_subset.jsonl")

        write_subset(paths["train"], dst, 10, seed=11)

        full_rows = load_dataset(paths["train"])
        subset_rows = load_dataset(dst)
        expected = load_subset(paths["train"], 10, seed=11)

        assert len(subset_rows) < len(full_rows)
        assert subset_rows == expected


def test_omitted_seed_is_not_the_old_public_default_of_42():
    """
    Council audit critical finding: the old default seed (42) is a public
    constant in this module's own history - together with the (also public)
    generation code, that let a candidate regenerate every held-out label
    without ever touching truth.json. Leaving `seed` unset must now produce
    a dataset that does NOT match what the old hardcoded default would have
    produced.
    """
    with tempfile.TemporaryDirectory() as unset_dir, tempfile.TemporaryDirectory() as public_dir:
        unset_paths = generate_split(unset_dir, n=50)  # seed omitted
        public_paths = generate_split(public_dir, n=50, seed=42)  # the old default, explicitly

        # Astronomically unlikely to collide by chance with a random 63-bit seed.
        assert load_truth(unset_paths["truth"]) != load_truth(public_paths["truth"])


def test_omitted_seed_differs_across_separate_datasets():
    """Two independent runs that both omit dataset.seed must not share a seed."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        paths_a = generate_split(a, n=50)
        paths_b = generate_split(b, n=50)
        assert load_truth(paths_a["truth"]) != load_truth(paths_b["truth"])


def test_omitted_seed_is_recorded_host_side_but_never_in_a_mounted_file():
    """
    The secret seed is written for the operator's own reference, but only
    to a path sandbox/executor.py never mounts (it mounts train.jsonl/
    test.jsonl by explicit name only - see test_reward_hacking.py's
    equivalent invariant tests for truth.json).
    """
    with tempfile.TemporaryDirectory() as d:
        generate_split(d, n=20)
        seed_path = os.path.join(d, ".generator_seed")
        assert os.path.exists(seed_path)
        with open(seed_path) as f:
            recorded_seed = int(f.read().strip())
        assert recorded_seed > 0

    executor_path = os.path.join(REPO_ROOT, "sandbox", "executor.py")
    with open(executor_path) as f:
        assert "generator_seed" not in f.read()


def test_explicit_seed_still_supported_for_reproducible_fixtures():
    """Passing an explicit seed (for tests/CI) must still work exactly as before."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        paths_a = generate_split(a, n=50, seed=123)
        paths_b = generate_split(b, n=50, seed=123)
        assert load_truth(paths_a["truth"]) == load_truth(paths_b["truth"])
        assert not os.path.exists(os.path.join(a, ".generator_seed"))


def test_truth_values_are_bare_labels_not_features():
    """
    truth.json must only ever map id -> 0/1 label, never features - it's
    the answer key and is never mounted into the sandbox (see
    sandbox/executor.py and test_reward_hacking.py).
    """
    with tempfile.TemporaryDirectory() as d:
        paths = generate_split(d, n=50, seed=9)
        truth = load_truth(paths["truth"])
        assert len(truth) > 0
        assert all(v in (0, 1) for v in truth.values())
