import json
import os
import random
import secrets
from typing import Dict, List, Optional

# Used only by subset_indices()/load_subset() to pick which already-visible
# TRAINING rows a progressive-scaling stage sees - a public, non-secret
# constant is fine there, since it never affects the held-out labels. It is
# NOT used by generate_split(), whose seed is security-sensitive - see
# generate_split's docstring.
DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 1000
DEFAULT_TEST_FRAC = 0.25
# Fraction of the TEST split (not of the whole dataset) carved out as the
# sealed holdout - see generate_split's docstring for why a second held-out
# set exists at all. 0.5 of test_frac means, at the defaults, an equal-sized
# selection set and holdout set.
DEFAULT_HOLDOUT_FRAC = 0.5


def _generate_rows(n_samples: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        x1 = rng.uniform(-1, 1)
        x2 = rng.uniform(-1, 1)
        noise = rng.uniform(-0.3, 0.3)
        label = 1 if (2 * x1 - x2 + noise) > 0 else 0
        rows.append({"x1": x1, "x2": x2, "label": label})
    return rows


def generate_split(dir_path: str, n: int = DEFAULT_N_SAMPLES, seed: Optional[int] = None,
                    test_frac: float = DEFAULT_TEST_FRAC,
                    holdout_frac: float = DEFAULT_HOLDOUT_FRAC) -> Dict[str, str]:
    """
    Writes a deterministic synthetic binary-classification dataset to
    `dir_path`, split so the ground truth for the held-out sets never
    reaches the sandbox:

      - train.jsonl:   {"x1","x2","label"} - mounted read-only into the sandbox
      - test.jsonl:    {"id","x1","x2"} (no label) - the SELECTION set,
                       mounted read-only, scored against every progressive
                       stage and used for baseline/merge gating
      - truth.json:    {id: label} for test.jsonl - stays on the host, never mounted
      - holdout.jsonl: {"id","x1","x2"} (no label) - the SEALED holdout,
                       mounted read-only ONLY for the one-time holdout
                       evaluation after a candidate has already cleared
                       every stage (see eval/pipeline.py's evaluate_stage
                       callers) - never seen during progressive scaling
      - holdout_truth.json: {id: label} for holdout.jsonl - stays on the
                       host, never mounted

    A candidate that only ever sees train.jsonl/test.jsonl/holdout.jsonl
    cannot read either held-out label set off disk, so a prediction score
    against either truth file reflects real generalization rather than a
    printed claim.

    WHY TWO HELD-OUT SETS, NOT ONE: every progressive-scaling stage and the
    baseline gate (eval/baseline.py) score a candidate against the SAME
    test.jsonl every single time a run happens - across many candidates and
    generations, that repeatedly re-uses the same finite sample to decide
    "did this genuinely improve." A merge rule of "beat the best score ever
    achieved on this exact sample" then ratchets upward on that sample's
    own sampling noise as much as on real improvement (classic multiple-
    comparisons/overfitting-to-the-validation-set risk, sometimes called
    "peeking"). holdout.jsonl is carved out separately, never used to
    accept or reject anything during the loop, and scored only once a
    candidate has already passed on test.jsonl - so its score is a much
    more honest (if still not perfectly independent across many
    candidates) estimate of real generalization, safe to show a human
    reviewer or an auditor alongside the selection score without it having
    been used to buy the merge. See orchestrator/run.py's _score_holdout
    and evolution/scheduler.py's equivalent call for where it's actually
    scored and how it's kept out of the merge decision itself.

    Skips regeneration if train.jsonl already exists, same as the old
    generate_dataset's no-clobber contract.

    SECURITY: `seed` fully determines both the generated labels and the
    train/test split - this module's own generation code is ordinary,
    readable Python, so anyone who also knows `seed` can call
    `_generate_rows`/this same shuffle themselves and reconstruct every
    held-out label without ever touching truth.json. That used to be
    exactly as bad as it sounds: the old default was the literal constant
    42, which is public in this file's own history. Leaving `seed` unset
    (the default, and the only setting recommended for a real run) now
    generates a fresh, cryptographically random seed via `secrets.randbits`
    and uses it once; it is written for the operator's own reference to
    `<dir_path>/.generator_seed` - a path that is NEVER mounted into the
    sandbox (sandbox/executor.py mounts train.jsonl/test.jsonl by explicit
    name, never dir_path itself - the same reason truth.json itself stays
    hidden, see eval/pipeline.py's top-of-file invariant). Brute-forcing a
    63-bit seed by replaying this module against a mounted train.jsonl row
    is computationally infeasible inside the sandbox's own CPU/time limits.
    Pass an explicit `seed` only for a reproducible test fixture or CI
    dataset that was never meant to resist this attack in the first place -
    doing so in a real deployment reintroduces exactly the leak this
    default closes.
    """
    train_path = os.path.join(dir_path, "train.jsonl")
    test_path = os.path.join(dir_path, "test.jsonl")
    truth_path = os.path.join(dir_path, "truth.json")
    holdout_path = os.path.join(dir_path, "holdout.jsonl")
    holdout_truth_path = os.path.join(dir_path, "holdout_truth.json")

    if os.path.exists(train_path):
        return {
            "train": train_path, "test": test_path, "truth": truth_path,
            "holdout": holdout_path, "holdout_truth": holdout_truth_path,
        }

    os.makedirs(dir_path, exist_ok=True)

    if seed is None:
        seed = secrets.randbits(63)
        with open(os.path.join(dir_path, ".generator_seed"), "w") as f:
            f.write(f"{seed}\n")

    rows = _generate_rows(n, seed)
    n_held_out = max(2, int(len(rows) * test_frac))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    held_out_order = order[:n_held_out]
    # The held-out portion is split again, selection (test.jsonl) vs. sealed
    # holdout - see generate_split's docstring for why. holdout_frac is a
    # fraction OF the held-out portion, not of the whole dataset.
    n_holdout = max(1, min(len(held_out_order) - 1, int(len(held_out_order) * holdout_frac)))
    holdout_idx = set(held_out_order[:n_holdout])
    test_idx = set(held_out_order[n_holdout:])

    truth = {}
    holdout_truth = {}
    with open(train_path, "w") as train_f, open(test_path, "w") as test_f, open(holdout_path, "w") as holdout_f:
        for i, row in enumerate(rows):
            if i in test_idx:
                test_f.write(json.dumps({"id": i, "x1": row["x1"], "x2": row["x2"]}) + "\n")
                truth[str(i)] = row["label"]
            elif i in holdout_idx:
                holdout_f.write(json.dumps({"id": i, "x1": row["x1"], "x2": row["x2"]}) + "\n")
                holdout_truth[str(i)] = row["label"]
            else:
                train_f.write(json.dumps(row) + "\n")

    with open(truth_path, "w") as f:
        json.dump(truth, f)
    with open(holdout_truth_path, "w") as f:
        json.dump(holdout_truth, f)

    return {
        "train": train_path, "test": test_path, "truth": truth_path,
        "holdout": holdout_path, "holdout_truth": holdout_truth_path,
    }


class DatasetError(Exception):
    """Raised when dataset.mode is misconfigured, or 'custom' mode is set but the expected files aren't there."""


def validate_custom_dataset(dir_path: str) -> Dict[str, str]:
    """
    dataset.mode: custom - the operator supplies their own train.jsonl/
    test.jsonl/truth.json at dir_path (already in whatever row shape their
    own candidate task and eval.scorer expect) instead of the built-in
    synthetic linear-boundary generator. Checks the three files exist,
    that train.jsonl/test.jsonl parse as JSONL (one JSON object per
    non-blank line), that truth.json parses as a JSON object, and that
    test.jsonl's rows carry an "id" field whose values are exactly
    truth.json's key set. Full row schema beyond "id" (what a "pred"/label
    value actually looks like) is still the operator's own eval.scorer's
    business, not this module's - deliberately not validated here.

    The id-set check exists because its failure mode was previously silent
    and expensive: a mismatched custom dataset would validate cleanly here,
    then have every single candidate fail eval.pipeline.binary_accuracy_scorer
    (or a custom scorer with the same expectation) with "prediction id set
    mismatch" - discovered only after burning a full sandbox run, and
    plausibly misread as "the candidate is bad" rather than "the dataset
    files don't agree with each other." Catching it here fails fast, before
    any sandbox execution, with a message that names the actual mismatch.

    holdout.jsonl/holdout_truth.json (see generate_split's docstring for
    why a second held-out set exists) are OPTIONAL here, for backward
    compatibility with a custom dataset directory set up before the sealed
    holdout existed: if both are present, they're validated the same way as
    test.jsonl/truth.json; if neither is present, the one-time holdout
    evaluation step is silently skipped for this dataset (a warning is
    logged where it's actually skipped - see orchestrator/run.py's
    _score_holdout). A partial pair (one file present, the other missing)
    is always an error, since that can only be an incomplete setup, never
    an intentional choice.

    The security invariant that matters - truth.json/holdout_truth.json
    never reaching the sandbox - is enforced by sandbox/executor.py
    mounting train.jsonl/test.jsonl/holdout.jsonl by explicit name only,
    never the whole directory. That holds regardless of whether these
    files were generated by generate_split() or dropped in by hand, so
    nothing here needs to special-case it.
    """
    train_path = os.path.join(dir_path, "train.jsonl")
    test_path = os.path.join(dir_path, "test.jsonl")
    truth_path = os.path.join(dir_path, "truth.json")
    holdout_path = os.path.join(dir_path, "holdout.jsonl")
    holdout_truth_path = os.path.join(dir_path, "holdout_truth.json")

    missing = [p for p in (train_path, test_path, truth_path) if not os.path.exists(p)]
    if missing:
        raise DatasetError(
            "dataset.mode: custom requires train.jsonl, test.jsonl, and truth.json to already "
            f"exist at {dir_path!r} - missing: {', '.join(os.path.basename(p) for p in missing)}"
        )

    holdout_exists = os.path.exists(holdout_path)
    holdout_truth_exists = os.path.exists(holdout_truth_path)
    if holdout_exists != holdout_truth_exists:
        present = "holdout.jsonl" if holdout_exists else "holdout_truth.json"
        absent = "holdout_truth.json" if holdout_exists else "holdout.jsonl"
        raise DatasetError(
            f"{dir_path!r} has {present} but not {absent} - the sealed holdout is optional, but "
            "an incomplete pair is always an error; either provide both or remove the one present."
        )

    _load_and_check_id_sets(train_path, test_path, truth_path)
    result = {"train": train_path, "test": test_path, "truth": truth_path,
              "holdout": None, "holdout_truth": None}
    if holdout_exists:
        _load_and_check_id_sets(train_path, holdout_path, holdout_truth_path, features_path_is_train=False)
        result["holdout"] = holdout_path
        result["holdout_truth"] = holdout_truth_path
    return result


def _load_jsonl_rows(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno} is not valid JSON: {e}")
    return rows


def _load_and_check_id_sets(train_path: str, features_path: str, truth_path: str,
                             features_path_is_train: bool = True) -> None:
    """
    Shared validation for a (features-file, truth-file) held-out pair -
    used for test.jsonl/truth.json and, if present, holdout.jsonl/
    holdout_truth.json. train_path is only re-checked (parses, non-empty)
    the first time this is called for a given dataset; features_path_is_
    train exists purely to avoid redundantly re-validating train.jsonl a
    second time when checking the holdout pair.
    """
    try:
        with open(truth_path) as f:
            truth = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetError(f"{truth_path} is not valid JSON: {e}")
    if not isinstance(truth, dict):
        raise DatasetError(f"{truth_path} must be a JSON object of {{id: label}}, got {type(truth).__name__}")

    if features_path_is_train:
        train_rows = _load_jsonl_rows(train_path)
        if not train_rows:
            raise DatasetError(f"{train_path} contains no rows")

    feature_rows = _load_jsonl_rows(features_path)
    if not feature_rows:
        raise DatasetError(f"{features_path} contains no rows")

    missing_id_rows = [i for i, row in enumerate(feature_rows) if "id" not in row]
    if missing_id_rows:
        raise DatasetError(
            f"{features_path}: {len(missing_id_rows)} row(s) have no \"id\" field "
            f"(e.g. row {missing_id_rows[0]}) - eval scoring matches predictions to truth by id"
        )

    feature_ids = {str(row["id"]) for row in feature_rows}
    truth_ids = {str(k) for k in truth.keys()}
    if feature_ids != truth_ids:
        only_in_features = sorted(feature_ids - truth_ids)[:5]
        only_in_truth = sorted(truth_ids - feature_ids)[:5]
        detail = []
        if only_in_features:
            detail.append(f"ids in {os.path.basename(features_path)} but not {os.path.basename(truth_path)} (e.g. {only_in_features})")
        if only_in_truth:
            detail.append(f"ids in {os.path.basename(truth_path)} but not {os.path.basename(features_path)} (e.g. {only_in_truth})")
        raise DatasetError(
            f"{features_path} and {truth_path} disagree on which ids exist - "
            + "; ".join(detail) +
            ". Every candidate would fail scoring with a silent id-set mismatch; fix the dataset files."
        )


def resolve_dataset(dir_path: str, mode: str = "synthetic", **generate_split_kwargs) -> Dict[str, str]:
    """
    Single entry point orchestrator/run.py calls to get train/test/truth/
    holdout/holdout_truth paths - dispatches to the built-in synthetic
    generator (mode: "synthetic", the default, fully backward compatible)
    or to a validated bring-your-own dataset (mode: "custom").
    generate_split_kwargs (n/seed/test_frac/holdout_frac) are only
    meaningful for "synthetic" and are ignored for "custom", where holdout/
    holdout_truth are None unless the operator's own directory happens to
    contain both files (see validate_custom_dataset).
    """
    if mode == "custom":
        return validate_custom_dataset(dir_path)
    if mode != "synthetic":
        raise DatasetError(f"dataset.mode must be 'synthetic' or 'custom', got {mode!r}")
    return generate_split(dir_path, **generate_split_kwargs)


def load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_truth(path: str) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)


def subset_indices(n: int, percentage: float, seed: int = DEFAULT_SEED) -> List[int]:
    """
    Deterministic index selection such that subsets at smaller percentages
    are always contained within subsets at larger percentages - so
    progressive-scaling stages see nested, not independently-sampled, data.
    Only ever applied to train.jsonl - the test set stays whole at every
    stage so scores stay comparable across stages.
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


def write_subset(src_path: str, dst_path: str, percentage: float, seed: int = DEFAULT_SEED) -> str:
    """
    Writes a HOST-SELECTED subset of src_path's rows to dst_path, so
    mounting dst_path (instead of the full file) into the sandbox enforces
    progressive scaling for real. Without this, SUBSET_PERCENTAGE is only
    an environment variable handed to the candidate - honor-system only,
    since sandbox/executor.py always mounted the full train.jsonl at every
    stage regardless of what a candidate's own code did with that env var.
    A candidate that ignores it now physically cannot see more than its
    stage's allotted rows, because the rest were never written to the file
    it can read in the first place. Returns dst_path.
    """
    rows = load_subset(src_path, percentage, seed)
    parent = os.path.dirname(dst_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(dst_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return dst_path
