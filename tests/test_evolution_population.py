import os
import subprocess
from unittest.mock import MagicMock

from evolution.population import EvolutionEngine
from generation.patch_generator import PatchGenerator, LLMClient
from generation.prompt_builder import PromptBuilder
from memory.db import ExperimentDB
from orchestrator.metrics import calculate_all_metrics
from memory.failure_analysis import analyze_failure
from vcs.git_controller import GitController

DUPLICATE_DIFF = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('always the same')\n"


class _AlwaysSameDiffClient(LLMClient):
    """A stub LLM client that always returns the identical diff - guaranteed to be flagged a duplicate once one copy is on record."""
    def generate_diff(self, prompt, target_file, current_content=""):
        return DUPLICATE_DIFF


def _init_repo(tmp_dir):
    repo_dir = os.path.join(tmp_dir, "repo")
    os.makedirs(repo_dir)
    with open(os.path.join(repo_dir, "candidate_script.py"), "w") as f:
        f.write("\n")
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)
    return repo_dir


def test_duplicate_exhaustion_produces_a_record_and_never_touches_the_sandbox(tmp_dir):
    """
    Wave 2 acceptance: when every retry for a candidate slot is rejected as
    a duplicate, the engine must record a distinct duplicate_exhausted
    outcome and skip the candidate entirely - never substituting a
    scoreless fallback diff that would otherwise burn a full
    sandbox x stages budget on something that cannot score.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_path=repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    # Pre-seed the exact diff the stub client will always return, so every
    # attempt is an exact-match duplicate from the very first retry.
    db.store_experiment(hypothesis="goal", diff=DUPLICATE_DIFF, rationale="seed", metrics={}, outcome="success")

    prompt_builder = PromptBuilder(db, {})
    patch_generator = PatchGenerator(_AlwaysSameDiffClient())
    sandbox = MagicMock()

    config = {
        "evolution": {"population_size": 2, "max_generations": 1, "duplicate_threshold": 0.25},
        "eval": {"stages": []},
    }

    engine = EvolutionEngine(
        config=config,
        git_controller=git_controller,
        sandbox=sandbox,
        evaluator=None,
        metrics_calculator=calculate_all_metrics,
        failure_analyzer=analyze_failure,
        patch_generator=patch_generator,
        prompt_builder=prompt_builder,
        db=db,
    )

    population = engine._generate_population("goal", n=2)

    assert population == []
    assert engine.duplicate_exhausted_count == 2
    sandbox.run_candidate.assert_not_called()

    results = db.retrieve_experiments(query="goal", k=10)
    exhausted = [r for r in results if r.get("outcome") == "duplicate_exhausted"]
    assert len(exhausted) == 2


def test_generate_candidate_creates_and_cleans_up_worktree_on_exhaustion(tmp_dir):
    """
    Wave 2.3: _generate_candidate creates the candidate's worktree before
    generating (so the diff is checked against real current content), and
    must roll it back if generation is ultimately exhausted - it must not
    leak a worktree/branch for a candidate that never gets scheduled.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_path=repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    db.store_experiment(hypothesis="goal", diff=DUPLICATE_DIFF, rationale="seed", metrics={}, outcome="success")

    prompt_builder = PromptBuilder(db, {})
    patch_generator = PatchGenerator(_AlwaysSameDiffClient())

    config = {"evolution": {"duplicate_threshold": 0.25}, "eval": {"stages": []}}
    engine = EvolutionEngine(
        config=config,
        git_controller=git_controller,
        sandbox=MagicMock(),
        evaluator=None,
        metrics_calculator=calculate_all_metrics,
        failure_analyzer=analyze_failure,
        patch_generator=patch_generator,
        prompt_builder=prompt_builder,
        db=db,
    )

    result = engine._generate_candidate("goal")

    assert result is None
    assert os.listdir(os.path.join(tmp_dir, "worktrees")) == []
    assert list(git_controller.repo.heads) == [git_controller.repo.heads[git_controller.original_branch]]


def _make_engine(tmp_dir, repo_dir):
    git_controller = GitController(repo_path=repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))
    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    config = {"evolution": {"duplicate_threshold": 0.25}, "eval": {"stages": []}}
    engine = EvolutionEngine(
        config=config,
        git_controller=git_controller,
        sandbox=MagicMock(),
        evaluator=None,
        metrics_calculator=calculate_all_metrics,
        failure_analyzer=analyze_failure,
        patch_generator=PatchGenerator(_AlwaysSameDiffClient()),
        prompt_builder=PromptBuilder(db, {}),
        db=db,
    )
    return engine, git_controller, db


def test_carry_over_elite_skips_a_merged_parent(tmp_dir):
    """
    Council audit finding: if the elite parent actually merged, its change
    is already part of the current base - blindly re-applying the same
    diff on top of that base wastes a full sandbox slot on either a no-op
    or a hard apply failure (the diff's removed-line context no longer
    matches). A merged parent must never be carried over.
    """
    repo_dir = _init_repo(tmp_dir)
    engine, git_controller, db = _make_engine(tmp_dir, repo_dir)

    merged_parent = {"diff": "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('merged')\n", "goal": "goal", "success": True}

    result = engine._carry_over_elite("goal", merged_parent)

    assert result is None
    assert os.listdir(os.path.join(tmp_dir, "worktrees")) == []


def test_carry_over_elite_skips_a_duplicate_of_memory(tmp_dir):
    """A held-but-unmerged parent whose diff already matches something on record must not burn a slot either."""
    repo_dir = _init_repo(tmp_dir)
    engine, git_controller, db = _make_engine(tmp_dir, repo_dir)
    db.store_experiment(hypothesis="goal", diff=DUPLICATE_DIFF, rationale="seed", metrics={}, outcome="success")

    held_parent = {"diff": DUPLICATE_DIFF, "goal": "goal", "success": False}

    result = engine._carry_over_elite("goal", held_parent)

    assert result is None
    assert os.listdir(os.path.join(tmp_dir, "worktrees")) == []


def test_carry_over_elite_skips_a_diff_that_no_longer_applies_and_rolls_back(tmp_dir):
    """
    A parent's diff generated against a stale base may simply fail to
    apply against the CURRENT base - it must be dry-run-checked, not
    handed straight to the scheduler where a real apply failure would be
    recorded as a "runtime" crash instead of being caught here.
    """
    repo_dir = _init_repo(tmp_dir)
    engine, git_controller, db = _make_engine(tmp_dir, repo_dir)

    stale_parent = {
        "diff": "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-this line does not exist in the base\n+print('stale')\n",
        "goal": "goal",
        "success": False,
    }

    result = engine._carry_over_elite("goal", stale_parent)

    assert result is None
    # The worktree it tentatively created for the dry-run check is rolled
    # back, not left behind.
    assert os.listdir(os.path.join(tmp_dir, "worktrees")) == []
    assert list(git_controller.repo.heads) == [git_controller.repo.heads[git_controller.original_branch]]


def test_carry_over_elite_succeeds_for_a_valid_unmerged_parent(tmp_dir):
    repo_dir = _init_repo(tmp_dir)
    engine, git_controller, db = _make_engine(tmp_dir, repo_dir)

    held_parent = {
        "diff": "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('still good')\n",
        "goal": "goal",
        "success": False,
    }

    result = engine._carry_over_elite("goal", held_parent)

    assert result is not None
    assert result["diff"] == held_parent["diff"]
    assert os.path.isdir(result["worktree_path"])
    # dry_run=True (matching _generate_candidate's own contract): only
    # checked, not actually applied yet - the scheduler's phase 1 applies
    # it for real. The worktree's content is still the pristine base.
    with open(os.path.join(result["worktree_path"], "candidate_script.py")) as f:
        assert f.read() == "\n"


def test_select_parents_returns_empty_list_when_every_candidate_slot_failed(tmp_dir):
    """
    Regression test: a real generation-1 run against a weaker/smaller LLM
    can have every slot exhaust its malformed-diff/duplicate retries,
    leaving zero scored candidates. 'tournament' (the default strategy)
    used to call max() on an empty sample in that case, crashing the
    whole run instead of falling back to ordinary generation for the next
    round (see run()'s `if mutation_source:` check, which already treats
    an empty parent list the same as no parents at all).
    """
    repo_dir = _init_repo(tmp_dir)
    engine, git_controller, db = _make_engine(tmp_dir, repo_dir)
    engine.config["selection_strategy"] = "tournament"

    assert engine._select_parents([]) == []


def test_select_parents_top_k_and_default_strategies_also_handle_an_empty_population(tmp_dir):
    repo_dir = _init_repo(tmp_dir)
    engine, git_controller, db = _make_engine(tmp_dir, repo_dir)

    for strategy in ("top-k", "some-unrecognized-strategy"):
        engine.config["selection_strategy"] = strategy
        assert engine._select_parents([]) == []
