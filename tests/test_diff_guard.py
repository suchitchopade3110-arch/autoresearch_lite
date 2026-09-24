import os
import tempfile

import git
import pytest

from generation.patch_generator import validate_and_apply_patch
from vcs.diff_guard import validate_diff_scope

SYMLINK_ESCAPE_DIFF = (
    "diff --git a/.eval_out b/.eval_out\n"
    "new file mode 120000\n"
    "index 0000000..abcdef1\n"
    "--- /dev/null\n"
    "+++ b/.eval_out\n"
    "@@ -0,0 +1 @@\n"
    "+../../dummy_data\n"
    "\\ No newline at end of file\n"
)

HARNESS_REWRITE_DIFF = (
    "--- a/eval/pipeline.py\n"
    "+++ b/eval/pipeline.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-old\n"
    "+return 1.0\n"
)

HONEST_CANDIDATE_DIFF = (
    "--- a/candidate_script.py\n"
    "+++ b/candidate_script.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-old\n"
    "+print('hello')\n"
)

PERMISSION_BIT_DIFF = (
    "diff --git a/candidate_script.py b/candidate_script.py\n"
    "old mode 100644\n"
    "new mode 100755\n"
)


def test_symlink_creation_outside_target_files_is_rejected():
    """
    The council audit's #1 critical finding: a diff can create .eval_out as
    a symlink (mode 120000) pointing outside the worktree - the host then
    follows it through chmod/mount and hands the sandbox rw access to
    whatever it points at. This must never pass, regardless of target.files.
    """
    ok, reason = validate_diff_scope(SYMLINK_ESCAPE_DIFF, allowed_files=["candidate_script.py"])
    assert ok is False
    assert "120000" in reason or ".eval_out" in reason


def test_symlink_creation_rejected_even_if_path_happens_to_be_allowed():
    """Mode 120000 is refused outright - being in target.files never excuses a symlink."""
    ok, reason = validate_diff_scope(SYMLINK_ESCAPE_DIFF, allowed_files=[".eval_out"])
    assert ok is False
    assert "120000" in reason


def test_diff_touching_a_file_outside_target_files_is_rejected():
    """Finding #3: a diff must not be able to rewrite the harness (e.g. eval/pipeline.py)."""
    ok, reason = validate_diff_scope(HARNESS_REWRITE_DIFF, allowed_files=["candidate_script.py"])
    assert ok is False
    assert "eval/pipeline.py" in reason


def test_permission_bit_change_is_rejected():
    ok, reason = validate_diff_scope(PERMISSION_BIT_DIFF, allowed_files=["candidate_script.py"])
    assert ok is False


def test_ordinary_candidate_diff_is_accepted():
    ok, reason = validate_diff_scope(HONEST_CANDIDATE_DIFF, allowed_files=["candidate_script.py"])
    assert ok is True
    assert reason == ""


def test_path_traversal_outside_worktree_is_rejected():
    traversal_diff = (
        "--- /dev/null\n"
        "+++ b/../../etc/passwd\n"
        "@@ -0,0 +1 @@\n"
        "+pwned\n"
    )
    ok, reason = validate_diff_scope(traversal_diff, allowed_files=["candidate_script.py"])
    assert ok is False


@pytest.fixture
def repo_dir():
    with tempfile.TemporaryDirectory() as d:
        repo = git.Repo.init(d)
        try:
            with open(os.path.join(d, "candidate_script.py"), "w") as f:
                f.write("old\n")
            repo.index.add(["candidate_script.py"])
            repo.index.commit("initial")
        finally:
            repo.close()
        yield d


def test_validate_and_apply_patch_refuses_symlink_diff_end_to_end(repo_dir):
    """
    End-to-end reproduction of the council audit's critical finding: even
    though `git apply --check` on its own would happily accept this diff
    (it's a structurally valid patch), validate_and_apply_patch must refuse
    it before it ever reaches disk, because it names a path outside
    target.files and creates a symlink.
    """
    ok = validate_and_apply_patch(
        SYMLINK_ESCAPE_DIFF, cwd=repo_dir, allowed_files=["candidate_script.py"],
    )
    assert ok is False
    assert not os.path.exists(os.path.join(repo_dir, ".eval_out"))


def test_validate_and_apply_patch_refuses_out_of_scope_diff_end_to_end(repo_dir):
    os.makedirs(os.path.join(repo_dir, "eval"), exist_ok=True)
    with open(os.path.join(repo_dir, "eval", "pipeline.py"), "w") as f:
        f.write("old\n")

    ok = validate_and_apply_patch(
        HARNESS_REWRITE_DIFF, cwd=repo_dir, allowed_files=["candidate_script.py"],
    )
    assert ok is False
    with open(os.path.join(repo_dir, "eval", "pipeline.py")) as f:
        assert f.read() == "old\n"


def test_validate_and_apply_patch_still_applies_in_scope_diffs(repo_dir):
    ok = validate_and_apply_patch(
        HONEST_CANDIDATE_DIFF, cwd=repo_dir, allowed_files=["candidate_script.py"],
    )
    assert ok is True
    with open(os.path.join(repo_dir, "candidate_script.py")) as f:
        assert f.read() == "print('hello')\n"
