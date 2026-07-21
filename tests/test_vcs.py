import os
import tempfile

import git
import pytest

from vcs.git_controller import GitController


@pytest.fixture
def repo_dir():
    with tempfile.TemporaryDirectory() as d:
        repo = git.Repo.init(d)

        test_file = os.path.join(d, "test.txt")
        with open(test_file, "w") as f:
            f.write("initial state")
        repo.index.add(["test.txt"])
        repo.index.commit("initial commit")

        yield d


def test_create_branch_does_not_touch_main_worktree(repo_dir):
    controller = GitController(repo_dir)
    original_branch = controller.original_branch

    branch_name, worktree_path = controller.create_branch("123")

    # main working tree stays checked out on the original branch throughout
    assert controller.repo.active_branch.name == original_branch
    assert os.path.isdir(worktree_path)
    assert git.Repo(worktree_path).active_branch.name == branch_name


def test_git_controller_rollback(repo_dir):
    controller = GitController(repo_dir)
    original_branch = controller.original_branch

    branch_name, worktree_path = controller.create_branch("123")
    wt_file = os.path.join(worktree_path, "test.txt")
    with open(wt_file, "w") as f:
        f.write("changed state")
    controller.commit_patch(worktree_path)

    # the candidate's change lives only in its own worktree
    with open(os.path.join(repo_dir, "test.txt")) as f:
        assert f.read() == "initial state"

    controller.rollback(branch_name, worktree_path)

    assert controller.repo.active_branch.name == original_branch
    assert branch_name not in [h.name for h in controller.repo.heads]
    assert not os.path.exists(worktree_path)


def test_git_controller_merge(repo_dir):
    controller = GitController(repo_dir)
    original_branch = controller.original_branch

    branch_name, worktree_path = controller.create_branch("456")
    wt_file = os.path.join(worktree_path, "test.txt")
    with open(wt_file, "w") as f:
        f.write("changed state")
    controller.commit_patch(worktree_path)

    controller.merge(branch_name, worktree_path)

    assert controller.repo.active_branch.name == original_branch
    with open(os.path.join(repo_dir, "test.txt")) as f:
        assert f.read() == "changed state"
    assert branch_name not in [h.name for h in controller.repo.heads]
    assert not os.path.exists(worktree_path)


def test_rollback_never_touches_uncommitted_changes_in_main_worktree(repo_dir):
    """
    Regression test: the pre-worktree implementation checked candidate
    branches out directly in the caller's repo and hard-reset + git-clean'd
    it on rollback, which would destroy any uncommitted work sitting in the
    caller's working tree. Worktree isolation must prevent that.
    """
    controller = GitController(repo_dir)

    scratch_file = os.path.join(repo_dir, "scratch.txt")
    with open(scratch_file, "w") as f:
        f.write("uncommitted work in progress")

    branch_name, worktree_path = controller.create_branch("789")
    with open(os.path.join(worktree_path, "test.txt"), "w") as f:
        f.write("candidate change")
    controller.commit_patch(worktree_path)

    controller.rollback(branch_name, worktree_path)

    assert os.path.exists(scratch_file)
    with open(scratch_file) as f:
        assert f.read() == "uncommitted work in progress"


def test_two_concurrent_candidates_get_independent_worktrees(repo_dir):
    """The evolutionary scheduler relies on this: candidates never share a checkout."""
    controller = GitController(repo_dir)

    branch_a, worktree_a = controller.create_branch("aaa")
    branch_b, worktree_b = controller.create_branch("bbb")

    assert worktree_a != worktree_b
    # each candidate edits its own new file, so both merges apply cleanly -
    # this test is about checkout isolation, not merge-conflict resolution
    with open(os.path.join(worktree_a, "a.txt"), "w") as f:
        f.write("from a")
    with open(os.path.join(worktree_b, "b.txt"), "w") as f:
        f.write("from b")

    assert not os.path.exists(os.path.join(worktree_a, "b.txt"))
    assert not os.path.exists(os.path.join(worktree_b, "a.txt"))

    controller.commit_patch(worktree_a)
    controller.commit_patch(worktree_b)
    controller.merge(branch_a, worktree_a)
    controller.merge(branch_b, worktree_b)

    assert os.path.exists(os.path.join(repo_dir, "a.txt"))
    assert os.path.exists(os.path.join(repo_dir, "b.txt"))
