import os
import uuid
import tempfile
import pytest
import git
from vcs.git_controller import GitController

@pytest.fixture
def repo_dir():
    with tempfile.TemporaryDirectory() as d:
        repo = git.Repo.init(d)

        # Create an initial commit so we have a master branch
        test_file = os.path.join(d, "test.txt")
        with open(test_file, "w") as f:
            f.write("initial state")
        repo.index.add(["test.txt"])
        repo.index.commit("initial commit")

        yield d

def test_git_controller_rollback(repo_dir):
    controller = GitController(repo_dir)
    original_branch = controller.original_branch

    # 1. Create a branch and make a change
    candidate_id = "123"
    branch_name = controller.create_branch(candidate_id)

    assert controller.repo.active_branch.name == branch_name

    test_file = os.path.join(repo_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("changed state")

    controller.commit_patch()

    # Check that it changed
    with open(test_file, "r") as f:
        assert f.read() == "changed state"

    # 2. Rollback
    controller.rollback(branch_name)

    # Check that we are back on original branch
    assert controller.repo.active_branch.name == original_branch

    # Check that the file was restored
    with open(test_file, "r") as f:
        assert f.read() == "initial state"

    # Check that the candidate branch is deleted
    assert branch_name not in [h.name for h in controller.repo.heads]

def test_git_controller_merge(repo_dir):
    controller = GitController(repo_dir)
    original_branch = controller.original_branch

    # 1. Create a branch and make a change
    candidate_id = "456"
    branch_name = controller.create_branch(candidate_id)

    test_file = os.path.join(repo_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("changed state")

    controller.commit_patch()

    # 2. Merge
    controller.merge_and_push(branch_name)

    # Check that we are back on original branch
    assert controller.repo.active_branch.name == original_branch

    # Check that the change is in the original branch
    with open(test_file, "r") as f:
        assert f.read() == "changed state"

    # Check that the candidate branch is deleted
    assert branch_name not in [h.name for h in controller.repo.heads]
