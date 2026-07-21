import os
import uuid
from typing import Tuple

import git


class GitController:
    """
    Manages candidate branches using git worktrees, so a candidate's checkout
    always lives in its own directory. create_branch/commit_patch/rollback/merge
    never touch the caller's main working tree (or any uncommitted work in it),
    and each worktree is independent enough that concurrent candidates (see
    evolution/scheduler.py) don't need to share a checkout at all.
    """
    def __init__(self, repo_path: str = ".", worktree_root: str = None):
        self.repo = git.Repo(repo_path)
        self.repo_path = os.path.abspath(repo_path)
        self.original_branch = self.repo.active_branch.name
        self.worktree_root = worktree_root or os.path.join(self.repo_path, ".candidate_worktrees")
        os.makedirs(self.worktree_root, exist_ok=True)

    def create_branch(self, candidate_id: str) -> Tuple[str, str]:
        """Creates a unique branch for the candidate in its own worktree. Returns (branch_name, worktree_path)."""
        branch_name = f"candidate-{candidate_id}-{uuid.uuid4().hex[:8]}"
        worktree_path = os.path.join(self.worktree_root, branch_name)
        self.repo.git.worktree("add", "-b", branch_name, worktree_path, self.original_branch)
        return branch_name, worktree_path

    def commit_patch(self, worktree_path: str, message: str = "Apply candidate patch") -> bool:
        """Commits all current changes within the candidate's worktree."""
        wt_repo = git.Repo(worktree_path)
        try:
            if not wt_repo.is_dirty(untracked_files=True):
                return False

            wt_repo.git.add(A=True)
            wt_repo.index.commit(message)
            return True
        finally:
            # Windows keeps a file lock on the worktree while this handle is
            # open, so a later `git worktree remove --force` on this same
            # path (in rollback/merge) fails with "Permission denied" unless
            # this is explicitly released first - Linux/macOS never enforce
            # that, which is why this only shows up on Windows.
            wt_repo.close()

    def rollback(self, branch_name: str, worktree_path: str) -> None:
        """Discards the candidate's worktree and deletes its branch. Never touches the main working tree."""
        if os.path.exists(worktree_path):
            self.repo.git.worktree("remove", "--force", worktree_path)
        self.repo.delete_head(branch_name, force=True)

    def merge(self, branch_name: str, worktree_path: str) -> None:
        """Merges the candidate branch into the original branch and cleans up its worktree."""
        if os.path.exists(worktree_path):
            self.repo.git.worktree("remove", "--force", worktree_path)
        self.repo.heads[self.original_branch].checkout()
        self.repo.git.merge(branch_name)
        self.repo.delete_head(branch_name, force=True)
