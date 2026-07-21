import git
import os
import uuid
from typing import Optional

class GitController:
    def __init__(self, repo_path: str = "."):
        self.repo = git.Repo(repo_path)
        self.original_branch = self.repo.active_branch.name

    def create_branch(self, candidate_id: str) -> str:
        """Creates a unique branch for the candidate and checks it out."""
        branch_name = f"candidate-{candidate_id}-{uuid.uuid4().hex[:8]}"
        new_branch = self.repo.create_head(branch_name)
        new_branch.checkout()
        return branch_name

    def commit_patch(self, message: str = "Apply candidate patch") -> bool:
        """Commits all current changes in the working tree."""
        if not self.repo.is_dirty(untracked_files=True):
            return False

        self.repo.git.add(A=True)
        self.repo.index.commit(message)
        return True

    def rollback(self, branch_name: str) -> None:
        """Reverts state by doing a hard reset, checking out the original branch, and deleting the candidate branch."""
        self.repo.git.reset('--hard')
        self.repo.git.clean('-fd')
        self.repo.heads[self.original_branch].checkout()

        # Delete the failed branch
        self.repo.delete_head(branch_name, force=True)

    def merge_and_push(self, branch_name: str) -> None:
        """Merges the candidate branch into the original branch and deletes the candidate branch."""
        self.repo.heads[self.original_branch].checkout()
        self.repo.git.merge(branch_name)
        self.repo.delete_head(branch_name, force=True)
