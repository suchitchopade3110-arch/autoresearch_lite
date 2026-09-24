import os
import time
import uuid
from typing import Dict, Optional, Set, Tuple

import git

from observability.logging_config import get_logger

_module_logger = get_logger(__name__)


class MergeConflict(Exception):
    """
    Raised when a candidate branch can't be rebased cleanly onto the
    target branch. The rebase is aborted before this is raised, so the
    candidate's worktree is left clean (not mid-rebase) and its branch
    untouched - the caller is expected to roll it back, same as any other
    failure outcome.
    """
    def __init__(self, branch_name: str, message: str):
        super().__init__(f"Candidate branch {branch_name} could not be rebased cleanly: {message}")
        self.branch_name = branch_name
        self.message = message


class GitController:
    """
    Manages candidate branches using git worktrees, so a candidate's checkout
    always lives in its own directory. create_branch/commit_patch/rollback/merge
    never touch the caller's main working tree (or any uncommitted work in it),
    and each worktree is independent enough that concurrent candidates (see
    evolution/scheduler.py) don't need to share a checkout at all.
    """
    def __init__(self, repo_path: str = ".", worktree_root: str = None, base_ref: str = None):
        self.repo = git.Repo(repo_path)
        self.repo_path = os.path.abspath(repo_path)

        if base_ref:
            self.original_branch = base_ref
            self._tracks_named_branch = base_ref in [h.name for h in self.repo.heads]
        else:
            try:
                self.original_branch = self.repo.active_branch.name
                self._tracks_named_branch = True
            except TypeError:
                # Detached HEAD (e.g. a CI checkout of a specific commit):
                # there is no branch name to read or to rebase/merge onto,
                # so pin to the current commit instead of crashing here.
                self.original_branch = self.repo.head.commit.hexsha
                self._tracks_named_branch = False

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

    @staticmethod
    def _worktree_age_seconds(worktree_path: str) -> Optional[float]:
        """Seconds since worktree_path's directory was last touched, or None if it can't be stat'd."""
        try:
            return time.time() - os.path.getmtime(worktree_path)
        except OSError:
            return None

    def cleanup_orphans(self, min_age_seconds: float = 0) -> Dict[str, int]:
        """
        Removes candidate worktrees and branches left behind by a previous
        run that crashed (or was killed) before it could roll back or merge
        them. Safe to call unconditionally at startup FOR A SINGLE
        orchestrator process: a freshly-started process never has any
        candidates of its own yet, so with min_age_seconds left at its
        default of 0, anything matching "candidate-*" under worktree_root
        (or a "candidate-*" branch) is treated as orphaned leftovers.

        CONCURRENT RUNS: git's worktree registry (`git worktree list`) is
        repo-global, not scoped to any one process or worktree_root - a
        SECOND orchestrator process started against the same repo_path
        would, at min_age_seconds=0, see the first process's still-active
        candidates as "orphaned" too and delete them out from under it.
        orchestrator/run.py's actual startup path passes a non-zero
        min_age_seconds (the approval gate's own timeout_seconds) for
        exactly this reason: a candidate worktree younger than that could
        still be legitimately in flight (e.g. awaiting a slow human
        reviewer); one older than it would already have timed out its own
        approval wait regardless, so reclaiming it is safe even if another
        process technically still owns it. This is a heuristic (worktree
        directory mtime as a proxy for "still active"), not a real
        distributed lock - it narrows the race rather than closing it
        outright. A branch is only force-deleted if it has no live
        worktree at all, or if its worktree was just reclaimed above (an
        otherwise-young worktree's branch is left alone even if the branch
        object itself looks old, e.g. before any patch was committed).

        Never touches original_branch or the main worktree.
        """
        removed_worktrees = 0

        try:
            listing = self.repo.git.worktree("list", "--porcelain")
        except git.GitCommandError:
            listing = ""

        # A worktree is identified as an orphan candidate by the branch IT
        # HAS CHECKED OUT (parsed from git's own porcelain output), not by
        # string-comparing its path against worktree_root - Windows can
        # report the same directory in short (8.3, e.g. "SUCHIT~1") or long
        # form inconsistently between git and Python's os.path, which
        # silently breaks path-based comparison even though both refer to
        # the same directory on disk.
        orphan_paths = []
        young_candidate_branches: Set[str] = set()
        current_path = None
        current_branch = None
        for line in listing.splitlines() + [""]:
            if line.startswith("worktree "):
                current_path = line[len("worktree "):]
            elif line.startswith("branch refs/heads/"):
                current_branch = line[len("branch refs/heads/"):]
            elif line == "":
                if (
                    current_path
                    and current_branch
                    and current_branch.startswith("candidate-")
                    and os.path.abspath(current_path) != self.repo_path
                ):
                    age = self._worktree_age_seconds(current_path)
                    if age is None or age >= min_age_seconds:
                        orphan_paths.append(current_path)
                    else:
                        young_candidate_branches.add(current_branch)
                current_path = None
                current_branch = None

        for path in orphan_paths:
            self.repo.git.worktree("remove", "--force", path)
            removed_worktrees += 1

        # Drops stale administrative files for worktrees whose directory was
        # already deleted from disk (e.g. a crash mid-removal), which the
        # loop above can't see since `worktree list` no longer reports them
        # as removable paths.
        self.repo.git.worktree("prune")

        removed_branches = 0
        for head in list(self.repo.heads):
            if (
                head.name.startswith("candidate-")
                and head.name != self.original_branch
                and head.name not in young_candidate_branches
            ):
                self.repo.delete_head(head.name, force=True)
                removed_branches += 1

        return {"removed_worktrees": removed_worktrees, "removed_branches": removed_branches}

    def _active_branch_name(self) -> Optional[str]:
        try:
            return self.repo.active_branch.name
        except TypeError:
            return None  # detached HEAD

    def current_base_tip(self) -> str:
        """The commit sha original_branch (or the pinned detached-HEAD commit) currently points at."""
        return self.repo.git.rev_parse(self.original_branch)

    def worktree_head(self, worktree_path: str) -> str:
        """
        The commit sha a worktree's HEAD currently points at. Used to
        capture the exact base a candidate was created from (create_branch
        points the new branch at original_branch's tip with no new commit
        yet, so this equals that base tip when called before any patch is
        applied) - see evolution/scheduler.py's base_commit_at_eval.
        """
        wt = git.Repo(worktree_path)
        try:
            return wt.head.commit.hexsha
        finally:
            wt.close()

    def tree_sha(self, commit_ish: str) -> str:
        """The tree object a commit points at - two commits with the same tree have byte-identical content."""
        return self.repo.git.rev_parse(f"{commit_ish}^{{tree}}")

    def rebase_onto_base(self, branch_name: str, worktree_path: str) -> Tuple[str, str]:
        """
        Rebases the candidate branch onto the CURRENT target ref tip, inside
        its own worktree, and returns (old_tip, new_tip) - the base it was
        rebased onto, and the resulting rebased commit - WITHOUT touching
        the shared ref or the main working tree at all. This is a
        deliberately tentative step: a caller can still walk away from it
        (leaving the worktree/branch for rollback, exactly like any other
        failure) if a check performed between rebasing and finalizing - a
        fresh baseline comparison, a re-evaluation of the rebased code -
        fails. See evolution/scheduler.py's phase 2 for why that gap
        matters: rebasing changes what the candidate's diff actually
        produces whenever the base moved since the candidate was evaluated
        (e.g. an earlier candidate in the same generation merged first),
        so a score computed before the rebase does not necessarily describe
        the code finalize_merge would actually publish.

        Raises MergeConflict if the rebase itself conflicts - the rebase is
        aborted first, so the worktree is left clean (not mid-rebase).
        """
        wt = git.Repo(worktree_path)
        try:
            old_tip = wt.git.rev_parse(self.original_branch)
            wt.git.rebase(self.original_branch)
            new_tip = wt.head.commit.hexsha
            return old_tip, new_tip
        except git.GitCommandError as e:
            try:
                wt.git.rebase("--abort")
            except git.GitCommandError:
                pass
            raise MergeConflict(branch_name, str(e))
        finally:
            wt.close()

    def finalize_merge(self, branch_name: str, worktree_path: str, old_tip: str, new_tip: str) -> None:
        """
        Advances the target ref to `new_tip` via `git update-ref` with a
        compare-and-swap check against `old_tip` (as returned by
        rebase_onto_base) - never a `git checkout`/`git merge` in the
        caller's main working tree. This is what makes "never touches the
        caller's main checkout" actually true: the previous implementation
        ran `git checkout <base>` directly in the main worktree, which
        could silently switch the caller's currently checked-out branch
        (if base_ref names a branch other than the one they're on), and
        its follow-up `git merge --ff-only` raised a bare (uncaught)
        GitCommandError - not MergeConflict - whenever the main working
        tree had uncommitted changes to a file the candidate also touched,
        crashing the run after approval had already been recorded and
        leaking the candidate's worktree.

        If the main worktree happens to already be sitting on
        original_branch (or, in the detached-HEAD/pinned-commit case, is
        still detached) AND has no uncommitted changes to tracked files,
        its working-tree files are also fast-forwarded to match (a plain
        reset, safe because nothing tracked is uncommitted) - purely a
        convenience so a human working directly in the main checkout sees
        the merge without a separate `git pull`/`checkout`. Otherwise the
        working tree is left completely untouched; only the ref moves.

        Raises MergeConflict if the compare-and-swap fails because
        original_branch moved concurrently since old_tip was captured
        (e.g. another candidate finalized first) - the candidate's
        worktree/branch are left in place for the caller to roll back
        (or re-rebase and retry), exactly like any other failure outcome.
        """
        # Must be read BEFORE update_ref moves the pointer below - once HEAD/
        # the branch points at new_tip while the index still reflects the
        # pre-merge tree, is_dirty() reports "dirty" purely from that
        # index/HEAD mismatch, not from any real uncommitted work.
        was_dirty = self.repo.is_dirty(untracked_files=False)

        ref_name = f"refs/heads/{self.original_branch}" if self._tracks_named_branch else "HEAD"
        try:
            # CAS-verified pointer move: works for a detached HEAD too (it
            # stays a raw-sha ref, never becomes symbolic) - if HEAD/the
            # branch no longer points at old_tip, someone else advanced it
            # concurrently and this fails instead of silently discarding
            # that other change.
            self.repo.git.update_ref(ref_name, new_tip, old_tip)
        except git.GitCommandError as e:
            raise MergeConflict(branch_name, f"{self.original_branch} moved concurrently: {e}")

        if not self._tracks_named_branch:
            # A detached HEAD has no branch to auto-advance to the new tip.
            # Without this, the next candidate would rebase onto this
            # now-stale commit, and its own merge could silently discard
            # this one instead of building on it.
            self.original_branch = new_tip

        on_target = (
            self._active_branch_name() == self.original_branch
            if self._tracks_named_branch
            else self.repo.head.is_detached
        )
        if on_target and not was_dirty:
            self.repo.head.reset(new_tip, index=True, working_tree=True)
        elif on_target:
            _module_logger.warning(
                f"Candidate {branch_name} merged (ref advanced), but the main working tree has "
                "uncommitted changes to tracked files - its files were left untouched. Run "
                "`git status`/`git checkout` there to see the merge."
            )

        self.repo.git.worktree("remove", "--force", worktree_path)
        self.repo.delete_head(branch_name, force=True)

    def merge(self, branch_name: str, worktree_path: str) -> None:
        """
        Convenience wrapper for a single-candidate-at-a-time caller (the
        sequential orchestrator path, and any test that doesn't need to
        inspect the rebased tip before finalizing): rebase_onto_base()
        followed immediately by finalize_merge(). See both for the full
        contract; evolution/scheduler.py's phase 2 calls them separately
        so it can re-check the baseline and re-evaluate the rebased result
        before deciding whether to finalize at all.
        """
        old_tip, new_tip = self.rebase_onto_base(branch_name, worktree_path)
        self.finalize_merge(branch_name, worktree_path, old_tip, new_tip)
