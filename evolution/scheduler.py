import concurrent.futures
import os
import shutil
import tempfile
import threading
from typing import Any, Dict, List, Optional

from approval.gate import await_approval_decision, create_approval_request, maybe_auto_approve
from approval.store import ApprovalStore
from generation.patch_generator import validate_and_apply_patch
from generation.static_check import check_syntax_multi
from observability.logging_config import bind, get_logger
from vcs.git_controller import MergeConflict


def _run_eval_stages(script_path, out_dir, pred_path, eval_stages, sandbox, evaluator,
                      metrics_calculator, failure_analyzer, truth, candidate_logger) -> Dict[str, Any]:
    """
    Runs `script_path` through every eval stage and scores it - the same
    stage loop used by phase 1 (evaluate_only) and by phase 2's
    re-evaluation of a rebased candidate (see execute_generation below).
    Factored out so both call sites can never drift apart on what
    "evaluating a candidate" actually means.
    """
    final_score = 0.0
    all_metrics: Dict[str, Any] = {}
    eval_passed = True
    failure_category = "success"
    error_msg = ""
    traceback_text = ""
    total_execution_time = 0.0
    last_subset = None
    has_failure_flags = False

    for stage in eval_stages:
        subset = stage['subset_percentage']
        threshold = stage['threshold']

        # Wipe any predictions left by the previous stage - without this, a
        # stage that exits 0 without writing predictions.jsonl would be
        # scored against the PRIOR stage's file.
        if os.path.exists(pred_path):
            os.remove(pred_path)

        env = {"SUBSET_PERCENTAGE": str(subset)}
        exec_result = sandbox.run_candidate(script_path, env_vars=env, out_dir=out_dir)
        total_execution_time += exec_result.get('execution_time', 0.0)
        exec_result['execution_time'] = total_execution_time

        stage_success, stage_score, stage_mismatch = evaluator.evaluate_stage(
            exec_result, subset, threshold, pred_path, truth, logger=candidate_logger
        )
        has_failure_flags = has_failure_flags or stage_mismatch
        last_subset = subset

        if not stage_success:
            eval_passed = False
            final_score = stage_score
            cat, msg, tb = failure_analyzer(exec_result, False)
            failure_category = cat
            error_msg = msg
            traceback_text = tb
            all_metrics = metrics_calculator(exec_result)
            break

        final_score = stage_score
        all_metrics = metrics_calculator(exec_result)

    return {
        'final_score': final_score,
        'all_metrics': all_metrics,
        'eval_passed': eval_passed,
        'failure_category': failure_category,
        'error_msg': error_msg,
        'traceback_text': traceback_text,
        'total_execution_time': total_execution_time,
        'last_subset': last_subset,
        'has_failure_flags': has_failure_flags,
    }


class ConcurrentScheduler:
    def __init__(self, max_workers: int, logger=None):
        self.max_workers = max_workers
        # Each candidate now gets its own git worktree, so patch application,
        # commit, and sandbox execution never share a checkout - only the
        # repo-level branch/merge/rollback calls below still touch the
        # single shared git_controller.repo object and need serializing.
        self.git_lock = threading.Lock()
        self.logger = logger or get_logger(__name__)

    def execute_generation(self,
                          candidates: List[Dict[str, Any]],
                          eval_stages: List[Dict[str, Any]],
                          git_controller,
                          sandbox,
                          evaluator,
                          metrics_calculator,
                          failure_analyzer,
                          approval_store: Optional[ApprovalStore] = None,
                          approval_config: Optional[Dict[str, Any]] = None,
                          truth: Optional[Dict[str, int]] = None,
                          baseline_store=None) -> List[Dict[str, Any]]:
        """
        Two phases, so a human is never a bottleneck on the sandbox pool:

        Phase 1 (bounded by max_workers, no human in the loop): apply each
        candidate's patch and run it through every eval stage. A candidate
        that fails evaluation is rolled back immediately here - it will
        never need approval.

        Phase 2 (unbounded, serial): every candidate that passed
        evaluation gets its approval request created up front, all at
        once, so a reviewer sees the whole generation together instead of
        candidates trickling in one at a time as sandbox slots free up.
        Only then are decisions awaited and candidates merged/rolled back -
        and for each one, its rebase is finalized only after re-checking
        both the baseline and (if the base actually moved since phase 1
        scored it) a fresh evaluation. See the phase 2 loop below for why:
        an earlier candidate finalizing first can move the base out from
        under a later one, invalidating both its score and its baseline
        comparison before finalize_merge would otherwise publish it
        unverified.
        """
        # Fail safe: if the caller didn't wire a store, still gate merges
        # rather than silently skipping approval - only an explicit, valid
        # approval.enabled: false in approval_config actually disables it.
        store = approval_store or ApprovalStore()
        gate_config = approval_config or {}
        truth = truth or {}
        eval_section = gate_config.get('eval')
        min_improvement = eval_section.get('min_improvement', 0.001) if isinstance(eval_section, dict) else 0.001

        def evaluate_only(candidate: Dict[str, Any]) -> Dict[str, Any]:
            """
            Phase 1 body: apply, commit, evaluate. Never touches approval or
            merge.

            KNOWN LIMITATION: hardcoded to "candidate_script.py", like
            evolution/population.py's candidate generation - evolutionary
            mode does not support target.files multi-file candidates yet.
            """
            c_id = candidate['id']
            diff = candidate['diff']
            candidate_logger = bind(self.logger, candidate_id=c_id)

            branch_name = candidate.get('branch_name')
            worktree_path = candidate.get('worktree_path')

            try:
                if not worktree_path:
                    # No pre-created worktree (e.g. an elite carryover) -
                    # create one now, same as before candidates got their
                    # own worktree at generation time.
                    with self.git_lock:
                        branch_name, worktree_path = git_controller.create_branch(c_id)
                # else: the worktree was already created (and the diff's
                # dry-run already checked) at generation time, against this
                # exact current file content - see evolution/population.py.

                # Captured BEFORE any patch is applied, so this is exactly
                # the base commit this candidate's diff was tested against -
                # create_branch points the new branch at the base tip with
                # no commit of its own yet. Phase 2 compares this against
                # the base at finalize time to decide whether a rebase
                # changed what's actually about to be published (see below).
                with self.git_lock:
                    base_commit_at_eval = git_controller.worktree_head(worktree_path)

                script_path = os.path.join(worktree_path, "candidate_script.py")
                if not os.path.exists(script_path):
                    with open(script_path, "w") as f:
                        f.write("\n")

                if diff.strip():
                    apply_error: List[str] = []
                    if not validate_and_apply_patch(
                        diff, cwd=worktree_path, logger=candidate_logger, error_out=apply_error,
                        allowed_files=["candidate_script.py"],
                    ):
                        # The real git-apply diagnostic, not just a generic
                        # label - see generation/patch_generator.py's
                        # error_out param.
                        detail = apply_error[0] if apply_error else "unknown error"
                        raise RuntimeError(f"Patch failed to apply for candidate {c_id}: {detail}")

                git_controller.commit_patch(worktree_path, f"Add candidate {c_id}")

                # Static pre-check, mirroring orchestrator/run.py's
                # sequential path - README claims this rejects bad code
                # before it ever reaches the sandbox, but evolutionary mode
                # used to skip it entirely, wasting a full sandbox run on a
                # candidate that can never score anything but a crash.
                syntax_ok, syntax_err = check_syntax_multi([script_path])
                if not syntax_ok:
                    candidate['branch_name'] = branch_name
                    candidate['worktree_path'] = worktree_path
                    candidate['base_commit_at_eval'] = base_commit_at_eval
                    candidate['eval_passed'] = False
                    candidate['final_score'] = 0.0
                    candidate['metrics'] = {}
                    candidate['failure_category'] = "syntax_error"
                    candidate['error_msg'] = syntax_err
                    candidate['traceback'] = syntax_err
                    candidate['total_execution_time'] = 0.0
                    candidate['last_subset'] = None
                    candidate['success'] = False
                    candidate['approval_decision'] = None
                    with self.git_lock:
                        git_controller.rollback(branch_name, worktree_path)
                    return candidate

                # Fresh host tempdir, NOT a path under worktree_path - see
                # orchestrator/run.py's sequential-mode equivalent for why: a
                # path inside the diff-controlled worktree can be turned
                # into a symlink escaping the sandbox's rw mount to
                # anywhere on the host. Removed in the `finally` below.
                out_dir = tempfile.mkdtemp(prefix="autoresearch-out-")
                pred_path = os.path.join(out_dir, "predictions.jsonl")
                try:
                    result = _run_eval_stages(
                        script_path, out_dir, pred_path, eval_stages, sandbox, evaluator,
                        metrics_calculator, failure_analyzer, truth, candidate_logger,
                    )
                finally:
                    shutil.rmtree(out_dir, ignore_errors=True)

                final_score = result['final_score']
                all_metrics = result['all_metrics']
                eval_passed = result['eval_passed']
                failure_category = result['failure_category']
                error_msg = result['error_msg']
                traceback_text = result['traceback_text']
                total_execution_time = result['total_execution_time']
                last_subset = result['last_subset']
                has_failure_flags = result['has_failure_flags']

                # Baseline gate - see orchestrator/run.py for the rationale
                # AND for why baseline_score/delta are None (not 0.0) when
                # nothing has ever merged at this stage: delta feeds
                # should_auto_approve's "improvement over baseline"
                # criterion, and a candidate with nothing to compare
                # against must always fall through to human review. This is
                # phase 1's read of the baseline - phase 2 re-reads and
                # re-checks it immediately before finalizing, since an
                # earlier candidate in this same generation may raise it in
                # the meantime.
                baseline_score = (
                    baseline_store.get(last_subset) if (baseline_store and last_subset is not None) else None
                )
                delta = (final_score - baseline_score) if baseline_score is not None else None
                if eval_passed and baseline_store and last_subset is not None:
                    if not baseline_store.passes(last_subset, final_score, min_improvement):
                        eval_passed = False
                        failure_category = "below_baseline"
                        shown_baseline = baseline_score if baseline_score is not None else 0.0
                        error_msg = f"score {final_score:.4f} did not beat baseline {shown_baseline:.4f} + {min_improvement}"

                all_metrics['baseline_score'] = baseline_score if baseline_score is not None else 0.0
                all_metrics['delta'] = delta
                # Consumed by approval/gate.py's should_auto_approve as the
                # require_no_failure_flags criterion - see
                # orchestrator/run.py's sequential-path equivalent.
                all_metrics['score_claim_mismatch'] = has_failure_flags
                generation_usage = candidate.get('generation_usage') or {}
                if generation_usage:
                    all_metrics['generation_input_tokens'] = generation_usage.get('input_tokens', 0)
                    all_metrics['generation_output_tokens'] = generation_usage.get('output_tokens', 0)
                    all_metrics['generation_cost_usd'] = generation_usage.get('estimated_cost_usd', 0.0)

                candidate['branch_name'] = branch_name
                candidate['worktree_path'] = worktree_path
                candidate['base_commit_at_eval'] = base_commit_at_eval
                candidate['eval_passed'] = eval_passed
                candidate['final_score'] = final_score
                candidate['metrics'] = all_metrics
                candidate['failure_category'] = failure_category
                candidate['error_msg'] = error_msg
                candidate['traceback'] = traceback_text
                candidate['total_execution_time'] = total_execution_time
                candidate['last_subset'] = last_subset
                candidate['success'] = False
                candidate['approval_decision'] = None

                if not eval_passed:
                    # Never needs approval - roll back now rather than
                    # carrying a dead candidate into phase 2.
                    with self.git_lock:
                        git_controller.rollback(branch_name, worktree_path)

                return candidate

            except Exception as e:
                with self.git_lock:
                    if branch_name and worktree_path:
                        try:
                            active_branches = [h.name for h in git_controller.repo.heads]
                        except AttributeError:
                            active_branches = git_controller.repo.heads.keys()
                        if branch_name in active_branches:
                            git_controller.rollback(branch_name, worktree_path)
                candidate['success'] = False
                candidate['eval_passed'] = False
                candidate['approval_decision'] = None
                candidate['failure_category'] = "runtime"
                candidate['error_msg'] = str(e)
                candidate['traceback'] = str(e)
                candidate['metrics'] = {}
                candidate['final_score'] = 0.0
                candidate['total_execution_time'] = 0.0
                return candidate

        # Phase 1 - bounded by max_workers, no human in the loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(evaluate_only, c) for c in candidates]
            evaluated = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Phase 2 - enqueue the whole generation's approval requests first,
        # so a reviewer sees them all together, then drain serially.
        passed = [c for c in evaluated if c.get('eval_passed')]
        for c in passed:
            request_id = create_approval_request(
                store, c['id'], c.get('goal', 'Optimization goal'), c['diff'], c['final_score'], c['metrics'], gate_config
            )
            c['_approval_request_id'] = request_id
            if request_id is None:
                c['approval_decision'] = 'skipped'
            else:
                c['approval_decision'] = maybe_auto_approve(
                    store, request_id, gate_config,
                    c['metrics'].get('delta'), bool(c['metrics'].get('score_claim_mismatch')),
                )

        for c in passed:
            if c['approval_decision'] is None:
                c['approval_decision'] = await_approval_decision(store, c['_approval_request_id'], gate_config)
            decision = c['approval_decision']

            with self.git_lock:
                if decision not in ("approved", "auto_approved", "skipped"):
                    git_controller.rollback(c['branch_name'], c['worktree_path'])
                    c['failure_category'] = "held"
                    c['error_msg'] = f"approval_decision={decision}"
                    c.pop('_approval_request_id', None)
                    continue

                # Captured before rebasing: whether this candidate ever
                # actually committed anything distinct from the base it was
                # evaluated against. Needed to tell apart two cases that
                # both end up with new_tip == old_tip after rebase_onto_base
                # below - a candidate with an empty/no-op diff from the
                # start (harmless, nothing to reject) vs. a candidate whose
                # real commit git's rebase drops entirely because it's a
                # byte-for-byte duplicate of what the current base already
                # has (see the no-op check further down).
                pre_rebase_tip = git_controller.worktree_head(c['worktree_path'])
                had_real_commit = pre_rebase_tip != c.get('base_commit_at_eval')

                try:
                    old_tip, new_tip = git_controller.rebase_onto_base(c['branch_name'], c['worktree_path'])
                except MergeConflict as e:
                    # Rebasing onto the current base failed - routine
                    # traffic when several candidates merge into the same
                    # base per generation, not a code failure. Distinct
                    # from "failure" so the population's staleness is
                    # visible in reporting.
                    git_controller.rollback(c['branch_name'], c['worktree_path'])
                    c['eval_passed'] = False
                    c['failure_category'] = "conflict"
                    c['error_msg'] = str(e)
                    c.pop('_approval_request_id', None)
                    continue

                # A candidate that DID commit something real, but whose
                # rebased result contributes nothing beyond the current
                # base, is a duplicate of a candidate that already
                # finalized earlier this same generation (MockLLM in
                # particular can generate identical diffs for distinct
                # candidates) - git's rebase silently drops such a commit
                # as "already applied" rather than erroring, which is
                # exactly why new_tip can equal old_tip here even though
                # this candidate is not the harmless empty-diff case.
                # Counting this as a "success" would credit it for
                # publishing nothing, and duplicate_checker never catches
                # it either, since memory is only written after the whole
                # generation.
                if had_real_commit and git_controller.tree_sha(new_tip) == git_controller.tree_sha(old_tip):
                    git_controller.rollback(c['branch_name'], c['worktree_path'])
                    c['eval_passed'] = False
                    c['failure_category'] = "no_op_after_rebase"
                    c['error_msg'] = (
                        "rebase produced no change relative to the current base - likely a duplicate "
                        "of a candidate that already merged earlier this generation"
                    )
                    c.pop('_approval_request_id', None)
                    continue

                final_score = c['final_score']
                last_subset = c.get('last_subset')

                # The base moved since phase 1 scored this candidate (an
                # earlier candidate in this same generation finalized in
                # between) - the rebase changed what its diff actually
                # produces, so the phase-1 score no longer verifiably
                # describes the code about to be published. Re-run it for
                # real rather than trusting a score computed against
                # different code.
                if c.get('base_commit_at_eval') != old_tip:
                    candidate_logger = bind(self.logger, candidate_id=c['id'])
                    candidate_logger.info(
                        f"Candidate {c['id']}: base moved since evaluation ({c.get('base_commit_at_eval')} -> "
                        f"{old_tip}) - re-evaluating the rebased result before finalizing."
                    )
                    rebased_script_path = os.path.join(c['worktree_path'], "candidate_script.py")
                    reval_out_dir = tempfile.mkdtemp(prefix="autoresearch-reval-")
                    try:
                        reval = _run_eval_stages(
                            rebased_script_path, reval_out_dir, os.path.join(reval_out_dir, "predictions.jsonl"),
                            eval_stages, sandbox, evaluator, metrics_calculator, failure_analyzer, truth,
                            candidate_logger,
                        )
                    finally:
                        shutil.rmtree(reval_out_dir, ignore_errors=True)

                    c['metrics'] = dict(c['metrics'], **reval['all_metrics'])
                    c['final_score'] = reval['final_score']
                    c['last_subset'] = reval['last_subset']
                    c['total_execution_time'] = c.get('total_execution_time', 0.0) + reval['total_execution_time']
                    final_score = reval['final_score']
                    last_subset = reval['last_subset']

                    if not reval['eval_passed']:
                        git_controller.rollback(c['branch_name'], c['worktree_path'])
                        c['eval_passed'] = False
                        c['failure_category'] = reval['failure_category']
                        c['error_msg'] = f"re-evaluation after rebase (base moved) failed: {reval['error_msg']}"
                        c['traceback'] = reval['traceback_text']
                        c.pop('_approval_request_id', None)
                        continue

                # Re-check the baseline against whatever it is RIGHT NOW,
                # under the same lock that serializes finalization - an
                # earlier candidate finalized earlier in this very loop may
                # have already raised it since phase 1 scored this one.
                if baseline_store and last_subset is not None:
                    if not baseline_store.passes(last_subset, final_score, min_improvement):
                        git_controller.rollback(c['branch_name'], c['worktree_path'])
                        c['eval_passed'] = False
                        c['failure_category'] = "below_baseline"
                        current_baseline = baseline_store.get(last_subset)
                        c['error_msg'] = (
                            f"score {final_score:.4f} at {last_subset}% no longer beats the current baseline "
                            f"{current_baseline if current_baseline is not None else 0.0:.4f} + {min_improvement} "
                            "(baseline advanced since this candidate was evaluated)"
                        )
                        c.pop('_approval_request_id', None)
                        continue

                try:
                    git_controller.finalize_merge(c['branch_name'], c['worktree_path'], old_tip, new_tip)
                    c['success'] = True
                    if baseline_store and last_subset is not None:
                        baseline_store.update_if_better(last_subset, final_score)
                except MergeConflict as e:
                    # The ref moved between rebase_onto_base and here (a
                    # vanishingly narrow window, but real under enough
                    # concurrency) - same handling as a rebase conflict.
                    git_controller.rollback(c['branch_name'], c['worktree_path'])
                    c['eval_passed'] = False
                    c['failure_category'] = "conflict"
                    c['error_msg'] = str(e)

            c.pop('_approval_request_id', None)

        return evaluated
