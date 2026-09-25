import argparse
import os
import shutil
import signal
import sys
import tempfile
import uuid

from config_schema import ConfigError, load_config
from observability.logging_config import bind, configure_logging, get_logger
from vcs.git_controller import GitController, MergeConflict
from sandbox.executor import SandboxExecutor
from eval.dataset import generate_split, load_truth, write_subset
from eval.baseline import BaselineStore
from eval.pipeline import EvalPipeline
from orchestrator.metrics import calculate_all_metrics

# Phase 2 imports
from memory.db import ExperimentDB
from memory.failure_analysis import analyze_failure
from generation.prompt_builder import PromptBuilder
from generation.patch_generator import PatchGenerator, MockLLMClient, AnthropicClient, LocalLLMClient
from generation.static_check import check_syntax_multi

# Phase 4 imports
from approval.store import ApprovalStore
from approval.gate import request_and_await_approval, resolve_approval_config
from reporting.report_generator import generate_report


def target_is_harness(target_repo_path: str, harness_root: str) -> bool:
    """True if target_repo_path resolves to the harness's own repository root."""
    return os.path.abspath(target_repo_path) == os.path.abspath(harness_root)


def _approval_db_path(config) -> str:
    approval_cfg = config.get('approval', {})
    return approval_cfg.get('db_path', 'approvals.db') if isinstance(approval_cfg, dict) else 'approvals.db'


def run_startup_cleanup(vcs: GitController, approval_store: ApprovalStore, config, logger) -> None:
    """
    Crash recovery: reclaims state left behind by a previous run that was
    killed or crashed before it could roll back/merge its candidates, or
    before an approval request it was awaiting ever timed out on its own.
    Safe to run unconditionally - both operations are no-ops when nothing
    was actually left behind.

    min_age_seconds=timeout_seconds (not 0): git's worktree registry is
    repo-global, so a second orchestrator process started against the same
    repo_path while this one is still running would otherwise see this
    process's own in-flight candidates as "orphaned" and delete them (see
    vcs/git_controller.py:cleanup_orphans's docstring). A candidate
    worktree younger than the approval gate's own timeout could still be
    legitimately in flight; one older than it would already have timed out
    its own approval wait regardless, so reclaiming it is safe either way.
    This narrows that race, it doesn't close it outright - concurrent
    orchestrator runs against the same repo_path are still not fully safe.
    """
    timeout_seconds = resolve_approval_config(config)["timeout_seconds"]

    result = vcs.cleanup_orphans(min_age_seconds=timeout_seconds)
    if result["removed_worktrees"] or result["removed_branches"]:
        logger.info(
            f"Crash recovery: removed {result['removed_worktrees']} orphan worktree(s) and "
            f"{result['removed_branches']} orphan branch(es) left by a previous run."
        )

    timed_out = approval_store.timeout_stale_requests(timeout_seconds)
    if timed_out:
        logger.info(f"Crash recovery: timed out {timed_out} approval request(s) left pending past their deadline.")


def _install_signal_handlers() -> None:
    """
    A bare Ctrl+C (or SIGTERM from an orchestrating process/container
    runtime) should exit promptly and predictably rather than leave a
    Python traceback - any in-progress candidate worktree/branch it leaves
    behind is reclaimed by run_startup_cleanup() on the next invocation, so
    no cleanup needs to happen inline here.
    """
    def _handle(signum, frame):
        print(
            f"\nReceived signal {signum}; exiting. Any in-progress candidate worktrees/branches or "
            "pending approvals will be reclaimed automatically the next time this is run (or via "
            "the 'cleanup' subcommand).",
            file=sys.stderr,
        )
        sys.exit(130 if signum == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def run_cleanup_command(config, logger) -> None:
    """Standalone `cleanup` subcommand: reclaim orphaned state without starting a run."""
    target_cfg = config.get('target', {})
    vcs = GitController(target_cfg.get('repo_path', '.'), base_ref=target_cfg.get('base_ref'))
    approval_store = ApprovalStore(_approval_db_path(config))

    result = vcs.cleanup_orphans()
    logger.info(f"Removed {result['removed_worktrees']} orphan worktree(s) and {result['removed_branches']} orphan branch(es).")

    timeout_seconds = resolve_approval_config(config)["timeout_seconds"]
    timed_out = approval_store.timeout_stale_requests(timeout_seconds)
    logger.info(f"Timed out {timed_out} stale pending approval request(s).")


def main():
    parser = argparse.ArgumentParser(description="Run the core loop")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "cleanup"],
                         help="'run' (default) executes the core loop; 'cleanup' reclaims orphaned "
                              "worktrees/branches and stale pending approvals without starting a run")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--goal", default="Improve the mock candidate script performance", help="The research goal")
    parser.add_argument("--mode", default="sequential", choices=["sequential", "evolutionary"], help="Mode to run the orchestrator in")
    parser.add_argument("--max-iterations", type=int, default=None, help="Sequential mode only: cap on candidates to try")
    parser.add_argument("--target-score", type=float, default=None, help="Sequential mode only: stop early once reached")
    parser.add_argument("--patience", type=int, default=None, help="Sequential mode only: stop after this many iterations with no improvement")
    args = parser.parse_args()

    _install_signal_handlers()

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    configure_logging()
    run_id = uuid.uuid4().hex[:12]
    logger = get_logger("orchestrator.run", run_id=run_id)

    if args.command == "cleanup":
        run_cleanup_command(config, logger)
        sys.exit(0)

    orch_cfg = config.get('orchestrator', {})
    dataset_cfg = config.get('dataset', {})
    eval_cfg = config.get('eval', {})
    target_cfg = config.get('target', {})

    dataset_dir = dataset_cfg.get('path', 'dummy_data')
    # dataset.seed is intentionally NOT defaulted to a fixed value here -
    # leaving it unset lets generate_split pick a fresh, secret, host-only
    # seed (see eval/dataset.py's docstring for why a fixed/public seed lets
    # a candidate regenerate the held-out labels without ever touching
    # truth.json). Only set dataset.seed in config for a reproducible test
    # fixture, never for a real run.
    dataset_paths = generate_split(
        dataset_dir,
        n=dataset_cfg.get('size', 1000),
        seed=dataset_cfg.get('seed'),
        test_frac=dataset_cfg.get('test_frac', 0.25),
    )
    # truth.json is loaded host-side only - it is never mounted into the
    # sandbox (see sandbox/executor.py), so a candidate can never read its
    # own answer key off disk.
    truth = load_truth(dataset_paths['truth'])

    # Initialize components. target.repo_path lets the repo under evolution
    # be a separate checkout from the harness's own repository; it defaults
    # to "." for backward compatibility, which does mean the harness's own
    # repo unless a config sets it explicitly - warn so that's a deliberate
    # choice, not an accident.
    target_repo_path = target_cfg.get('repo_path', '.')
    harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if target_is_harness(target_repo_path, harness_root):
        logger.warning(
            f"target.repo_path resolves to the harness's own repository ({harness_root}). "
            "Evolutionary candidates will be committed directly into the autoresearch_lite codebase "
            "itself. Set target.repo_path in your config to a separate git repository to keep the "
            "harness and the code under evolution independent."
        )
    vcs = GitController(target_repo_path, base_ref=target_cfg.get('base_ref'))
    sandbox = SandboxExecutor(config.get('sandbox', {}), dataset_dir=dataset_dir)
    evaluator = EvalPipeline(eval_cfg)
    baseline_store = BaselineStore(eval_cfg.get('state_path', 'state.json'))
    min_improvement = eval_cfg.get('min_improvement', 0.001)

    # Initialize Phase 2 components
    db = ExperimentDB()
    gen_cfg = config.get('generation', {})
    prompt_builder = PromptBuilder(db, gen_cfg)
    # generation.client defaults to "mock" so the test suite needs no
    # network access or API key. "anthropic" reads ANTHROPIC_API_KEY (via
    # the SDK's own env lookup) - never from config, so a committed config
    # file can never leak a key. "local" talks to an OpenAI-compatible
    # local server (Ollama/vLLM/llama.cpp/...) - no key at all, cost is
    # always $0, but expect a higher malformed-diff retry rate than a
    # frontier model.
    if gen_cfg.get('client') == 'anthropic':
        llm_client = AnthropicClient(model=gen_cfg.get('model', 'claude-sonnet-5'))
    elif gen_cfg.get('client') == 'local':
        llm_client = LocalLLMClient(
            base_url=gen_cfg.get('base_url', 'http://localhost:11434/v1'),
            model=gen_cfg.get('model', 'qwen2.5-coder:32b'),
        )
    else:
        llm_client = MockLLMClient()
    patch_generator = PatchGenerator(llm_client)

    # Phase 4: human-approval gate. resolve_approval_config (inside the gate)
    # defaults to "required" for any missing/malformed approval config - see
    # approval/gate.py.
    approval_store = ApprovalStore(_approval_db_path(config))

    # Crash recovery: reclaim anything a previous, killed/crashed run left
    # behind before this run creates any candidates of its own.
    run_startup_cleanup(vcs, approval_store, config, logger)

    if args.mode == "evolutionary":
        from evolution.population import EvolutionEngine
        engine = EvolutionEngine(
            config=config,
            git_controller=vcs,
            sandbox=sandbox,
            evaluator=evaluator,
            metrics_calculator=calculate_all_metrics,
            failure_analyzer=analyze_failure,
            patch_generator=patch_generator,
            prompt_builder=prompt_builder,
            db=db,
            approval_store=approval_store,
            truth=truth,
            baseline_store=baseline_store,
            run_id=run_id,
            train_path=dataset_paths['train'],
        )
        engine.run(args.goal)
        generate_report(db, approval_store, logger=logger)
        sys.exit(0)

    eval_stages = config.get('eval', {}).get('stages', [])
    # config_schema.py's TargetConfig validator guarantees "candidate_script.py"
    # is always present in this list - the sandbox's Dockerfile CMD always
    # executes that exact filename.
    target_files = target_cfg.get('files', ['candidate_script.py'])

    def run_iteration(candidate_id: str, goal: str):
        candidate_logger = bind(logger, candidate_id=candidate_id)
        candidate_logger.info(f"--- Starting iteration for candidate {candidate_id} ---")

        # 1. VCS branching - each candidate gets its own worktree, so this
        # never touches the caller's main checkout.
        branch_name, worktree_path = vcs.create_branch(candidate_id)
        candidate_logger.info(f"Created branch {branch_name} (worktree: {worktree_path})")

        script_path = os.path.join(worktree_path, "candidate_script.py")
        extra_file_paths = [os.path.join(worktree_path, f) for f in target_files if f != "candidate_script.py"]

        # 2. Phase 2 Generation - apply the patch inside the candidate's own
        # worktree (cwd=worktree_path), not the shared main checkout.
        candidate_logger.info("Building prompt...")
        prompt = prompt_builder.build_prompt(goal)

        candidate_logger.info("Generating and applying patch...")
        apply_success, diff = patch_generator.generate_and_apply(
            prompt, target_files, cwd=worktree_path, logger=candidate_logger
        )
        # Only AnthropicClient/LocalLLMClient set this - MockLLMClient makes
        # no API calls, so there's no cost/tokens to attribute.
        generation_usage = getattr(patch_generator.llm_client, "last_usage", {}) or {}

        if not apply_success:
            candidate_logger.warning("Patch application failed (malformed diff). Rejecting candidate.")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Prompt generated malformed diff",
                metrics={},
                outcome="failure",
                failure_reason="Malformed diff rejected by git apply.",
                # The real git-apply diagnostic (e.g. "error: patch failed:
                # file.py:10"), not just this generic label - without it,
                # every malformed-diff failure record looked identical
                # regardless of what was actually wrong with that diff.
                traceback=patch_generator.last_apply_error or None,
            )
            vcs.rollback(branch_name, worktree_path)
            return False, 0.0

        vcs.commit_patch(worktree_path, f"Add candidate {candidate_id}")

        # 3. Static Analysis Pre-check
        candidate_logger.info("Running static analysis...")
        syntax_ok, syntax_err = check_syntax_multi([script_path] + extra_file_paths)
        if not syntax_ok:
            candidate_logger.warning(f"Static check failed: {syntax_err}")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Prompt generated syntax error",
                metrics={},
                outcome="failure",
                failure_reason=syntax_err,
                traceback=syntax_err,
            )
            vcs.rollback(branch_name, worktree_path)
            return False, 0.0

        # 4. Execute in sandbox once per progressive-scaling stage, so each
        # stage's score reflects that stage's own dataset subset. Each run
        # writes predictions.jsonl to out_dir - that, scored against
        # held-out truth, is the only real score; anything the candidate
        # prints is a diagnostic at best (see eval/pipeline.py).
        # Deliberately a fresh host tempdir, NOT a path under worktree_path:
        # this is where the sandbox's rw bind mount points, and the
        # worktree's contents are exactly what a candidate's diff controls.
        # A path inside the worktree can be turned into a symlink (e.g.
        # .eval_out -> ../../dummy_data, giving the sandbox rw access to the
        # dataset directory - or worse, .git/hooks, a route to host code
        # execution). vcs/diff_guard.py already refuses such a diff outright,
        # but a tempdir the diff never gets a chance to name is immune to
        # this regardless of any gap in that guard. Removed in the `finally`
        # below - nothing after the stage loop needs it.
        out_dir = tempfile.mkdtemp(prefix="autoresearch-out-")
        pred_path = os.path.join(out_dir, "predictions.jsonl")
        # Separate from out_dir on purpose: out_dir is mounted as a whole
        # rw directory (/app/out), so a file written directly into it would
        # be visible (and writable) to the candidate. subset_dir is never
        # mounted itself - only the one per-stage file inside it that
        # train_path_override names explicitly.
        subset_dir = tempfile.mkdtemp(prefix="autoresearch-subset-")

        eval_passed = True
        final_score = 0.0
        last_subset = None
        metrics = {}
        execution_result = None
        has_failure_flags = False

        try:
            for stage in eval_stages:
                subset = stage['subset_percentage']
                threshold = stage['threshold']

                # Wipe any predictions left by the previous stage - without
                # this, a stage whose script exits 0 without writing
                # predictions.jsonl would be scored against the PRIOR
                # stage's file instead of failing with "no predictions
                # written".
                if os.path.exists(pred_path):
                    os.remove(pred_path)

                # Host-selected subset, not just the SUBSET_PERCENTAGE env
                # var - without this, every stage mounted the SAME full
                # train.jsonl, and a candidate that simply ignored the env
                # var (or trained on the full file regardless of what it
                # claimed) would never be caught.
                subset_train_path = write_subset(
                    dataset_paths['train'], os.path.join(subset_dir, "train_subset.jsonl"), subset,
                )

                candidate_logger.info(f"Running in sandbox (subset={subset}%)...")
                execution_result = sandbox.run_candidate(
                    script_path, env_vars={"SUBSET_PERCENTAGE": str(subset)}, out_dir=out_dir,
                    extra_files=extra_file_paths or None, train_path_override=subset_train_path,
                )

                if execution_result['timeout']:
                    candidate_logger.warning("Execution TIMED OUT")

                metrics = calculate_all_metrics(execution_result)
                candidate_logger.info(f"Plugin Metrics: {metrics}")

                stage_success, final_score, stage_mismatch = evaluator.evaluate_stage(
                    execution_result, subset, threshold, pred_path, truth, logger=candidate_logger
                )
                has_failure_flags = has_failure_flags or stage_mismatch
                last_subset = subset
                if not stage_success:
                    eval_passed = False
                    break
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(subset_dir, ignore_errors=True)

        if eval_passed:
            candidate_logger.info("Candidate passed all evaluation stages.")

        # 4b. Baseline gate - clearing every stage's absolute threshold is
        # not enough to merge; the final stage's score must also beat the
        # best score ever actually merged for that stage. Without this, a
        # candidate that regresses relative to what's already in place can
        # still merge as long as it clears the (fixed) threshold.
        #
        # baseline_score is None (not 0.0) when nothing has ever merged at
        # this stage - and so is delta, which feeds should_auto_approve's
        # "improvement over baseline" criterion (see approval/gate.py). A
        # candidate with no baseline to compare against must always fall
        # through to human review, never auto-merge just because
        # final_score - 0.0 trivially clears the auto-approve threshold.
        baseline_score = baseline_store.get(last_subset) if last_subset is not None else None
        delta = (final_score - baseline_score) if baseline_score is not None else None
        below_baseline = False
        if eval_passed and last_subset is not None:
            if not baseline_store.passes(last_subset, final_score, min_improvement):
                below_baseline = True
                eval_passed = False
                candidate_logger.info(
                    f"Candidate {candidate_id} scored {final_score:.4f} at {last_subset}%, which does not beat "
                    f"baseline {baseline_score if baseline_score is not None else 0.0:.4f} + "
                    f"min_improvement {min_improvement}. Rejecting despite clearing the absolute threshold."
                )

        metrics['baseline_score'] = baseline_score if baseline_score is not None else 0.0
        metrics['delta'] = delta
        # Consumed by approval/gate.py's should_auto_approve as the
        # require_no_failure_flags criterion - currently the only known
        # flag is a candidate's printed SCORE claim not matching its real,
        # scored result (see eval/pipeline.py's reward-hacking guard).
        metrics['score_claim_mismatch'] = has_failure_flags
        if generation_usage:
            metrics['generation_input_tokens'] = generation_usage.get('input_tokens', 0)
            metrics['generation_output_tokens'] = generation_usage.get('output_tokens', 0)
            metrics['generation_cost_usd'] = generation_usage.get('estimated_cost_usd', 0.0)

        # 5. Analyze failure and log to Memory
        if below_baseline:
            category, error_text, traceback_text = "below_baseline", (
                f"score {final_score:.4f} did not beat baseline {baseline_score:.4f} + {min_improvement}"
            ), ""  # no real trace applies - this is a threshold comparison, not a crash
        else:
            category, error_text, traceback_text = analyze_failure(execution_result, eval_passed)

        merged = False
        if not eval_passed:
            candidate_logger.info(f"Candidate {candidate_id} failed ({category}). Rolling back.")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Generated patch failed",
                metrics=metrics,
                outcome="failure",
                failure_reason=error_text,
                traceback=traceback_text or None,
            )
            vcs.rollback(branch_name, worktree_path)
        else:
            # 6. Human-approval gate - genuinely blocks the merge path.
            # Only "approved", "auto_approved" (approval.auto_approve's
            # criteria cleared - see approval/gate.py), or "skipped" (gate
            # explicitly disabled) may proceed to merge; "rejected" and
            # "timed_out" roll back.
            candidate_logger.info(f"Candidate {candidate_id} passed evaluation with score {final_score:.4f}. Awaiting approval...")
            decision = request_and_await_approval(
                approval_store, candidate_id, goal, diff, final_score, metrics, config
            )

            if decision in ("approved", "auto_approved", "skipped"):
                candidate_logger.info(f"Candidate {candidate_id} approved ({decision}). Merging.")
                try:
                    vcs.merge(branch_name, worktree_path)
                    merged = True
                    if last_subset is not None:
                        baseline_store.update_if_better(last_subset, final_score)
                    db.store_experiment(
                        hypothesis=goal,
                        diff=diff,
                        rationale="Generated patch passed evaluation and approval",
                        metrics=metrics,
                        outcome="success"
                    )
                except MergeConflict as e:
                    # Rebasing onto the current base failed - record before
                    # merging, not after, so a failed merge is never
                    # recorded as "success". Distinct from "failure" since
                    # it's a population-staleness signal, not a code defect.
                    candidate_logger.warning(f"Candidate {candidate_id} could not be merged (rebase conflict): {e}")
                    db.store_experiment(
                        hypothesis=goal,
                        diff=diff,
                        rationale="Generated patch passed evaluation and approval but failed to rebase cleanly",
                        metrics=metrics,
                        outcome="conflict",
                        failure_reason=str(e)
                    )
                    vcs.rollback(branch_name, worktree_path)
            else:
                candidate_logger.info(f"Candidate {candidate_id} was not merged (approval decision: {decision}). Rolling back.")
                db.store_experiment(
                    hypothesis=goal,
                    diff=diff,
                    rationale="Generated patch passed evaluation but was not approved",
                    metrics=metrics,
                    outcome="held",
                    failure_reason=f"approval_decision={decision}"
                )
                vcs.rollback(branch_name, worktree_path)

        candidate_logger.info(f"--- Finished iteration for candidate {candidate_id} ---\n")
        return merged, final_score

    max_iterations = args.max_iterations or orch_cfg.get('max_iterations', 1)
    target_score = args.target_score if args.target_score is not None else orch_cfg.get('target_score', 1.0)
    patience = args.patience or orch_cfg.get('patience', max_iterations)

    best_score = 0.0
    any_success = False
    iterations_since_improvement = 0

    for i in range(max_iterations):
        candidate_id = uuid.uuid4().hex[:8]
        success, final_score = run_iteration(candidate_id, args.goal)

        if success:
            any_success = True
        if final_score > best_score:
            best_score = final_score
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

        if any_success and best_score >= target_score:
            logger.info(f"Target score {target_score} reached (best={best_score:.4f}). Stopping.")
            break
        if iterations_since_improvement >= patience:
            logger.info(f"No improvement in {patience} iterations. Stopping early.")
            break

    generate_report(db, approval_store, logger=logger)

    if not any_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
