import argparse
import os
import sys
import uuid

import yaml

from vcs.git_controller import GitController
from sandbox.executor import SandboxExecutor
from eval.dataset import generate_dataset
from eval.pipeline import EvalPipeline
from orchestrator.metrics import calculate_all_metrics

# Phase 2 imports
from memory.db import ExperimentDB
from memory.failure_analysis import analyze_failure
from generation.prompt_builder import PromptBuilder
from generation.patch_generator import PatchGenerator, MockLLMClient
from generation.static_check import check_syntax


def main():
    parser = argparse.ArgumentParser(description="Run the core loop")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--goal", default="Improve the mock candidate script performance", help="The research goal")
    parser.add_argument("--mode", default="sequential", choices=["sequential", "evolutionary"], help="Mode to run the orchestrator in")
    parser.add_argument("--max-iterations", type=int, default=None, help="Sequential mode only: cap on candidates to try")
    parser.add_argument("--target-score", type=float, default=None, help="Sequential mode only: stop early once reached")
    parser.add_argument("--patience", type=int, default=None, help="Sequential mode only: stop after this many iterations with no improvement")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    orch_cfg = config.get('orchestrator', {})
    dataset_cfg = config.get('dataset', {})

    dataset_path = dataset_cfg.get('path', 'dummy_data/dataset.jsonl')
    generate_dataset(dataset_path, n_samples=dataset_cfg.get('size', 1000), seed=dataset_cfg.get('seed', 42))

    # Initialize components
    vcs = GitController()
    sandbox = SandboxExecutor(config.get('sandbox', {}), dataset_path=dataset_path)
    evaluator = EvalPipeline(config.get('eval', {}))

    # Initialize Phase 2 components
    db = ExperimentDB()
    prompt_builder = PromptBuilder(db, config.get('generation', {}))
    llm_client = MockLLMClient()
    patch_generator = PatchGenerator(llm_client)

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
            db=db
        )
        engine.run(args.goal)
        sys.exit(0)

    eval_stages = config.get('eval', {}).get('stages', [])

    def run_iteration(candidate_id: str, goal: str):
        print(f"--- Starting iteration for candidate {candidate_id} ---")

        # 1. VCS branching - each candidate gets its own worktree, so this
        # never touches the caller's main checkout.
        branch_name, worktree_path = vcs.create_branch(candidate_id)
        print(f"Created branch {branch_name} (worktree: {worktree_path})")

        script_path = os.path.join(worktree_path, "candidate_script.py")

        # 2. Phase 2 Generation - apply the patch inside the candidate's own
        # worktree (cwd=worktree_path), not the shared main checkout.
        print("Building prompt...")
        prompt = prompt_builder.build_prompt(goal)

        print("Generating and applying patch...")
        apply_success, diff = patch_generator.generate_and_apply(prompt, "candidate_script.py", cwd=worktree_path)

        if not apply_success:
            print("Patch application failed (malformed diff). Rejecting candidate.")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Prompt generated malformed diff",
                metrics={},
                outcome="failure",
                failure_reason="Malformed diff rejected by git apply."
            )
            vcs.rollback(branch_name, worktree_path)
            return False

        vcs.commit_patch(worktree_path, f"Add candidate {candidate_id}")

        # 3. Static Analysis Pre-check
        print("Running static analysis...")
        syntax_ok, syntax_err = check_syntax(script_path)
        if not syntax_ok:
            print(f"Static check failed: {syntax_err}")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Prompt generated syntax error",
                metrics={},
                outcome="failure",
                failure_reason=syntax_err
            )
            vcs.rollback(branch_name, worktree_path)
            return False

        # 4. Execute in sandbox once per progressive-scaling stage, so each
        # stage's score reflects that stage's own dataset subset.
        success = True
        final_score = 0.0
        metrics = {}
        execution_result = None

        for stage in eval_stages:
            subset = stage['subset_percentage']
            threshold = stage['threshold']

            print(f"Running in sandbox (subset={subset}%)...")
            execution_result = sandbox.run_candidate(script_path, env_vars={"SUBSET_PERCENTAGE": str(subset)})

            if execution_result['timeout']:
                print("Execution TIMED OUT")

            metrics = calculate_all_metrics(execution_result)
            print(f"Plugin Metrics: {metrics}")

            stage_success, final_score = evaluator.evaluate_stage(execution_result, subset, threshold)
            if not stage_success:
                success = False
                break

        if success:
            print("Candidate passed all evaluation stages.")

        # 5. Analyze failure and log to Memory
        category, error_text = analyze_failure(execution_result, success)

        if success:
            print(f"Candidate {candidate_id} succeeded with score {final_score:.4f}. Merging.")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Generated patch passed evaluation",
                metrics=metrics,
                outcome="success"
            )
            vcs.merge(branch_name, worktree_path)
        else:
            print(f"Candidate {candidate_id} failed ({category}). Rolling back.")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Generated patch failed",
                metrics=metrics,
                outcome="failure",
                failure_reason=error_text
            )
            vcs.rollback(branch_name, worktree_path)

        print(f"--- Finished iteration for candidate {candidate_id} ---\n")
        return success, final_score

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
            print(f"Target score {target_score} reached (best={best_score:.4f}). Stopping.")
            break
        if iterations_since_improvement >= patience:
            print(f"No improvement in {patience} iterations. Stopping early.")
            break

    if not any_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
