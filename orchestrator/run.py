import argparse
import yaml
import uuid
import os
import sys

from vcs.git_controller import GitController
from sandbox.executor import SandboxExecutor
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
    # For Phase 2, we could specify the target file or goal via CLI, but hardcode for now
    parser.add_argument("--goal", default="Improve the mock candidate script performance", help="The research goal")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    vcs = GitController()
    sandbox = SandboxExecutor(config.get('sandbox', {}))
    evaluator = EvalPipeline(config.get('eval', {}))

    # Initialize Phase 2 components
    db = ExperimentDB()
    prompt_builder = PromptBuilder(db, config.get('generation', {}))
    llm_client = MockLLMClient()
    patch_generator = PatchGenerator(llm_client)

    def run_iteration(candidate_id: str, goal: str):
        print(f"--- Starting iteration for candidate {candidate_id} ---")

        # 1. VCS Branching (do this first so patch application hits the branch)
        branch_name = vcs.create_branch(candidate_id)
        print(f"Created branch {branch_name}")

        script_path = "candidate_script.py"

        # 2. Phase 2 Generation
        print("Building prompt...")
        prompt = prompt_builder.build_prompt(goal)

        print("Generating and applying patch...")
        apply_success, diff = patch_generator.generate_and_apply(prompt, script_path)

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
            vcs.rollback(branch_name)
            return False

        vcs.commit_patch(f"Add candidate {candidate_id}")

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
            vcs.rollback(branch_name)
            return False

        # 4. Execute in Sandbox
        print("Running in sandbox...")
        abs_script_path = os.path.abspath(script_path)
        execution_result = sandbox.run_candidate(abs_script_path)

        print(f"Execution finished. Exit code: {execution_result['exit_code']}")
        if execution_result['timeout']:
            print("Execution TIMED OUT")

        metrics = calculate_all_metrics(execution_result)
        print(f"Plugin Metrics: {metrics}")

        # 5. Evaluate Proxy Data
        success, final_score = evaluator.evaluate(execution_result)

        # 6. Analyze failure and log to Memory
        category, error_text = analyze_failure(execution_result, success)

        if success:
            print(f"Candidate {candidate_id} succeeded with score {final_score:.2f}. Merging.")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Generated patch passed evaluation",
                metrics=metrics,
                outcome="success"
            )
            vcs.merge_and_push(branch_name)
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
            vcs.rollback(branch_name)

        print(f"--- Finished iteration for candidate {candidate_id} ---\n")
        return success

    candidate_id = uuid.uuid4().hex[:8]
    success = run_iteration(candidate_id, args.goal)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
