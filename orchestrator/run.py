import argparse
import yaml
import uuid
import os
import sys

from vcs.git_controller import GitController
from sandbox.executor import SandboxExecutor
from eval.pipeline import EvalPipeline
from orchestrator.candidate import generate_candidate
from orchestrator.metrics import calculate_all_metrics

def main():
    parser = argparse.ArgumentParser(description="Run the core loop")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    vcs = GitController()
    sandbox = SandboxExecutor(config.get('sandbox', {}))
    evaluator = EvalPipeline(config.get('eval', {}))

    # Core loop skeleton: propose candidate -> execute -> evaluate -> record result

    # State scoping using closure-like structure for the iteration
    def run_iteration(candidate_id: str):
        print(f"--- Starting iteration for candidate {candidate_id} ---")

        # 1. Propose candidate
        script_code = generate_candidate()

        # 2. VCS Branching
        branch_name = vcs.create_branch(candidate_id)
        print(f"Created branch {branch_name}")

        # Apply patch to disk
        script_path = "candidate_script.py"
        with open(script_path, "w") as f:
            f.write(script_code)

        vcs.commit_patch(f"Add candidate {candidate_id}")

        # 3. Execute in Sandbox
        print("Running in sandbox...")
        # Resolve absolute path for volume mount
        abs_script_path = os.path.abspath(script_path)
        execution_result = sandbox.run_candidate(abs_script_path)

        print(f"Execution finished. Exit code: {execution_result['exit_code']}")
        if execution_result['timeout']:
            print("Execution TIMED OUT")

        # Optional: calculate plugin metrics
        metrics = calculate_all_metrics(execution_result)
        print(f"Plugin Metrics: {metrics}")

        # 4. Evaluate Proxy Data
        success, final_score = evaluator.evaluate(execution_result)

        # 5. VCS Rollback or Merge
        if success:
            print(f"Candidate {candidate_id} succeeded with score {final_score:.2f}. Merging.")
            vcs.merge_and_push(branch_name)
        else:
            print(f"Candidate {candidate_id} failed. Rolling back.")
            vcs.rollback(branch_name)

        # Clean up untracked script if it exists
        if os.path.exists(script_path):
            os.remove(script_path)

        print(f"--- Finished iteration for candidate {candidate_id} ---\n")
        return success

    # Just running one dummy loop iteration for this phase
    candidate_id = uuid.uuid4().hex[:8]
    success = run_iteration(candidate_id)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
