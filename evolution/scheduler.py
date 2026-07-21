import concurrent.futures
import threading
import os
import shutil
from typing import List, Dict, Any, Callable

class ConcurrentScheduler:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.git_lock = threading.Lock()

    def execute_generation(self,
                          candidates: List[Dict[str, Any]],
                          eval_stages: List[Dict[str, Any]],
                          git_controller,
                          sandbox,
                          evaluator,
                          metrics_calculator,
                          failure_analyzer) -> List[Dict[str, Any]]:
        results = []

        def process_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
            c_id = candidate['id']
            diff = candidate['diff']
            goal = candidate.get('goal', 'Optimization goal')

            import tempfile
            temp_dir = tempfile.mkdtemp(prefix=f"sandbox_exec_{c_id}_")
            script_path = os.path.join(temp_dir, "candidate_script.py")

            with open(script_path, "w") as f:
                f.write("\n")

            branch_name = None

            try:
                with self.git_lock:
                    branch_name = git_controller.create_branch(c_id)

                    if diff.strip():
                        patch_file = f"temp_{c_id}.patch"
                        with open(patch_file, "w") as f:
                            f.write(diff)

                        import subprocess
                        subprocess.run(["git", "apply", patch_file], check=True, capture_output=True)
                        if os.path.exists(patch_file):
                            os.remove(patch_file)

                    if os.path.exists("candidate_script.py"):
                        with open("candidate_script.py", "r") as f:
                            patched_code = f.read()
                    else:
                        patched_code = "\n"

                    git_controller.commit_patch(f"Add candidate {c_id}")

                    git_controller.repo.heads[git_controller.original_branch].checkout()

                with open(script_path, "w") as f:
                    f.write(patched_code)

                final_score = 0.0
                all_metrics = {}
                success = True
                failure_category = "success"
                error_msg = ""
                total_execution_time = 0.0

                for stage in eval_stages:
                    subset = stage['subset_percentage']
                    threshold = stage['threshold']

                    env = {"SUBSET_PERCENTAGE": str(subset)}
                    exec_result = sandbox.run_candidate(script_path, env_vars=env)
                    total_execution_time += exec_result.get('execution_time', 0.0)

                    exec_result['execution_time'] = total_execution_time

                    stage_success, stage_score = evaluator.evaluate_stage(exec_result, subset, threshold)

                    if not stage_success:
                        success = False
                        final_score = stage_score
                        cat, msg = failure_analyzer(exec_result, False)
                        failure_category = cat
                        error_msg = msg
                        all_metrics = metrics_calculator(exec_result)
                        break

                    final_score = stage_score
                    all_metrics = metrics_calculator(exec_result)

                with self.git_lock:
                    if success:
                        git_controller.merge_and_push(branch_name)
                    else:
                        git_controller.rollback(branch_name)

                candidate['success'] = success
                candidate['final_score'] = final_score
                candidate['metrics'] = all_metrics
                candidate['failure_category'] = failure_category
                candidate['error_msg'] = error_msg
                candidate['total_execution_time'] = total_execution_time
                return candidate

            except Exception as e:
                with self.git_lock:
                    if branch_name:
                        try:
                            active_branches = [h.name for h in git_controller.repo.heads]
                        except AttributeError:
                            active_branches = git_controller.repo.heads.keys()
                        if branch_name in active_branches:
                            git_controller.rollback(branch_name)
                candidate['success'] = False
                candidate['failure_category'] = "runtime"
                candidate['error_msg'] = str(e)
                candidate['metrics'] = {}
                candidate['total_execution_time'] = 0.0
                return candidate
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_candidate, c) for c in candidates]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results