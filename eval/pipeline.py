from typing import List, Dict, Any, Tuple
import random

class EvalPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.stages = config.get('stages', [])
        self.dataset_path = config.get('path', 'dummy_data/')
        self.correlation_log: List[Tuple[float, float]] = []

    def evaluate_stage(self, execution_result: Dict[str, Any], subset_percentage: int, threshold: float) -> Tuple[bool, float]:
        """Evaluates a single stage. Returns (success, score)."""
        if execution_result['exit_code'] != 0 or execution_result.get('timeout', False):
            return False, 0.0

        score = self._mock_score(subset_percentage, execution_result)
        success = score >= threshold

        print(f"Stage {subset_percentage}%: Score={score:.2f}, Threshold={threshold}")
        if not success:
            print(f"Candidate failed at {subset_percentage}% subset.")

        return success, score

    def evaluate(self, execution_result: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluates the candidate using progressive scaling.
        Returns a tuple of (success, final_score).
        """
        if execution_result['exit_code'] != 0 or execution_result['timeout']:
            return False, 0.0

        stage_scores = {}
        for stage in self.stages:
            subset = stage['subset_percentage']
            threshold = stage['threshold']

            success, score = self.evaluate_stage(execution_result, subset, threshold)
            stage_scores[subset] = score

            if not success:
                return False, score

        # If we passed all stages, track correlation between 1% and 100% stages
        if 1 in stage_scores and 100 in stage_scores:
            self.correlation_log.append((stage_scores[1], stage_scores[100]))

        print("Candidate passed all evaluation stages.")
        return True, stage_scores.get(100, 0.0)

    def _mock_score(self, subset: int, execution_result: Dict[str, Any]) -> float:
        """Generates a dummy score for testing purposes."""
        # Just to make it deterministic if stdout contains a target score
        stdout = execution_result.get('stdout', '')
        if "MOCK_SCORE:" in stdout:
            try:
                score_str = stdout.split("MOCK_SCORE:")[1].split()[0]
                return float(score_str)
            except Exception:
                pass

        # Otherwise return a random passing score with a small chance to fail
        return random.uniform(0.5, 1.0)
