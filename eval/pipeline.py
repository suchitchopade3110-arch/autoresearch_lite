import re
from typing import Any, Dict, List, Tuple

SCORE_PATTERN = re.compile(r"SCORE:\s*([-+]?\d*\.?\d+)")


class EvalPipeline:
    """
    Progressive-scaling evaluator. Both the sequential orchestrator loop and
    the concurrent evolutionary scheduler execute the candidate once per
    stage (passing SUBSET_PERCENTAGE) and call evaluate_stage() with that
    stage's real execution_result - this class no longer mocks or
    randomizes a score, it parses the candidate's actual reported "SCORE:"
    line for the subset it was actually run against.
    """
    def __init__(self, config: Dict[str, Any]):
        self.stages = config.get('stages', [])
        self.correlation_log: List[Tuple[float, float]] = []

    def evaluate_stage(self, execution_result: Dict[str, Any], subset_percentage: int, threshold: float) -> Tuple[bool, float]:
        """Evaluates a single stage's real execution result. Returns (success, score)."""
        if execution_result['exit_code'] != 0 or execution_result.get('timeout', False):
            return False, 0.0

        score = self._parse_score(execution_result)
        success = score >= threshold

        print(f"Stage {subset_percentage}%: Score={score:.4f}, Threshold={threshold}")
        if not success:
            print(f"Candidate failed at {subset_percentage}% subset.")

        return success, score

    def evaluate(self, execution_result: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluates a single execution result against every stage's threshold.
        Kept for callers that only run the candidate once; prefer driving
        one sandbox execution per stage (see orchestrator/run.py and
        evolution/scheduler.py) so each stage's score reflects that stage's
        own subset rather than reusing one run's output for every stage.
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

        if 1 in stage_scores and 100 in stage_scores:
            self.correlation_log.append((stage_scores[1], stage_scores[100]))

        print("Candidate passed all evaluation stages.")
        return True, stage_scores.get(100, 0.0)

    @staticmethod
    def _parse_score(execution_result: Dict[str, Any]) -> float:
        match = SCORE_PATTERN.search(execution_result.get('stdout', ''))
        if not match:
            return 0.0
        return float(match.group(1))
