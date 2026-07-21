from typing import Dict, Any, Tuple

def analyze_failure(execution_result: Dict[str, Any], evaluation_success: bool) -> Tuple[str, str]:
    """
    Analyzes the execution result and evaluation success to categorize the failure.
    Returns (category, error_text).
    Categories: syntax, runtime, timeout, resource-limit, metric-regression, success
    """
    if execution_result.get('timeout', False):
        return "timeout", execution_result.get('stderr', 'Execution timed out.')

    exit_code = execution_result.get('exit_code', 0)
    stderr = execution_result.get('stderr', '')

    if exit_code != 0:
        if exit_code == 137 or "Killed" in stderr:
            return "resource-limit", f"Process killed (likely OOM). Exit code: {exit_code}.\n{stderr}"

        if "SyntaxError:" in stderr or "IndentationError:" in stderr:
            return "syntax", stderr

        return "runtime", stderr

    if not evaluation_success:
        return "metric-regression", "Candidate failed proxy evaluation thresholds."

    return "success", ""
