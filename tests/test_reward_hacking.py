import json
import os
import subprocess
import sys

from eval.pipeline import EvalPipeline

CONFIG = {"stages": [{"subset_percentage": 100, "threshold": 0.5}]}

# The entire "solution" is a printed claim - no prediction file, no attempt
# at the actual task. If the pipeline ever trusts stdout for the score,
# this is the trivial exploit that gets a perfect merge for free.
REWARD_HACK_SCRIPT = 'print("SCORE: 1.0")\n'


def test_printed_score_claim_is_never_trusted(tmp_dir):
    """
    Permanent regression test - must never be deleted or skipped. A
    candidate that never engages with the actual task, and simply prints a
    perfect score claim, must not be able to buy a merge that way. The eval
    pipeline must score real predictions against held-out truth, never
    anything the candidate writes to stdout.
    """
    script_path = os.path.join(tmp_dir, "candidate_script.py")
    with open(script_path, "w") as f:
        f.write(REWARD_HACK_SCRIPT)

    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    assert "SCORE: 1.0" in result.stdout  # sanity: the candidate did make the claim

    pred_path = os.path.join(tmp_dir, "predictions.jsonl")  # the script wrote nothing here
    truth = {"0": 1, "1": 0, "2": 1, "3": 0}

    pipeline = EvalPipeline(CONFIG)
    score, reason = pipeline.score_predictions(pred_path, truth)

    assert score == 0.0
    assert reason  # a real reason string, not silent 0.0
