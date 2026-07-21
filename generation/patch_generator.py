from abc import ABC, abstractmethod
import os
import subprocess
import tempfile
from typing import Optional

class LLMClient(ABC):
    @abstractmethod
    def generate_diff(self, prompt: str, target_file: str) -> str:
        """Generates a valid unified diff for the target file based on the prompt."""
        pass

class MockLLMClient(LLMClient):
    """
    A mock LLM client for testing. Returns a deterministic patch that
    replaces the (blank placeholder) target file with a script that
    actually reads the mounted dataset (DATASET_PATH/SUBSET_PERCENTAGE) and
    reports a real accuracy, rather than a hardcoded score - so running it
    end-to-end exercises the real per-stage evaluation pipeline instead of
    faking the number.
    TODO: Phase 3 - Implement a real LLMClient (e.g. OpenAIClient, AnthropicClient).
    """
    def generate_diff(self, prompt: str, target_file: str) -> str:
        return f"""--- a/{target_file}
+++ b/{target_file}
@@ -1 +1,30 @@
-
+import json
+import os
+import random
+
+DATASET_PATH = os.environ.get("DATASET_PATH", "/app/data/dataset.jsonl")
+SUBSET_PERCENTAGE = float(os.environ.get("SUBSET_PERCENTAGE", "100"))
+SEED = int(os.environ.get("DATASET_SEED", "42"))
+
+
+def load_subset():
+    with open(DATASET_PATH) as f:
+        rows = [json.loads(line) for line in f if line.strip()]
+    rng = random.Random(SEED)
+    order = list(range(len(rows)))
+    rng.shuffle(order)
+    k = max(1, int(len(rows) * SUBSET_PERCENTAGE / 100))
+    idx = order[:k]
+    return [rows[i] for i in idx]
+
+
+def predict(x1, x2):
+    return 1 if (2.0 * x1 - 1.0 * x2) > 0 else 0
+
+
+def main():
+    rows = load_subset()
+    correct = sum(1 for r in rows if predict(r["x1"], r["x2"]) == r["label"])
+    accuracy = correct / len(rows) if rows else 0.0
+    print(f"SCORE: {{accuracy:.4f}}", flush=True)
+
+
+if __name__ == "__main__":
+    main()
"""

def validate_and_apply_patch(diff_content: str, cwd: Optional[str] = None, dry_run: bool = False) -> bool:
    """
    Validates a patch by attempting to apply it cleanly, then applies it
    unless `dry_run` is set. `cwd` is the git working tree the patch should
    be applied against (a candidate's own worktree) - defaults to the
    caller's current directory if omitted, matching the pre-worktree
    behavior.

    Use dry_run=True for a validity check with no side effects (e.g.
    checking a candidate's diff is even applicable before scheduling it) -
    without it, every successful call mutates `cwd`'s working tree for
    real, so checking the same diff twice would fail the second time.
    Returns True if successful, False otherwise.
    """
    fd, patch_file = tempfile.mkstemp(suffix=".patch")
    try:
        # newline='' disables Python's platform line-ending translation, so
        # the diff's own '\n' bytes are written through unchanged. Without
        # it, the default text mode on Windows rewrites every '\n' to
        # '\r\n' - including inside the patch file's own hunk lines - which
        # confuses git apply's line-based hunk parser and causes it to
        # silently apply only part of the hunk (observed: the last two
        # lines of a 30-line hunk went missing, with no error at all).
        with os.fdopen(fd, "w", newline='') as f:
            f.write(diff_content)

        subprocess.run(["git", "apply", "--check", patch_file], check=True, capture_output=True, cwd=cwd)
        if not dry_run:
            subprocess.run(["git", "apply", patch_file], check=True, capture_output=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Patch validation/application failed: {e.stderr}")
        return False
    finally:
        if os.path.exists(patch_file):
            os.remove(patch_file)

class PatchGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_and_apply(self, prompt: str, target_file: str, cwd: Optional[str] = None) -> bool:
        """
        Generates a patch and attempts to apply it in `cwd` (a candidate's
        own worktree). Returns (success, diff).
        """
        diff = self.llm_client.generate_diff(prompt, target_file)

        print("Generated diff:")
        print(diff)

        return validate_and_apply_patch(diff, cwd=cwd), diff
