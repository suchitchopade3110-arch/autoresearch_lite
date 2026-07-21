from abc import ABC, abstractmethod
import subprocess
import os

class LLMClient(ABC):
    @abstractmethod
    def generate_diff(self, prompt: str, target_file: str) -> str:
        """Generates a valid unified diff for the target file based on the prompt."""
        pass

class MockLLMClient(LLMClient):
    """
    A mock LLM client for testing. Returns a deterministic dummy patch.
    TODO: Phase 3 - Implement a real LLMClient (e.g. OpenAIClient, AnthropicClient).
    """
    def generate_diff(self, prompt: str, target_file: str) -> str:
        # A simple valid unified diff that ignores the original file content (treats as empty)
        # and replaces it with dummy code, ensuring it works even if the file exists.
        return f"""--- a/{target_file}
+++ b/{target_file}
@@ -1 +1,2 @@
-
+print('Hello from candidate script', flush=True)
+print('MOCK_SCORE: 0.99', flush=True)
"""

def validate_and_apply_patch(diff_content: str) -> bool:
    """
    Validates a patch by attempting to apply it cleanly.
    Returns True if successful, False otherwise.
    """
    patch_file = "candidate.patch"
    with open(patch_file, "w") as f:
        f.write(diff_content)

    try:
        # Check if the patch can be applied cleanly
        subprocess.run(["git", "apply", "--check", patch_file], check=True, capture_output=True)
        # Apply it
        subprocess.run(["git", "apply", patch_file], check=True, capture_output=True)
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

    def generate_and_apply(self, prompt: str, target_file: str) -> bool:
        """
        Generates a patch and attempts to apply it.
        Returns True if successful.
        """
        diff = self.llm_client.generate_diff(prompt, target_file)

        print("Generated diff:")
        print(diff)

        return validate_and_apply_patch(diff), diff
