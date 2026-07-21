import subprocess
import time
from typing import Dict, Any

class SandboxExecutor:
    """
    Executes a script inside a Docker sandbox.

    SECURITY NOTE:
    This sandbox enforces wall-clock timeout and CPU/memory constraints via Docker.
    It does NOT protect against deliberate malicious breakouts. Do not run untrusted
    malware here. It's meant for safe experimental ML code.
    """
    def __init__(self, config: Dict[str, Any]):
        self.timeout = config.get('timeout_seconds', 10)
        self.cpu_limit = config.get('cpu_limit', '1.0')
        self.memory_limit = config.get('memory_limit', '512m')

        # Build the image once when executor is instantiated (or could be external)
        self._build_image()

    def _build_image(self):
        subprocess.run(
            ["docker", "build", "-t", "ml-sandbox", "-f", "sandbox/Dockerfile", "sandbox/"],
            check=True,
            capture_output=True
        )

    def run_candidate(self, script_path: str, env_vars: Dict[str, str] = None) -> Dict[str, Any]:
        """Runs the given script inside the docker sandbox."""
        start_time = time.time()

        import uuid
        container_name = f"sandbox-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "run", "--rm",
            f"--name={container_name}",
            f"--cpus={self.cpu_limit}",
            f"--memory={self.memory_limit}",
        ]

        if env_vars:
            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])

        cmd.extend([
            "-v", f"{script_path}:/app/candidate_script.py:ro",
            "ml-sandbox"
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            execution_time = time.time() - start_time
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "timeout": False
            }
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            # explicitly stop the container to avoid orphaned processes
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            return {
                "exit_code": -1,
                "stdout": e.stdout.decode() if e.stdout else "",
                "stderr": e.stderr.decode() if e.stderr else f"Execution timed out after {self.timeout} seconds.",
                "execution_time": execution_time,
                "timeout": True
            }
