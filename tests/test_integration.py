import subprocess
import os
import tempfile
import shutil

def test_full_loop_execution():
    # Create a temporary directory to act as the repository
    with tempfile.TemporaryDirectory() as temp_repo:
        # Copy the project files to the temp directory
        for item in ["configs", "orchestrator", "eval", "sandbox", "vcs"]:
            shutil.copytree(item, os.path.join(temp_repo, item))

        # Initialize a new git repository
        subprocess.run(["git", "init"], cwd=temp_repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_repo, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_repo, check=False)
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_repo, check=True)

        # Run the loop in the temporary directory
        cmd = ["python", "-m", "orchestrator.run", "--config", "configs/example.yaml"]

        # Use python from the current path but set cwd to temp_repo
        env = os.environ.copy()
        env["PYTHONPATH"] = temp_repo
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_repo, env=env)

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0
    assert "Candidate passed all evaluation stages." in result.stdout
    assert "Merging." in result.stdout
