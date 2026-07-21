import subprocess
import os
import tempfile
import shutil

def test_full_loop_execution():
    # Create a temporary directory to act as the repository
    with tempfile.TemporaryDirectory() as temp_repo:
        # Copy the project files to the temp directory
        for item in ["configs", "orchestrator", "eval", "sandbox", "vcs", "memory", "generation", "approval", "reporting"]:
            shutil.copytree(item, os.path.join(temp_repo, item))

        # create gitignore
        with open(os.path.join(temp_repo, ".gitignore"), "w") as f:
            f.write("chroma_db/\n__pycache__/\n")

        with open(os.path.join(temp_repo, "candidate_script.py"), "w") as f:
            f.write("\n")

        # This is an unattended/automated run with no dashboard operator, so
        # explicitly declare that - the default config requires approval
        # (fail-safe), which would otherwise leave every candidate held
        # pending for the full timeout with nobody there to decide.
        config_path = os.path.join(temp_repo, "configs", "example.yaml")
        with open(config_path) as f:
            config_text = f.read()
        config_text = config_text.replace("  enabled: true", "  enabled: false")
        with open(config_path, "w") as f:
            f.write(config_text)

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
