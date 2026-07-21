import os
import tempfile
import pytest
from sandbox.executor import SandboxExecutor

@pytest.fixture
def sandbox():
    # Setup Sandbox with 10 second timeout for normal tests
    config = {
        'timeout_seconds': 10,
        'cpu_limit': "0.5",
        'memory_limit': "256m"
    }
    return SandboxExecutor(config)

@pytest.fixture
def sandbox_short_timeout():
    # Setup Sandbox with 1 second timeout
    config = {
        'timeout_seconds': 1,
        'cpu_limit': "0.5",
        'memory_limit': "256m"
    }
    return SandboxExecutor(config)

def test_sandbox_successful_execution(sandbox):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("print('Hello from test')\n")
        script_path = f.name

    try:
        # absolute path for Docker volume
        abs_path = os.path.abspath(script_path)
        result = sandbox.run_candidate(abs_path)

        assert result['exit_code'] == 0
        assert not result['timeout']
        assert "Hello from test" in result['stdout']
    finally:
        os.remove(script_path)

def test_sandbox_timeout_enforcement(sandbox_short_timeout):
    # A script that sleeps for 5 seconds (sandbox timeout is 1 sec)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import time\ntime.sleep(5)\nprint('Done')\n")
        script_path = f.name

    try:
        abs_path = os.path.abspath(script_path)
        result = sandbox_short_timeout.run_candidate(abs_path)

        assert result['exit_code'] == -1
        assert result['timeout'] is True
        assert "Done" not in result['stdout']
    finally:
        os.remove(script_path)

def test_sandbox_failure(sandbox):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("raise ValueError('Test error')\n")
        script_path = f.name

    try:
        abs_path = os.path.abspath(script_path)
        result = sandbox.run_candidate(abs_path)

        assert result['exit_code'] != 0
        assert not result['timeout']
        assert "Test error" in result['stderr']
    finally:
        os.remove(script_path)
