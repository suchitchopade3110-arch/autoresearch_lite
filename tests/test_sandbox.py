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

def test_sandbox_env_vars_and_dataset_mount():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as data_f:
        data_f.write('{"x1": 1.0}\n')
        dataset_path = data_f.name

    config = {'timeout_seconds': 10, 'cpu_limit': "0.5", 'memory_limit': "256m"}
    sandbox_with_dataset = SandboxExecutor(config, dataset_path=dataset_path)

    script_content = (
        "import os\n"
        "print('SUBSET_PERCENTAGE=' + os.environ.get('SUBSET_PERCENTAGE', 'missing'))\n"
        "print('DATASET_PATH=' + os.environ.get('DATASET_PATH', 'missing'))\n"
        "print(open(os.environ['DATASET_PATH']).read())\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        abs_script = os.path.abspath(script_path)
        result = sandbox_with_dataset.run_candidate(abs_script, env_vars={"SUBSET_PERCENTAGE": "5"})

        assert result['exit_code'] == 0
        assert "SUBSET_PERCENTAGE=5" in result['stdout']
        assert "DATASET_PATH=/app/data/dataset.jsonl" in result['stdout']
        assert '"x1": 1.0' in result['stdout']
    finally:
        os.remove(script_path)
        os.remove(dataset_path)

def test_sandbox_has_no_network_access(sandbox):
    script_content = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    print('CONNECTED')\n"
        "except OSError:\n"
        "    print('NO_NETWORK')\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        abs_path = os.path.abspath(script_path)
        result = sandbox.run_candidate(abs_path)
        assert "NO_NETWORK" in result['stdout']
    finally:
        os.remove(script_path)
