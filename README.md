# ML Research Agent - Phase 1

This repository contains Phase 1 ("The Core Loop & Sandbox") of an autonomous ML research agent.

## What is implemented in Phase 1
- **Orchestrator:** The core loop that proposes a candidate, runs it in a sandbox, evaluates it, and merges or rolls back based on results.
- **Git State Controller:** A version control wrapper that manages branches, commits, merges, and hard rollbacks.
- **Execution Sandbox:** A Docker-based sandbox to execute candidates safely.
- **Proxy Dataset Pipeline:** A mock evaluation pipeline that uses progressive scaling (1% -> 5% -> 20% -> 100%) to gate evaluations.

## What is NOT implemented yet (Deferred to Later Phases)
- **LLM-Based Code Generation:** The `generate_candidate()` function is currently a stub that returns a mock script. Code generation using LLMs is slated for Phase 2.
- **RAG Memory / Context:** Fetching previous results or papers is not yet implemented.
- **Evolutionary Search:** The current orchestrator runs a sequential, single-candidate loop. Multi-objective ranking or genetic search is out of scope.
- **Dashboard/UI:** The system outputs to CLI only.

## Security Disclaimer
The execution sandbox uses standard Docker limits (`--cpus` and `--memory`) and enforces wall-clock timeouts via `subprocess`. It is designed to prevent runaway experimental scripts (e.g., infinite loops or OOMs) from crashing the host. **It does NOT provide hardened security against zero-days, container escapes, or malicious privilege escalation.** Do not execute untrusted malware in this sandbox.

## How to run locally

### Prerequisites
- Docker must be installed and running.
- Python 3.9+
- Packages: `pip install -r requirements.txt` (or install manually: `GitPython`, `PyYAML`, `pytest`)

### Run the Core Loop
Execute a single dummy loop iteration:

```bash
python -m orchestrator.run --config configs/example.yaml
```

### Config Schema (`configs/example.yaml`)

```yaml
sandbox:
  timeout_seconds: 5         # Maximum wall-clock time for the candidate
  cpu_limit: "0.5"           # Docker CPU limit
  memory_limit: "256m"       # Docker memory limit

dataset:
  path: "dummy_data/"        # Path to the dataset (unused in Phase 1 stub)

eval:
  stages:                    # Progressive scaling stages
    - subset_percentage: 1
      threshold: 0.5
    - subset_percentage: 5
      threshold: 0.6
    - subset_percentage: 20
      threshold: 0.7
    - subset_percentage: 100
      threshold: 0.8
```

## Running Tests

Run the test suite using pytest:

```bash
pytest
```
