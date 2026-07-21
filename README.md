# ML Research Agent - Phase 2

This repository contains Phase 2 ("RAG Memory & Patch Generation") of an autonomous ML research agent, built on top of the Phase 1 core loop.

## What is implemented in Phase 2
- **Orchestrator:** The core loop that proposes a candidate, runs it in a sandbox, evaluates it, and merges or rolls back based on results.
- **Git State Controller:** A version control wrapper that manages branches, commits, merges, and hard rollbacks.
- **Execution Sandbox:** A Docker-based sandbox to execute candidates safely.
- **Proxy Dataset Pipeline:** A mock evaluation pipeline that uses progressive scaling (1% -> 5% -> 20% -> 100%) to gate evaluations.
- **Experiment Memory (RAG):** Uses a local ChromaDB instance to store and retrieve hypotheses, patches, outcomes, and rationales for each experiment iteration.
- **Failure Analysis:** Extracts structured failure records from runtime errors, syntax errors, and metric regressions.
- **Prompt Builder:** Retrieves past successes and failures from memory to construct context-aware instructions.
- **Patch Generation:** Deterministically generates and validates `unified diff` patches (`MockLLMClient` currently implements this).
- **Static Analysis Pre-check:** Rejects malformed or invalid syntax code changes prior to sandbox execution.

## What is NOT implemented yet (Deferred to Later Phases)
- **Real LLM Integration:** `MockLLMClient` returns a predefined template patch. Swap in a real model in `generation/patch_generator.py` (e.g. OpenAI/Anthropic SDKs).
- **Evolutionary Search:** The current orchestrator runs a sequential, single-candidate loop. Multi-objective ranking or genetic search is out of scope.
- **Dashboard/UI:** The system outputs to CLI only.

## Security Disclaimer
The execution sandbox uses standard Docker limits (`--cpus` and `--memory`) and enforces wall-clock timeouts via `subprocess`. It is designed to prevent runaway experimental scripts (e.g., infinite loops or OOMs) from crashing the host. **It does NOT provide hardened security against zero-days, container escapes, or malicious privilege escalation.** Do not execute untrusted malware in this sandbox.

## How to run locally

### Prerequisites
- Docker must be installed and running.
- Python 3.9+
- Packages: `pip install -r requirements.txt` (or install manually: `GitPython`, `PyYAML`, `pytest`, `chromadb`, `pydantic`)

### Run the Core Loop
Execute a single dummy loop iteration:

```bash
python -m orchestrator.run --config configs/example.yaml --goal "Improve model performance"
```

### Swapping in a real LLM

Update `generation/patch_generator.py` and implement the `LLMClient` interface:

```python
class MyRealLLMClient(LLMClient):
    def generate_diff(self, prompt: str, target_file: str) -> str:
        # Call your API here and return the string unified diff
        return api.call(prompt)
```

## Running Tests

Run the test suite using pytest:

```bash
pytest
```