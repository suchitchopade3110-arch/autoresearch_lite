# ML Research Agent

An autonomous ML research agent loop: propose a candidate (as a diff), apply and commit it on its own git branch, run it in a sandbox against progressively larger subsets of a dataset, and merge or roll back based on the result - with RAG-style memory of past attempts and, optionally, a concurrent evolutionary search over a population of candidates per generation.

## What's implemented
- **Orchestrator (`orchestrator/run.py`):** two modes.
  - `--mode sequential` (default): one candidate at a time; loops up to `orchestrator.max_iterations`, stopping early once `target_score` is reached or `patience` iterations pass with no improvement.
  - `--mode evolutionary`: a population of candidates per generation, evaluated concurrently and evolved via `evolution/population.py`'s `EvolutionEngine` (selection, mutation-by-prompt, Pareto/weighted scoring, adaptive population sizing).
- **Git State Controller (`vcs/git_controller.py`):** every candidate gets its own `git worktree`, so branching, committing, merging, and rolling back a candidate never touches the caller's main checkout (or any uncommitted work in it) - and concurrent candidates in the evolutionary path never share a checkout with each other.
- **Execution Sandbox (`sandbox/executor.py`):** runs each candidate in Docker as a non-root user, with `--network none`, a read-only root filesystem, dropped capabilities, and the existing CPU/memory limits and wall-clock timeout.
- **Real evaluation pipeline (`eval/dataset.py`, `eval/pipeline.py`):** a deterministic synthetic dataset (a noisy linear boundary) with nested percentage subsets. Both orchestrator modes execute the candidate once per progressive-scaling stage (passing `SUBSET_PERCENTAGE`) and parse the candidate's actual reported `SCORE:` - no mocked or randomized scores.
- **Experiment Memory (RAG) (`memory/db.py`):** a local ChromaDB instance storing hypotheses, diffs, outcomes, metrics, and rationale per experiment.
- **Failure Analysis (`memory/failure_analysis.py`):** categorizes failures (syntax, runtime, timeout, resource-limit, metric-regression).
- **Prompt Builder (`generation/prompt_builder.py`):** retrieves past successes/failures from memory into the next prompt.
- **Patch Generation (`generation/patch_generator.py`):** validates and applies unified diffs; `MockLLMClient` is the placeholder generator - it emits a real, dataset-driven script (reads `DATASET_PATH`/`SUBSET_PERCENTAGE`, prints an actual `SCORE:`), not a hardcoded score, so the eval pipeline has something genuine to gate on even before a real LLM is wired in.
- **Static Analysis Pre-check (`generation/static_check.py`):** rejects malformed/invalid syntax before sandbox execution.

## What's NOT implemented yet
- **Real LLM integration.** `MockLLMClient` always returns the same diff regardless of prompt/history - implement the `LLMClient` interface with a real model to get genuinely different candidates per iteration. Because the mock's diff assumes a fixed starting file, concurrent evolutionary candidates whose worktree is created *after* an earlier candidate in the same generation has already merged will fail to apply (the file has moved on) - this is expected with a non-context-aware mock, not a bug, and goes away once a real LLM sees the file's current content in its prompt.
- **Dashboard/UI.** CLI only.

## Security Disclaimer
The sandbox runs candidates as a non-root user, with `--network none`, a read-only root filesystem, dropped capabilities, and standard Docker `--cpus`/`--memory` limits plus a wall-clock timeout via `subprocess`. This meaningfully raises the bar against a candidate trying to exfiltrate data, persist state, or exceed its resource limits. **It still does NOT provide hardened security against zero-days, container escapes, or a deliberately adversarial kernel exploit.** Do not execute untrusted malware in this sandbox.

## How to run locally

### Prerequisites
- Docker must be installed and running.
- Python 3.9+
- `pip install -r requirements.txt` (GitPython, PyYAML, pytest, chromadb)

### Sequential mode
```bash
python -m orchestrator.run --config configs/example.yaml --goal "Improve model performance"
# optional: --max-iterations 10 --target-score 0.9 --patience 3
```

### Evolutionary mode
```bash
python -m orchestrator.run --config configs/example.yaml --goal "Improve model performance" --mode evolutionary
```
Writes `evolution_report.jsonl` (one line per generation) and stores every candidate's outcome in the ChromaDB memory (`chroma_db/`).

### Swapping in a real LLM

Implement the `LLMClient` interface in `generation/patch_generator.py`:

```python
class MyRealLLMClient(LLMClient):
    def generate_diff(self, prompt: str, target_file: str) -> str:
        # Call your API here and return the string unified diff
        return api.call(prompt)
```

### Config Schema (`configs/example.yaml`)

```yaml
sandbox:
  timeout_seconds: 10
  cpu_limit: "0.5"
  memory_limit: "256m"

dataset:
  path: "dummy_data/dataset.jsonl"  # generated once if it doesn't exist
  size: 1000
  seed: 42

eval:
  stages:                          # progressive scaling
    - subset_percentage: 1
      threshold: 0.5
    - subset_percentage: 5
      threshold: 0.6
    - subset_percentage: 20
      threshold: 0.7
    - subset_percentage: 100
      threshold: 0.8

orchestrator:                      # sequential mode only
  max_iterations: 1
  target_score: 1.0
  patience: 1

evolution:                         # evolutionary mode only
  population_size: 5
  max_generations: 3
  max_concurrent_sandboxes: 3
  duplicate_threshold: 0.1
  selection_strategy: tournament
  scoring_strategy: pareto

generation:
  max_retrieved_failures: 2
  max_retrieved_successes: 2
  prompt_char_budget: 4000
```

## Running Tests

```bash
pytest
```

`test_dataset.py`, `test_eval_pipeline.py`, `test_generation.py`, `test_vcs.py`, and `test_memory.py` are pure Python (`test_memory.py` needs `chromadb` installed) and need no Docker. `test_sandbox.py` and `test_integration*.py` build and run the Docker sandbox image and require Docker to be running.
