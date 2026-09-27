# Autoresearch Lite

[![Tests](https://github.com/suchitchopade3110-arch/autoresearch_lite/actions/workflows/tests.yml/badge.svg)](https://github.com/suchitchopade3110-arch/autoresearch_lite/actions/workflows/tests.yml)

An autonomous ML research agent loop. It proposes a candidate code change as a unified diff, applies and commits it on its own `git worktree`/branch, evaluates it in a sandboxed Docker container against progressively larger subsets of a dataset, and merges or rolls it back based on the result — gated by RAG-style memory of past attempts, a mandatory human-approval step before any merge, crash recovery, and an optional concurrent evolutionary search over a population of candidates per generation.

## Architecture

```
                    ┌─────────────────────┐
                    │   Orchestrator      │  orchestrator/run.py
                    │ (sequential | evo)  │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                       ▼
┌───────────────┐    ┌───────────────────┐    ┌──────────────────┐
│ Prompt Builder │──▶│ Patch Generator    │──▶│ Diff-Scope Guard  │
│ (RAG memory)   │    │ (LLM client)       │    │ (vcs/diff_guard)  │
└───────────────┘    └───────────────────┘    └────────┬─────────┘
                                                         ▼
                                              ┌──────────────────┐
                                              │ Git Controller    │  per-candidate worktree
                                              │ (vcs/git_controller)│
                                              └────────┬─────────┘
                                                         ▼
                                              ┌──────────────────┐
                                              │ Execution Sandbox │  Docker, --network none,
                                              │ (sandbox/executor) │  non-root, ro rootfs
                                              └────────┬─────────┘
                                                         ▼
                                              ┌──────────────────┐
                                              │ Eval Pipeline      │  staged scoring +
                                              │ (eval/pipeline)    │  sealed holdout
                                              └────────┬─────────┘
                                                         ▼
                                              ┌──────────────────┐
                                              │ Approval Gate      │  SQLite, human decision
                                              │ (approval/)        │  required by default
                                              └────────┬─────────┘
                                                         ▼
                                    merge (compare-and-swap ref update) or rollback
```

`memory/db.py` (ChromaDB) and `api/main.py` (dashboard/API) both read/write the same on-disk stores (`chroma_db/`, `approvals.db`, `evolution_report.jsonl`), so there is no separate data path to drift out of sync between the orchestrator process and the dashboard process.

## Core components

| Component | File | Responsibility |
|---|---|---|
| Orchestrator | `orchestrator/run.py` | Drives the loop; `--mode sequential` (default, one candidate at a time, up to `max_iterations`, early-stops on `target_score`/`patience`) or `--mode evolutionary` (population per generation via `evolution/population.py`'s `EvolutionEngine`). `cleanup` subcommand reclaims orphaned state; runs automatically at every `run` start; `SIGINT`/`SIGTERM` exit cleanly. |
| Config validation | `config_schema.py` | Pydantic schema validated at load time — a malformed/missing `eval.stages` is rejected up front instead of silently scoring every candidate 0.0. |
| Prompt Builder | `generation/prompt_builder.py` | Retrieves past successes/failures from memory into the next generation prompt. |
| Patch Generator | `generation/patch_generator.py` | Validates and applies unified diffs via `LLMClient.generate_diff(prompt, target_file, current_content)`. Three implementations: `MockLLMClient` (default, deterministic, no network/key), `AnthropicClient` (`generation.client: anthropic`, reads `ANTHROPIC_API_KEY` from env only), `LocalLLMClient` (`generation.client: local`, OpenAI-compatible endpoint via `generation.base_url`). Both real clients retry up to `generation.max_apply_retries` (default 3) on `git apply --check` failure, feeding stderr back into the next prompt, and record token counts/estimated cost. |
| Static Analysis | `generation/static_check.py` | Rejects malformed/invalid syntax before it ever reaches the sandbox. |
| Diff-Scope Guard | `vcs/diff_guard.py` | Checked before `git apply`: rejects any diff touching a path outside `target.files`, creating a symlink, or changing permission bits/renames/copies. Enforced identically in both modes via `generation/patch_generator.py:validate_and_apply_patch(..., allowed_files=target.files)`. |
| Git State Controller | `vcs/git_controller.py` | One `git worktree` per candidate — branching/committing/merging/rollback never touches the caller's main checkout. Merge = rebase candidate onto target inside its own worktree, then advance the target ref via compare-and-swap `git update-ref` (never `git checkout`/`git merge` on the shared tree). Conflicts raise `MergeConflict` (recorded as `conflict`), never leave the shared checkout mid-merge. Detects/warns if a separate linked worktree elsewhere is also checked out on the target branch. |
| Execution Sandbox | `sandbox/executor.py` | Docker container, non-root user, `--network none`, read-only root filesystem, dropped capabilities, `--pids-limit`, per-process fd `--ulimit`, size-capped `/tmp` tmpfs, `--cpus`/`--memory` limits, wall-clock timeout. |
| Eval Pipeline | `eval/dataset.py`, `eval/pipeline.py` | Deterministic synthetic dataset by default (noisy linear boundary), split into `train`/`test`/`holdout` (mounted read-only at their respective stages) plus `truth.json`/`holdout_truth.json` (host-only, never mounted). Host-selected subset files enforce the progressive-scaling stage boundary regardless of what a candidate claims. Predictions are scored against `truth.json` via a pluggable `eval.scorer`; the printed `SCORE:` line is a diagnostic only, never trusted for gating. A merge also requires beating `eval.min_improvement` over the best-known score for that stage (`eval/baseline.py`). |
| Sealed Holdout | `eval/pipeline.py:run_holdout_evaluation` | A second held-out split never touched by any stage or the baseline gate; a candidate that clears every stage is scored once more against it (`metrics.holdout_score`), shown alongside the selection score but never fed back into any auto-approve criterion — flags overfitting to the selection sample. |
| Experiment Memory (RAG) | `memory/db.py` | Local ChromaDB store of hypotheses, diffs, outcomes, metrics, rationale (cosine distance). Exact-diff repeats are caught cheaply via metadata lookup (`has_exact_diff`) before paying for an embedding + nearest-neighbor search. |
| Failure Analysis | `memory/failure_analysis.py` | Categorizes failures: syntax, runtime, timeout, resource-limit, metric-regression. |
| Multi-objective Scoring | `evolution/scoring.py` | Real evaluation score drives selection; a failed candidate can never outrank a successful one under either scoring strategy. |
| Approval Gate | `approval/` | Every candidate that passes evaluation is held pending human decision, in both modes. Defaults to *required* even on a missing/malformed `approval` config (`approval/gate.py:resolve_approval_config`) — only an explicit, valid `approval.enabled: false` disables it. A timeout with no decision persists as `timed_out` and never merges. In evolutionary mode, every candidate in a generation gets its approval request created up front so a human is never a bottleneck on the sandbox pool. Decisions persist in SQLite (`approvals.db`). |
| Dashboard + API | `api/main.py` | FastAPI app + self-hosted-CSS page (no third-party CDN, auto-refreshing) for reviewing/approving/rejecting candidates, plus `/api/pending`, `/api/approvals`, `/api/history`, `/api/report`. Reads directly from the same ChromaDB store, approval DB, and `evolution_report.jsonl` the orchestrator writes. Protected by HTTP Basic Auth (fail-safe: required unless explicitly disabled) and an httponly double-submit-cookie CSRF token. |
| Report Generator | `reporting/report_generator.py` | `compute_kpis()` is the single function both the dashboard and the end-of-run report (`reports/latest_report.md`) call, so the two surfaces can't diverge. Tracks merge rate, duplicate-avoidance rate, compute cost per improvement, approval outcomes. |
| Crash Recovery | `vcs/git_controller.py:cleanup_orphans`, `approval/store.py:timeout_stale_requests`, `sandbox/executor.py:cleanup_orphan_containers` | Reclaims orphaned worktrees/branches, stale pending approvals, and still-running sandbox containers automatically at startup or via `cleanup`. Containers are host-global, not repo-scoped — not fully safe with two concurrent orchestrator processes on the same host. |
| Structured Logging | `observability/logging_config.py` | JSON logs to stdout with `run_id` (and `candidate_id` where applicable) bound to every record. |

## Loop mechanics (sequential mode)

```
for iteration in 1..max_iterations:
    prompt          = PromptBuilder.build(goal, memory.retrieve(goal))
    diff            = LLMClient.generate_diff(prompt, target_file, current_content)
    static_check(diff)                              # reject malformed syntax
    diff_guard.validate(diff, allowed_files)         # reject out-of-scope diff
    worktree        = GitController.create_worktree(candidate_branch)
    apply_and_commit(worktree, diff)
    for stage in eval.stages:                        # progressive scaling
        run in sandbox(subset=stage.subset_percentage)
        score = score_predictions(...)
        if score < stage.threshold: reject; break
    if all stages passed and score beats baseline.min_improvement:
        holdout_score = run_holdout_evaluation(worktree)   # informational only
        approval = ApprovalGate.request(candidate)
        wait up to approval.timeout_seconds
        if approved: GitController.merge(worktree)         # rebase + CAS ref update
        else: rollback
    memory.record(diff, outcome, metrics)
    if score >= target_score or patience exceeded: break
```

Evolutionary mode (`--mode evolutionary`) runs this per candidate concurrently across a population (`evolution.population_size`) per generation, evaluated via `evolution/scheduler.py`, selected/mutated via `evolution/population.py`'s `EvolutionEngine` (tournament/other selection, mutation-by-prompt, Pareto or weighted scoring, adaptive population sizing — all seeded via `evolution.random_seed`).

## Security model

- **Sandbox isolation:** Docker, `--network none`, non-root user, read-only root filesystem, dropped capabilities, `--pids-limit` (fork-bomb protection), per-process `--ulimit` (fd exhaustion), size-capped `/tmp` tmpfs, `--cpus`/`--memory` limits, `subprocess`-enforced wall-clock timeout. **Not** hardened against zero-days, container escapes, or a deliberately adversarial kernel exploit — do not run untrusted malware in it.
- **Diff scope enforcement:** every diff is validated against `target.files` *before* `git apply` — `git apply` itself has no such concept. Prevents a candidate from rewriting the harness's own evaluation code or symlinking a sandbox output path to an arbitrary host path.
- **No shared-checkout corruption:** merges never run in the caller's working tree; a rebase conflict or a concurrent ref move raises a typed error instead of leaving a shared checkout mid-merge.
- **No answer-key leakage:** `truth.json`/`holdout_truth.json` stay host-side, never mounted into the sandbox. The dataset generator's seed is a fresh, secret, host-only value by default — a fixed/public seed would let a candidate regenerate held-out labels from the (public) generator code.
- **No reward hacking via output forgery:** `predictions.jsonl` is opened with a symlink AND named-pipe (FIFO) check plus a size cap before scoring, regardless of scorer — a FIFO has no fixed size and would defeat a size cap the same way a symlink to `/dev/zero` would. A candidate's self-reported `SCORE:` line is compared against the real score only as a diagnostic, never trusted for gating (see `tests/test_reward_hacking.py`).
- **Fail-safe approval gate:** required by default even on a missing/malformed config; only an explicit `approval.enabled: false` disables it.
- **Fail-safe dashboard auth:** HTTP Basic Auth required unless `DASHBOARD_AUTH_DISABLED=true` is set explicitly. Approve/reject POSTs carry a double-submit-cookie CSRF token and validate `Origin` (fallback `Referer`) against `Host` — a check no other port can forge, closing the gap that `SameSite=Strict` alone leaves (SameSite's "site" check is host-based and ignores port).

## Setup

### Prerequisites

- Docker installed and running
- Python 3.9+
- `pip install -r requirements.txt`

### 1. Start the dashboard

```bash
# Local/single-user use:
DASHBOARD_AUTH_DISABLED=true uvicorn api.main:app --reload

# Or, to require a login:
DASHBOARD_USERNAME=admin DASHBOARD_PASSWORD=change-me uvicorn api.main:app --reload
```

Open `http://localhost:8000` for pending approvals and run history. Leave it running — the orchestrator blocks on decisions made here.

### 2. Run the orchestrator

```bash
# Sequential
python -m orchestrator.run --config configs/example.yaml --goal "Improve model performance" \
  --max-iterations 10 --target-score 0.9 --patience 3

# Evolutionary
python -m orchestrator.run --config configs/example.yaml --goal "Improve model performance" --mode evolutionary
```

No approval decision within `approval.timeout_seconds` (default 1800s) holds the candidate — it will not merge.

### 3. Recovering from a crash

```bash
python -m orchestrator.run cleanup --config configs/example.yaml
```

Also runs automatically at the start of every `run` invocation; useful standalone after a crash without immediately starting a new run.

### Running without a human present (CI, demos)

Set `approval.enabled: false` explicitly. `configs/example.yaml` ships with `enabled: true`.

### Separating the harness from the code under evolution

`target.repo_path` defaults to `.` (the harness's own repo). Point it at a separate checkout for anything beyond local experimentation — the orchestrator warns at startup if it resolves back to the harness's own directory.

### Bring your own project

Three config changes, no harness source edits:

1. `target.repo_path` + `target.files` — the checkout and file(s) a candidate may touch.
2. `dataset.mode: custom` — skips the synthetic generator, validates `train.jsonl`/`test.jsonl`/`truth.json` exist at `dataset.path` (filenames only checked, not row schema — see `eval/dataset.py:validate_custom_dataset`). Keep `truth.json` out of any path the sandbox would mount.
3. `eval.scorer: "your_module:your_function"` — dotted path to `(preds: Dict[str, Any], truth: Dict[str, Any]) -> Tuple[float, str]`, replacing `eval.pipeline:binary_accuracy_scorer`. Return `(0.0, "reason")` for a clean failure rather than raising.

```yaml
target:
  repo_path: "/path/to/your/project"
  files: ["your_module/candidate_script.py"]

dataset:
  path: "your_dataset_dir"
  mode: custom

eval:
  scorer: "your_module.scoring:score"
  stages:
    - subset_percentage: 100
      threshold: 0.0
```

### Swapping in a real LLM

```bash
export ANTHROPIC_API_KEY=...
```
```yaml
generation:
  client: anthropic
  # model: claude-sonnet-5   # default
```

For a local model (no API key, no per-token cost, higher malformed-diff retry rate than a frontier model):

```yaml
generation:
  client: local
  base_url: "http://localhost:11434/v1"   # any OpenAI-compatible /chat/completions endpoint
  max_apply_retries: 5                    # raise for a weaker local model
```

A code-tuned model (Qwen2.5-Coder, DeepSeek-Coder, ...) applies far more reliably than a general chat model.

To integrate another provider, implement `LLMClient`:

```python
class MyRealLLMClient(LLMClient):
    def generate_diff(self, prompt: str, target_file: str, current_content: str) -> str:
        return api.call(prompt, current_content)
```

## Config reference (`configs/example.yaml`)

```yaml
target:
  repo_path: "."
  # base_ref: main

sandbox:
  timeout_seconds: 5
  cpu_limit: "0.5"
  memory_limit: "256m"
  pids_limit: 128
  ulimit_nofile: 1024
  tmpfs_size_mb: 64

dataset:
  path: "dummy_data"
  size: 1000
  # seed intentionally omitted — see eval/dataset.py:generate_split docstring
  test_frac: 0.25

eval:
  stages:                          # non-empty, progressive scaling
    - subset_percentage: 1
      threshold: 0.5
    - subset_percentage: 5
      threshold: 0.6
    - subset_percentage: 20
      threshold: 0.7
    - subset_percentage: 100
      threshold: 0.8
  min_improvement: 0.001
  state_path: "state.json"

orchestrator:                      # sequential mode only
  max_iterations: 1
  target_score: 1.0
  patience: 1

evolution:                         # evolutionary mode only
  population_size: 5
  max_generations: 3
  max_concurrent_sandboxes: 3
  duplicate_threshold: 0.25
  selection_strategy: tournament
  scoring_strategy: pareto
  random_seed: 42

generation:
  max_retrieved_failures: 2
  max_retrieved_successes: 2
  prompt_char_budget: 4000
  client: mock                     # or "anthropic" / "local"

approval:                          # both modes
  enabled: true                    # missing/malformed config also defaults to true
  timeout_seconds: 1800
  poll_interval_seconds: 5
  db_path: "approvals.db"
```

## Known limitations

- **`MockLLMClient` always returns the same diff regardless of prompt/history** — the zero-setup default, not a bug. No live end-to-end run with a real LLM client has been executed as part of this codebase's remediation work; there is no empirical evidence yet that the loop improves a real task with a real model, only that its safety/integrity mechanisms hold. Validate with `generation.client: anthropic` against a task with real headroom (the built-in synthetic task is near its achievable ceiling on the first try), across multiple seeds, against a random-search baseline at equal compute budget.
- **Carbon-footprint proxy is not a real methodology.** `energy_proxy = execution_time * energy_proxy_watts_constant` (default 10.0) is a placeholder for relative comparison, not CodeCarbon or a grid-intensity constant.
- **Evolutionary mode's population all edits one file.** Every candidate in a generation patches `candidate_script.py` from the same base; once the first candidate finalizes, others touching the same lines either conflict on rebase (`conflict`) or collapse to `no_op_after_rebase`. Multi-file candidates aren't supported in evolutionary mode yet (sequential mode already supports `target.files` with multiple entries).
- **A human's approval doesn't necessarily cover the exact code that ships (evolutionary mode).** If another candidate in the same generation merges first, an approved diff is rebased onto the new base before finalizing and re-evaluated for real (never published unverified) — but a human never explicitly re-reviewed that specific (diff, new-base) pair. Recorded transparently via `metrics.rebased_after_approval`, `approved_base_commit`, `merged_base_commit` (shown as a dashboard badge) rather than silently treated as equivalent.

## Testing

```bash
pytest
```

Most test files are pure Python, no Docker required. `test_sandbox.py` and `test_integration*.py` build/run the Docker sandbox image (exception: `test_sandbox.py::test_docker_run_command_includes_resource_hardening_flags`, which mocks `subprocess.run`). CI (`.github/workflows/tests.yml`) runs the full suite on every push/PR against `main`.

## License

MIT — see [LICENSE](LICENSE).
