# Remediation record

Every gap identified across the pre-wave audit and the four remediation
waves, mapped to the commit that closed it and the test that proves it.
Commit hashes are short SHAs on `main`; PR numbers refer to
`suchitchopade3110-arch/autoresearch_lite`.

## Pre-wave audit fixes (PRs #2-#9)

| Gap | Fix | Proving test |
|---|---|---|
| Evolutionary scoring let a fast-failing candidate outrank a real success on execution-speed/energy alone | `27753d3` clamps every failed candidate's composite score below every successful one | `tests/test_scoring.py` |
| Duplicate detection never actually fired - `retrieve_experiments` didn't return a `distance` field, so the threshold check always compared against `inf` | `27753d3` | `tests/test_duplicate_checker.py::test_retrieve_experiments_includes_distance` |
| No human-approval gate before a merge | `20af43c` (PR #5) adds `approval/`, a SQLite-backed store, and a dashboard/API to decide pending candidates | `tests/test_approval_gate.py`, `tests/test_approval_store.py`, `tests/test_api.py` |
| Windows: `git worktree remove` failed with a file-lock `PermissionError` because a `git.Repo` handle was left open | `f745a65` | `tests/test_vcs.py` (Windows-run) |
| Windows: integration tests invoked a bare `python`, which can resolve to a different interpreter than the one running pytest | `566eb98` | `tests/test_integration*.py` |
| Windows: a candidate diff's patch was silently truncated - `.patch` files were written with platform line-ending translation, corrupting `\n` inside hunks | `ed92264`, `d8ed092` (the real root cause: a wrong declared hunk line count) | `tests/test_generation.py::test_mock_llm_diff_hunk_header_matches_actual_line_count` |
| `run_iteration` could crash the whole orchestrator loop by returning `False` instead of a `(bool, float)` tuple on a rejected patch | `2dc1400` | `tests/test_integration.py` |

## Wave 1 - close the reward-hacking gap (PR #10)

| Gap | Fix | Proving test |
|---|---|---|
| The eval pipeline trusted a candidate's own printed `SCORE:` claim to decide pass/fail - a candidate could buy a merge by lying | `0738ff4` replaces stdout-trust with `score_predictions()`, scoring real predictions against held-out `truth.json` the candidate never sees | `tests/test_reward_hacking.py::test_printed_score_claim_is_never_trusted` |
| `truth.json` (the held-out labels) had no guarantee against ever being mounted into the sandbox | `0738ff4` - `eval/dataset.py` keeps it host-side only; `sandbox/executor.py` only ever mounts `train.jsonl`/`test.jsonl` | `tests/test_reward_hacking.py::test_truth_json_absent_from_every_docker_mount_argument_repo_wide` |
| A merge only required clearing a fixed threshold, not beating what was already merged - a regression could still land | `0738ff4` adds `eval/baseline.py`'s `min_improvement` gate | `tests/test_baseline.py` |
| `MockLLMClient` didn't write real predictions, so the reward-hacking fix had nothing genuine to score | `0738ff4` rewrites it to implement an honest baseline and write `predictions.jsonl` | `tests/test_generation.py` |
| The `truth.json`-mount canary test shelled out to `grep`, which doesn't exist on Windows | `0d94809` | `tests/test_reward_hacking.py::test_truth_json_absent_from_every_docker_mount_argument_repo_wide` |

## Wave 2 - make the agent an agent (PR #11)

| Gap | Fix | Proving test |
|---|---|---|
| Every LLM client generated a diff without ever seeing the file's current content - every candidate after the first in a run was dead-on-arrival (diffs against stale state never apply) | `07c66b2` threads `current_content` through `LLMClient.generate_diff` | `tests/test_evolution_population.py` |
| No real, network-backed LLM client existed - only the deterministic mock | `07c66b2` adds `AnthropicClient` (retries on `git apply --check` failure, feeds stderr back into the prompt, tracks token/cost usage) | `tests/test_anthropic_client.py` |
| Duplicate-detection retries silently fell back to a scoreless placeholder diff on exhaustion, burning a full sandbox budget on something that could never score | `07c66b2` returns `None` and records a `duplicate_exhausted`/`failure` outcome instead | `tests/test_evolution_population.py` |
| The sandbox image had no ML libraries, so any real candidate script needing numpy/pandas/scikit-learn would fail immediately | `07c66b2` adds them to `sandbox/requirements-sandbox.txt`, installed before `COPY` for layer caching | `tests/test_sandbox.py::test_sandbox_has_ml_libraries` |

## Wave 3 - resilience (PR #12)

| Gap | Fix | Proving test |
|---|---|---|
| A merge conflict between two candidates could leave the shared checkout mid-merge (`MERGE_HEAD` present, working tree dirty) | `18703cf` - `merge()` rebases inside the candidate's own worktree first, then fast-forwards; raises `MergeConflict` instead of ever touching the shared checkout with a real conflict | `tests/test_vcs.py::test_second_conflicting_merge_raises_and_leaves_the_main_checkout_clean` |
| A human reviewer was a bottleneck on the sandbox pool - the old design awaited each candidate's approval before the next one's sandbox stage could start | `18703cf` - two-phase scheduler: every candidate's sandbox/eval work completes first, then every approval request is created up front | `tests/test_evolution_scheduler.py::test_all_sandbox_evaluation_completes_before_any_approval_request_blocks` |
| An empty `eval.stages` list let every candidate pass evaluation with a score of 0.0 (no stage ever ran to fail it) | `18703cf` - `config_schema.py` rejects it at load time with a readable error | `tests/test_config_schema.py::test_empty_eval_stages_rejected_with_readable_error` |
| The repo under evolution was implicitly the harness's own repository, with no way to separate them, and no warning when that happens | `18703cf` - `target.repo_path`/`target.base_ref` config keys, plus a startup warning | `tests/test_config_schema.py::test_target_is_harness_*` |
| `GitController` crashed via `repo.active_branch.name` if the target repo was in a detached HEAD state | `18703cf` - falls back to the current commit sha, and correctly advances it after a merge | `tests/test_vcs.py::test_detached_head_*` |
| A crashed/killed run left orphaned candidate worktrees/branches and stuck-pending approval requests with nothing to reclaim them | `18703cf` - `cleanup_orphans()`, `timeout_stale_requests()`, a `cleanup` subcommand, and automatic startup cleanup | `tests/test_vcs.py::test_cleanup_orphans_*`, `tests/test_approval_store.py::test_timeout_stale_requests_*`, `tests/test_orchestrator_cleanup.py` |
| `SIGINT`/`SIGTERM` had no handler - a stop signal propagated as an unhandled signal/traceback | `18703cf` | `tests/test_orchestrator_cleanup.py::test_sigterm_after_install_signal_handlers_exits_cleanly_instead_of_a_traceback` |
| Every operational message was a bare `print()` - no way to correlate log lines to a run or candidate | `18703cf` - `observability/logging_config.py`, structured JSON logs with `run_id`/`candidate_id` bound to every record | `tests/test_logging_config.py` |
| `cleanup_orphans()` identified orphan worktrees by comparing filesystem paths, which Windows can report inconsistently (short vs. long form) between git and Python, silently breaking the match | `85be1fd` - identifies orphans by the branch checked out in git's own porcelain listing instead | `tests/test_vcs.py::test_cleanup_orphans_removes_leftover_worktrees_and_branches_from_a_crashed_run` (Windows-run) |
| A raw `sqlite3` connection opened via `with conn:` in a test left a file handle open on Windows, blocking temp-directory cleanup | `85be1fd` | `tests/test_approval_store.py::test_timeout_stale_requests_*` (Windows-run) |
| `os.kill(pid, SIGTERM)` against the current process calls `TerminateProcess()` directly on Windows, bypassing Python's signal handler and killing the whole test run instead of raising `SystemExit` | `e296675` - invokes the installed handler directly instead of raising a real OS signal | `tests/test_orchestrator_cleanup.py::test_sigterm_after_install_signal_handlers_exits_cleanly_instead_of_a_traceback` |

## Wave 4 - hygiene

| Gap | Fix | Proving test |
|---|---|---|
| No LICENSE or packaging metadata | `LICENSE` (MIT), `pyproject.toml` | Build check: `python -m build --sdist` / `setuptools.find_packages()` discovers exactly the 11 real packages |
| No CI | `.github/workflows/tests.yml` runs the full suite (including Docker) on every push/PR to `main` | N/A - CI itself |
| `evolution/` had no `__init__.py`, unlike every other package | `evolution/__init__.py` | Packaging discovery check above |
| The dashboard had no authentication - anyone who could reach it could approve/reject | `api/main.py` - HTTP Basic Auth, fail-safe (required unless `DASHBOARD_AUTH_DISABLED=true`, and rejects everything if enabled with no credentials configured) | `tests/test_api_security.py::test_dashboard_requires_auth_by_default_when_no_credentials_configured`, `test_dashboard_rejects_wrong_credentials`, `test_dashboard_accepts_correct_credentials` |
| The approve/reject endpoints had no CSRF protection - a cross-site auto-submitting form could forge a decision | `api/main.py` - double-submit-cookie CSRF token on both forms | `tests/test_api_security.py::test_approve_without_csrf_token_is_rejected`, `test_approve_with_mismatched_csrf_token_is_rejected`, `test_approve_with_matching_csrf_token_succeeds` |
| The sandbox had no limit on process count or open file descriptors, and its `/tmp` tmpfs was unbounded (RAM-backed) | `sandbox/executor.py` - `--pids-limit`, `--ulimit nofile=`, a size-capped `--tmpfs` | `tests/test_sandbox.py::test_docker_run_command_includes_resource_hardening_flags` |
| `ExperimentDB` had no class docstring, unlike every other core class in the codebase | `memory/db.py` | N/A - documentation only |
| An exact-diff repeat paid for a full embedding + nearest-neighbor search before ever checking if it was a byte-for-byte duplicate | `memory/db.py:has_exact_diff`, wired into `evolution/duplicate_checker.py` | `tests/test_duplicate_checker.py::test_exact_duplicate_short_circuits_before_the_expensive_embedding_query` |
| `energy_estimate`'s placeholder methodology was implemented but not clearly flagged as such | Documented explicitly in README's "What's NOT implemented yet" (no code change - this is a deliberate, already-correct design choice, not a bug) | N/A - documentation |
| README was out of date with Waves 2-4 (target/harness separation, crash recovery, structured logging, dashboard security, sandbox hardening) | README rewritten | N/A - documentation |

## Wave 5 - council audit hardening

A three-auditor blind council review (`responses.md`/`autoresearch_lite_council_audit.md`)
found that the eval-integrity and merge-safety invariants Wave 1/3 believed
closed did not survive an adversarial candidate diff. Every finding below
was independently re-verified against this codebase before being fixed.

| # | Gap | Fix | Proving test |
|---|---|---|---|
| 1 | A diff could create `.eval_out` as a symlink (mode `120000`); the host followed it through `makedirs`/`chmod 0o777`/the rw bind mount, giving the sandbox rw access to `dummy_data/` (`truth.json`) or, in principle, anywhere else on the host | `vcs/diff_guard.py` refuses any diff touching a path outside `target.files` or creating a symlink/mode change, BEFORE `git apply` ever runs; `out_dir` is now a fresh host `tempfile.mkdtemp()` outside the worktree entirely (`orchestrator/run.py`, `evolution/scheduler.py`); `sandbox/executor.py` also refuses to mount an already-existing symlink as defense in depth | `tests/test_diff_guard.py`, `tests/test_sandbox.py::test_run_candidate_refuses_a_symlinked_*` |
| 2 | Held-out labels were regenerable from the public generator code plus the public default seed (`42`) - a candidate reading only `test.jsonl` could reconstruct `truth.json` without ever touching it | `eval/dataset.py:generate_split` generates a fresh, secret, host-only seed via `secrets.randbits` when `dataset.seed` is left unset (the new recommended default); the shipped example config no longer sets it | `tests/test_dataset.py::test_omitted_seed_*` |
| 3 | A diff was never restricted to `target.files` - with the default `target.repo_path: "."`, a candidate could rewrite the harness's own evaluation/approval code, configs, or tests | `vcs/diff_guard.py`, wired into every `validate_and_apply_patch` call site (`PatchGenerator.generate_and_apply`, `evolution/scheduler.py`, `evolution/population.py`) | `tests/test_diff_guard.py` |
| 4 | `merge()` ran `git checkout <base>` then `git merge --ff-only` directly in the caller's main working tree - it could silently switch the caller's checked-out branch, and a dirty tree raised an uncaught `GitCommandError` (not `MergeConflict`), crashing the run after approval was already recorded | `vcs/git_controller.py` - `merge()` (now `rebase_onto_base` + `finalize_merge`) advances the target ref via a compare-and-swap `git update-ref`; the working tree is only fast-forwarded when it's already on the target branch and has no uncommitted changes to tracked files | `tests/test_vcs.py::test_merge_never_switches_the_active_branch_when_base_ref_differs`, `test_merge_does_not_crash_when_main_worktree_has_uncommitted_changes` |
| 5 | `BaselineStore.get()` defaulted to `0.0` for an absent stage, so a fresh install's first candidate auto-approved with no human ever in the loop, despite `should_auto_approve`'s own docstring promising otherwise | `eval/baseline.py:get()` returns `None` (not `0.0`) when nothing has ever merged at a stage; both orchestrator paths compute `delta=None` in that case | `tests/test_baseline.py::test_get_returns_none_with_no_history`, `tests/test_auto_approve.py::test_scheduler_never_auto_approves_on_a_fresh_baseline` |
| 6 | Evolutionary phase 2 finalized every approved candidate using its phase-1 score/baseline check, even after an earlier candidate in the same generation had already moved the base and/or raised the baseline; a byte-for-byte duplicate that rebased to nothing was still recorded as a success | `evolution/scheduler.py`'s phase 2 now rebases (without finalizing), re-checks the baseline against a fresh read taken under the same lock, and re-evaluates for real if the base moved since phase 1 scored it; a rebase that collapses to the current base's own tree (git silently drops a duplicate commit) is rejected as `no_op_after_rebase` | `tests/test_evolution_scheduler.py::test_phase2_rechecks_the_baseline_immediately_before_finalizing`, `test_phase2_rejects_a_duplicate_whose_rebase_collapses_to_the_current_base`, `test_phase2_reevaluates_when_the_base_moved_since_phase1_scored_it` |
| 7 | "A failed candidate can never outrank a successful one" was keyed on `success` (== actually merged), which is `False` for every candidate in a generation where nothing merges (common with a human gate) - a near-instant crash could then outrank a candidate that scored 0.95+ and become a parent/elite | `evolution/scoring.py` keys the clamp on `eval_passed` instead | `tests/test_scoring.py::test_clamp_keys_on_eval_passed_not_merged` |
| 8 | The score-claim-mismatch flag lived on `EvalPipeline.last_stage_flags`, a single shared instance's mutable attribute read "immediately after" `evaluate_stage()` - unsafe under the concurrent calls evolutionary mode actually makes, and never reset on an early "process crashed" return | `eval/pipeline.py:evaluate_stage` now returns `(success, score, score_claim_mismatch)` directly; the attribute no longer exists | `tests/test_eval_pipeline.py::test_evaluate_stage_returns_the_mismatch_flag_directly_not_via_shared_state`, `test_evaluate_stage_mismatch_flag_is_false_not_stale_on_a_crashed_stage` |
| 9 | Evolutionary mode never ran the static syntax pre-check the sequential path always does, burning a full sandbox run on a candidate that could never produce anything but a crash | `evolution/scheduler.py`'s `evaluate_only` calls `check_syntax_multi` right after committing the patch, before ever touching the sandbox | `tests/test_evolution_scheduler.py::test_syntax_error_is_rejected_before_ever_reaching_the_sandbox` |
| 11 | `predictions.jsonl` was read host-side with no size limit and followed symlinks - a candidate pointing it at `/dev/zero` (or just writing a huge file) turned the host-side read into a denial of service | `eval/pipeline.py:score_predictions` refuses a symlinked predictions file and caps the read at `MAX_PREDICTIONS_FILE_BYTES` | `tests/test_eval_pipeline.py::test_score_predictions_refuses_a_symlinked_predictions_file`, `test_score_predictions_refuses_an_oversized_predictions_file` |
| 12 | Startup cleanup treated every `candidate-*` worktree/branch as orphaned unconditionally - git's worktree registry is repo-global, so a second orchestrator process against the same `repo_path` would delete a still-running first process's own in-flight candidates | `vcs/git_controller.py:cleanup_orphans` accepts `min_age_seconds`; `orchestrator/run.py`'s real startup path passes the approval gate's own `timeout_seconds` (a worktree younger than that could still be legitimately in flight) | `tests/test_vcs.py::test_cleanup_orphans_spares_young_worktrees_when_min_age_is_set` |
| 13 | A candidate's own stderr/traceback was stored to memory and replayed verbatim into the next prompt sent to the same LLM generating future candidates - a prompt-injection channel | `generation/prompt_builder.py` wraps candidate-originated text in an explicit `<untrusted-candidate-output>` fence with an instruction not to treat it as directions | `tests/test_prompt_builder.py::test_candidate_output_in_the_prompt_is_fenced_as_untrusted` |
| 14 | The dashboard loaded `https://cdn.tailwindcss.com` - an unpinned third-party script with no possible Subresource Integrity hash, running on the page whose forms approve/reject merges; its CSRF cookie was JS-readable for no functional reason (the token is embedded server-side into each form already) | `api/static/dashboard.css` - a small hand-written, self-hosted stylesheet with no external script at all; the CSRF cookie is now `httponly=True` | `tests/test_api_security.py::test_dashboard_serves_its_own_css_with_no_external_script`, `test_csrf_cookie_is_httponly` |

Findings #10 (progressive-scaling stages are honour-system) and #15/#16/#17
(the approval-timeout race, elite-carryover re-application, and remaining
grep-style tests) are tracked but not yet fixed in this wave - see the
council audit document for their detail. #10 is partially addressed as a
side effect of #1: `predictions.jsonl` can no longer be pre-committed into
`.eval_out` by a diff, and it is now wiped between stages in both
orchestrator paths.

## Final verification checklist

- [x] Full suite green (non-Docker): confirmed in this environment (no Docker daemon available here).
- [ ] Full suite green including Docker: requires a machine with Docker running - verify with `pytest -q`.
- [x] `tests/test_reward_hacking.py` (the reward-hacking canary) still passes.
- [x] `git grep truth.json -- '*.py'` shows no occurrence inside a Docker mount argument (`-v ...`) - only host-side loading code.
- [x] A deliberately injected mid-run conflict leaves the repo clean: `tests/test_vcs.py::test_second_conflicting_merge_raises_and_leaves_the_main_checkout_clean` constructs exactly this (two candidates editing the same line) and asserts no `MERGE_HEAD`, no dirty working tree.
- [ ] A real end-to-end evolutionary run with a real LLM client (`generation.client: anthropic`, `ANTHROPIC_API_KEY` set) and the approval gate on: requires a real API key, which was never provided to (or guessed by) this remediation work per its own ground rules - run manually with:
  ```bash
  ANTHROPIC_API_KEY=... python -m orchestrator.run --config configs/example.yaml --mode evolutionary
  ```
  and confirm `reports/latest_report.md` and `evolution_report.jsonl` reflect real generation cost/token usage.
