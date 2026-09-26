import os
import uuid
import json
import random
from typing import List, Dict, Any, Optional
from generation.prompt_builder import PromptBuilder
from generation.patch_generator import PatchGenerator, validate_and_apply_patch
from memory.db import ExperimentDB
from evolution.duplicate_checker import is_duplicate
from evolution.scheduler import ConcurrentScheduler
from evolution.scoring import score_candidates
from evolution.reporting import log_generation_report
from observability.logging_config import bind, get_logger

class EvolutionEngine:
    def __init__(self, config: Dict[str, Any], git_controller, sandbox, evaluator, metrics_calculator, failure_analyzer, patch_generator: PatchGenerator, prompt_builder: PromptBuilder, db: ExperimentDB, approval_store=None, truth=None, baseline_store=None, run_id=None, train_path=None):
        self.config = config.get('evolution', {})
        self.full_config = config
        self.eval_config = config.get('eval', {})
        self.git_controller = git_controller
        self.sandbox = sandbox
        self.evaluator = evaluator
        self.metrics_calculator = metrics_calculator
        self.failure_analyzer = failure_analyzer
        self.patch_generator = patch_generator
        self.prompt_builder = prompt_builder
        self.db = db
        self.approval_store = approval_store
        self.truth = truth or {}
        self.baseline_store = baseline_store
        # The full training file - a HOST-SELECTED subset of it (not just
        # the SUBSET_PERCENTAGE env var) is mounted per stage by the
        # scheduler when this is set. See eval/dataset.py:write_subset.
        self.train_path = train_path
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.logger = get_logger(__name__, run_id=self.run_id)

        self.pop_size = self.config.get('population_size', 5)
        self.max_gens = self.config.get('max_generations', 3)
        self.scheduler = ConcurrentScheduler(self.config.get('max_concurrent_sandboxes', 3), logger=self.logger)
        # a dedicated instance, not the global random module, so selection is
        # reproducible given the same seed without affecting anything else
        # that happens to use `random` in-process.
        self.rng = random.Random(self.config.get('random_seed', 42))

        self.best_scores = []
        self.duplicate_avoidance_count = 0
        self.duplicate_exhausted_count = 0
        self.generation_failed_count = 0
        # The most recent malformed-diff rejection's real git-apply
        # diagnostic, set by _generate_candidate and read by
        # _generate_population - same side-channel pattern already used by
        # the counters above to learn *why* _generate_candidate returned
        # None without changing its return type.
        self._last_malformed_detail = ""

    def _generate_candidate(self, goal: str, mutation_context: str = "") -> Optional[Dict[str, Any]]:
        """
        Creates the candidate's worktree FIRST, so the diff is generated
        and dry-run-checked against the file content that will actually
        receive the real apply later - checking against any other
        directory's state (the old behaviour) is meaningless once every
        candidate has its own worktree with its own current file content.

        Returns None if no usable diff was found after max_retries -
        callers must not substitute a scoreless fallback diff in that
        case (it would burn a full sandbox x stages budget on something
        that cannot score); the worktree is rolled back before returning.

        KNOWN LIMITATION: hardcoded to "candidate_script.py", unlike the
        sequential path (orchestrator/run.py), which supports target.files
        for multi-file candidates. This bypasses PatchGenerator.
        generate_and_apply() entirely (calling llm_client.generate_diff()
        directly), so multi-file support added there doesn't reach here.
        Extending evolutionary mode to multi-file candidates needs its own
        design pass - not silently half-supported in this one.
        """
        candidate_id = uuid.uuid4().hex[:8]
        candidate_logger = bind(self.logger, candidate_id=candidate_id)
        branch_name, worktree_path = self.git_controller.create_branch(candidate_id)
        script_path = os.path.join(worktree_path, "candidate_script.py")
        if not os.path.exists(script_path):
            with open(script_path, "w") as f:
                f.write("\n")
        with open(script_path) as f:
            current_content = f.read()

        max_retries = 3
        last_rejection = "malformed"
        for _ in range(max_retries):
            prompt = self.prompt_builder.build_prompt(goal)
            if mutation_context:
                prompt += f"\nMutation Instruction: {mutation_context}"

            diff = self.patch_generator.llm_client.generate_diff(prompt, "candidate_script.py", current_content)
            # Captured immediately, not read later from the shared
            # llm_client - candidates are generated in a batch before any
            # scheduling happens, so by schedule time last_usage would only
            # reflect whichever candidate was generated most recently.
            generation_usage = getattr(self.patch_generator.llm_client, "last_usage", {}) or {}

            error_out: List[str] = []
            if not validate_and_apply_patch(
                diff, cwd=worktree_path, dry_run=True, logger=candidate_logger, error_out=error_out,
                allowed_files=["candidate_script.py"],
            ):
                last_rejection = "malformed"
                self._last_malformed_detail = error_out[0] if error_out else ""
                continue

            dup_threshold = self.config.get('duplicate_threshold', 0.25)
            if not is_duplicate(diff, self.db, dup_threshold, hypothesis=goal):
                return {
                    'id': candidate_id,
                    'diff': diff,
                    'goal': goal,
                    'generation_usage': generation_usage,
                    'branch_name': branch_name,
                    'worktree_path': worktree_path,
                }
            else:
                self.duplicate_avoidance_count += 1
                last_rejection = "duplicate"

        self.git_controller.rollback(branch_name, worktree_path)
        if last_rejection == "duplicate":
            self.duplicate_exhausted_count += 1
        else:
            self.generation_failed_count += 1
        return None

    def _carry_over_elite(self, goal: str, parent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Re-includes the best candidate from the previous generation
        unmodified, so a strong solution isn't lost to mutation/selection
        noise even if every mutated sibling regresses. Two things a naive
        "just copy the diff into next_generation" version gets wrong (see
        the council audit):

          1. If the parent actually merged, its change is now already part
             of the current base - re-applying the SAME diff on top of
             that base is, at best, a pointless no-op and at worst a hard
             apply failure (the diff's removed-line context no longer
             matches what's there), wasting a full sandbox slot on a
             candidate that was never going to produce anything new. A
             merged parent is never carried over.
          2. The apply is dry-run-checked against a FRESH worktree based
             on the CURRENT base (not whatever base the parent was
             originally generated against) and duplicate-checked before
             being handed to the scheduler - exactly like a freshly
             generated candidate, not silently exempt from either check.

        Returns None (rolling back any worktree it created) if the parent
        merged, is a duplicate of something already in memory, or its diff
        no longer applies cleanly to the current base - the caller then
        fills that population slot via ordinary generation instead.
        """
        if parent.get('success'):
            return None

        dup_threshold = self.config.get('duplicate_threshold', 0.25)
        if is_duplicate(parent['diff'], self.db, dup_threshold, hypothesis=goal):
            return None

        candidate_id = uuid.uuid4().hex[:8]
        candidate_logger = bind(self.logger, candidate_id=candidate_id)
        branch_name, worktree_path = self.git_controller.create_branch(candidate_id)

        error_out: List[str] = []
        if not validate_and_apply_patch(
            parent['diff'], cwd=worktree_path, dry_run=True, logger=candidate_logger, error_out=error_out,
            allowed_files=["candidate_script.py"],
        ):
            self.git_controller.rollback(branch_name, worktree_path)
            return None

        return {
            'id': candidate_id,
            'diff': parent['diff'],
            'goal': goal,
            'branch_name': branch_name,
            'worktree_path': worktree_path,
        }

    def _select_parents(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # A generation can legitimately produce zero candidates - every slot
        # exhausted its malformed-diff or duplicate retries (a real,
        # observed outcome with a weaker/smaller LLM, not just a
        # theoretical edge case). Without this guard, 'tournament' below
        # calls max() on an empty sample and crashes the whole run instead
        # of just falling back to ordinary (non-mutated) generation for the
        # next round - see run()'s `if mutation_source:` check.
        if not candidates:
            return []

        strategy = self.config.get('selection_strategy', 'tournament')

        sorted_cands = sorted(candidates, key=lambda x: x.get('composite_score', -float('inf')), reverse=True)

        if strategy == 'top-k':
            k = max(1, len(sorted_cands) // 2)
            return sorted_cands[:k]
        elif strategy == 'tournament':
            parents = []
            k = max(1, len(sorted_cands) // 2)
            for _ in range(k):
                tournament = self.rng.sample(sorted_cands, min(3, len(sorted_cands)))
                best = max(tournament, key=lambda x: x.get('composite_score', -float('inf')))
                parents.append(best)
            return parents
        else:
            return sorted_cands[:1]

    def _adaptive_sizing(self) -> bool:
        sizing = self.config.get('adaptive_sizing', {})
        if not sizing.get('enabled', False):
            return False

        conv_gens = sizing.get('convergence_generations', 2)
        if len(self.best_scores) >= conv_gens + 1:
            recent_best = self.best_scores[-conv_gens:]
            prev_best = self.best_scores[-(conv_gens+1)]

            if all(abs(b - prev_best) < 0.01 for b in recent_best):
                self.pop_size = max(sizing.get('min_population', 2), self.pop_size - 1)
                return True
            else:
                self.pop_size = min(sizing.get('max_population', 10), self.pop_size + 1)
        return False

    def _record_exhausted_slot(self, goal: str, rejection: str, detail: str = "") -> None:
        outcome = "duplicate_exhausted" if rejection == "duplicate" else "failure"
        self.db.store_experiment(
            hypothesis=goal,
            diff="",
            rationale="Candidate generation exhausted retries",
            metrics={},
            outcome=outcome,
            failure_reason=f"exhausted {rejection} retries with no usable diff",
            # The last retry attempt's real git-apply diagnostic for a
            # "malformed" exhaustion - there's no equivalent real trace for
            # a "duplicate" exhaustion (nothing failed to apply, it was
            # rejected for being too similar to a past attempt).
            traceback=detail or None,
        )

    def _generate_population(self, goal: str, n: int, mutation_source: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        generation = []
        for _ in range(n):
            if mutation_source:
                parent = self.rng.choice(mutation_source)
                mutation_context = f"Vary the approach used in diff:\n{parent['diff']}"
            else:
                mutation_context = ""

            before_dup = self.duplicate_exhausted_count
            before_fail = self.generation_failed_count
            candidate = self._generate_candidate(goal, mutation_context)
            if candidate is not None:
                generation.append(candidate)
            elif self.duplicate_exhausted_count > before_dup:
                self._record_exhausted_slot(goal, "duplicate")
            elif self.generation_failed_count > before_fail:
                self._record_exhausted_slot(goal, "malformed", self._last_malformed_detail)
        return generation

    def run(self, goal: str):
        self.logger.info(f"Initializing Population of {self.pop_size}...")
        current_generation = self._generate_population(goal, self.pop_size)

        for gen in range(self.max_gens):
            self.duplicate_avoidance_count = 0
            self.duplicate_exhausted_count = 0
            self.generation_failed_count = 0
            self.logger.info(f"\n=== Running Generation {gen + 1}/{self.max_gens} ===")

            evaluated = self.scheduler.execute_generation(
                current_generation,
                self.eval_config.get('stages', []),
                self.git_controller,
                self.sandbox,
                self.evaluator,
                self.metrics_calculator,
                self.failure_analyzer,
                approval_store=self.approval_store,
                approval_config=self.full_config,
                truth=self.truth,
                baseline_store=self.baseline_store,
                train_path=self.train_path,
            )

            scored = score_candidates(evaluated, self.config)

            for c in scored:
                if c['success']:
                    outcome = "success"
                elif c.get('failure_category') == 'conflict':
                    outcome = "conflict"  # rebase-onto-base failed - real signal about population staleness, not a code failure
                elif c.get('eval_passed'):
                    outcome = "held"  # passed evaluation but not approved for merge
                else:
                    outcome = "failure"

                self.db.store_experiment(
                    hypothesis=goal,
                    diff=c['diff'],
                    rationale=f"Generation {gen} candidate",
                    metrics=c['metrics'],
                    outcome=outcome,
                    failure_reason=c.get('error_msg', None),
                    traceback=c.get('traceback') or None,
                )

            valid_scores = [c['composite_score'] for c in scored if 'composite_score' in c]
            best_score = max(valid_scores) if valid_scores else 0.0
            worst_score = min(valid_scores) if valid_scores else 0.0
            self.best_scores.append(best_score)

            total_time = sum(c.get('total_execution_time', 0.0) for c in scored)

            convergence_signal = self._adaptive_sizing()

            log_generation_report(
                generation=gen + 1,
                scored_candidates=scored,
                best_score=best_score,
                worst_score=worst_score,
                compute_time_spent=total_time,
                convergence_signal=convergence_signal,
                duplicate_avoidance_count=self.duplicate_avoidance_count,
                duplicate_exhausted_count=self.duplicate_exhausted_count,
                logger=self.logger,
            )

            if gen == self.max_gens - 1:
                break

            parents = self._select_parents(scored)

            next_generation = []

            if parents:
                elite = self._carry_over_elite(goal, parents[0])
                if elite is not None:
                    next_generation.append(elite)

            next_generation.extend(
                self._generate_population(goal, self.pop_size - len(next_generation), mutation_source=parents)
            )

            current_generation = next_generation