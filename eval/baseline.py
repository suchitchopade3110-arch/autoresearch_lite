import json
import os
from typing import Dict, Optional

DEFAULT_MIN_IMPROVEMENT = 0.001


class BaselineStore:
    """
    Tracks the best-known score per eval stage, persisted to disk. Gating a
    merge on an absolute threshold alone lets a candidate that regresses
    relative to what's already been achieved still merge, as long as it
    clears that threshold (e.g. a candidate that scores 0.81 merges even
    after a 0.95 candidate already merged, because 0.81 >= 0.8). Requiring
    a genuine improvement over the best-known score closes that gap.
    """
    def __init__(self, path: str):
        self.path = path
        self._scores: Dict[str, float] = self._load()

    def _load(self) -> Dict[str, float]:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f).get("baseline_scores", {})
        return {}

    def _save(self):
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"baseline_scores": self._scores}, f)

    def get(self, stage_key) -> Optional[float]:
        """
        Returns the best-known score for this stage, or None if nothing has
        ever merged at it yet. None (not 0.0) matters to callers computing
        "improvement over baseline" for approval.auto_approve's threshold
        (see approval/gate.py:should_auto_approve): a candidate with
        nothing to compare against isn't a 100%-improvement over a 0.0
        floor, it's simply not comparable, and must always fall through to
        human review rather than auto-merge on a fresh install.
        """
        key = str(stage_key)
        return self._scores[key] if key in self._scores else None

    def update_if_better(self, stage_key, score: float) -> None:
        key = str(stage_key)
        if score > self._scores.get(key, 0.0):
            self._scores[key] = score
            self._save()

    def passes(self, stage_key, score: float, min_improvement: float = DEFAULT_MIN_IMPROVEMENT) -> bool:
        """
        The absolute merge gate: unlike get()'s auto-approve-facing Optional
        return, a genuinely absent baseline here floors at 0.0 - a first
        candidate (nothing merged yet at this stage) must still be able to
        merge on its own eval-stage-threshold merits; there is nothing yet
        to have "regressed" relative to.
        """
        baseline = self.get(stage_key)
        return score > (baseline if baseline is not None else 0.0) + min_improvement
