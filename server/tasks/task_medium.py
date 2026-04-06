"""
Task 2 (Medium): Logic bug detection.
The agent must find off-by-one errors, wrong boundaries, algorithm mistakes, and similar flaws.
Difficulty: Medium — requires reasoning about program behavior, not only syntax.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional

from ..models import CodeReviewObservation


DATA_PATH = Path(__file__).parent.parent / "data" / "medium_cases.json"


class MediumTask:
    TASK_ID = "logic_bug_review"
    DESCRIPTION = (
        "You are a senior engineer reviewing a pull request. "
        "Your job is to identify logic bugs: incorrect loop bounds, off-by-one errors, "
        "wrong boundary conditions for intervals, broken recursion or memoization, "
        "and algorithmic mistakes. Provide specific line-level comments and explain "
        "why the behavior is wrong."
    )

    def __init__(self, seed: Optional[int] = None):
        with open(DATA_PATH, encoding="utf-8") as f:
            self._cases = json.load(f)
        self._rng = random.Random(seed)
        self._current_case = None

    def reset(self, step: int = 0) -> CodeReviewObservation:
        self._current_case = self._rng.choice(self._cases)
        return CodeReviewObservation(
            task_id=self.TASK_ID,
            task_description=self.DESCRIPTION,
            code_snippet=self._current_case["code_snippet"],
            language=self._current_case["language"],
            context=self._current_case["context"],
            step=step,
            max_steps=3,
        )

    def get_ground_truth(self) -> dict:
        return self._current_case
