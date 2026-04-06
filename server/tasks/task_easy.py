"""
Task 1 (Easy): Style and formatting review.
The agent must identify PEP8/language-convention violations.
Difficulty: Easy — issues are syntactic and visible without deep analysis.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional
from ..models import CodeReviewObservation


DATA_PATH = Path(__file__).parent.parent / "data" / "easy_cases.json"


class EasyTask:
    TASK_ID = "style_review"
    DESCRIPTION = (
        "You are a senior engineer reviewing a pull request. "
        "Your job is to identify style violations, naming convention issues, "
        "and formatting problems in the code. Focus on correctness of style "
        "according to the language's official style guide (PEP 8 for Python, "
        "Airbnb style for JavaScript). Provide specific line-level comments."
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
