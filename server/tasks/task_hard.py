"""
Task 3 (Hard): Security vulnerability detection.
The agent must recognize injection, unsafe deserialization, SSRF, secrets in code, and similar CWE-class issues.
Difficulty: Hard — requires security knowledge and careful reading of data flows.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional

from ..models import CodeReviewObservation


DATA_PATH = Path(__file__).parent.parent / "data" / "hard_cases.json"


class HardTask:
    TASK_ID = "security_review"
    DESCRIPTION = (
        "You are a senior engineer reviewing a pull request with a security lens. "
        "Your job is to identify vulnerabilities such as SQL injection, path traversal, "
        "hardcoded secrets, insecure deserialization, SSRF, and unsafe comparison of secrets. "
        "Assign severity appropriately and suggest concrete mitigations (parameterized queries, "
        "allowlists, hmac.compare_digest, safe formats, etc.)."
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
