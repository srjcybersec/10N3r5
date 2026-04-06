"""
CodeReviewEnv — the core RL environment.

Episode structure:
  - reset() selects a random task and returns the initial observation
  - step(action) grades the action and returns (obs, reward, done, info)
  - state() returns the full current state for debugging

Multi-step design:
  - Each episode runs for up to max_steps steps
  - After the first step, feedback is provided to guide refinement
  - Reward is dense: each step can improve score by refining comments
  - Episode ends when max_steps is reached OR agent submits final review
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

from .models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewReward,
    EnvironmentState,
)
from .graders import grade_action
from .tasks.task_easy import EasyTask
from .tasks.task_medium import MediumTask
from .tasks.task_hard import HardTask


TASK_REGISTRY = {
    "style_review": (EasyTask, "easy"),
    "logic_bug_review": (MediumTask, "medium"),
    "security_review": (HardTask, "hard"),
}


class CodeReviewEnv:
    MAX_STEPS = 3  # Agent can refine its review up to 3 times

    def __init__(self, task_id: str = "style_review", seed: Optional[int] = None):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(TASK_REGISTRY)}")

        self._task_id = task_id
        self._seed = seed
        task_cls, self._difficulty = TASK_REGISTRY[task_id]
        self._task = task_cls(seed=seed)

        self._step = 0
        self._done = False
        self._current_obs: Optional[CodeReviewObservation] = None
        self._ground_truth: Optional[dict] = None
        self._episode_rewards: list[float] = []
        self._cumulative_reward: float = 0.0
        self._last_reward: Optional[CodeReviewReward] = None

    def reset(self) -> CodeReviewObservation:
        self._step = 0
        self._done = False
        self._episode_rewards = []
        self._cumulative_reward = 0.0
        self._last_reward = None

        self._current_obs = self._task.reset(step=self._step)
        self._ground_truth = self._task.get_ground_truth()
        return self._current_obs

    def step(
        self, action: CodeReviewAction
    ) -> Tuple[CodeReviewObservation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1

        # Grade the action
        reward_obj = grade_action(action, self._ground_truth, self._difficulty)
        reward = reward_obj.total

        self._episode_rewards.append(reward)
        self._cumulative_reward += reward
        self._last_reward = reward_obj

        done = self._step >= self.MAX_STEPS
        self._done = done

        # Build next observation with feedback for refinement
        feedback = reward_obj.explanation if not done else None
        next_obs = CodeReviewObservation(
            task_id=self._task_id,
            task_description=self._current_obs.task_description,
            code_snippet=self._current_obs.code_snippet,
            language=self._current_obs.language,
            context=self._current_obs.context,
            step=self._step,
            max_steps=self.MAX_STEPS,
            previous_comments=action.comments,
            feedback=feedback,
        )
        self._current_obs = next_obs

        info = {
            "reward_breakdown": reward_obj.model_dump(),
            "step": self._step,
            "difficulty": self._difficulty,
        }

        return next_obs, reward, done, info

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self._task_id,
            step=self._step,
            done=self._done,
            current_observation=self._current_obs,
            cumulative_reward=self._cumulative_reward,
            episode_rewards=self._episode_rewards,
            metadata={
                "difficulty": self._difficulty,
                "seed": self._seed,
            },
        )
