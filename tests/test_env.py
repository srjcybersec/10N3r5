"""Basic smoke tests for the environment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import CodeReviewEnv
from server.models import CodeReviewAction, ReviewComment, ReviewCategory, Severity


def test_reset_returns_observation():
    env = CodeReviewEnv(task_id="style_review", seed=42)
    obs = env.reset()
    assert obs.task_id == "style_review"
    assert obs.code_snippet
    assert obs.step == 0


def test_step_returns_reward_in_range():
    env = CodeReviewEnv(task_id="style_review", seed=42)
    env.reset()
    action = CodeReviewAction(
        comments=[ReviewComment(
            line_number=1,
            category=ReviewCategory.STYLE,
            severity=Severity.WARNING,
            message="Function name should be snake_case",
        )],
        overall_verdict="request_changes",
        summary="Minor style issues detected.",
        confidence=0.8,
    )
    obs, reward, done, info = env.step(action)
    assert 0.0 <= reward <= 1.0
    assert "reward_breakdown" in info


def test_episode_ends_after_max_steps():
    env = CodeReviewEnv(task_id="style_review", seed=42)
    env.reset()
    action = CodeReviewAction(
        comments=[],
        overall_verdict="approve",
        summary="Looks good.",
        confidence=0.5,
    )
    for _ in range(env.MAX_STEPS):
        obs, reward, done, info = env.step(action)
    assert done is True


def test_all_tasks():
    for task_id in ["style_review", "logic_bug_review", "security_review"]:
        env = CodeReviewEnv(task_id=task_id, seed=0)
        obs = env.reset()
        assert obs.task_id == task_id
    print("All task tests passed.")


if __name__ == "__main__":
    test_reset_returns_observation()
    test_step_returns_reward_in_range()
    test_episode_ends_after_max_steps()
    test_all_tasks()
    print("All tests passed.")
