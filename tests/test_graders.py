"""Tests for the grader's scoring logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.graders import grade_action
from server.models import CodeReviewAction, ReviewComment, ReviewCategory, Severity

SAMPLE_GT = {
    "ground_truth_issues": [
        {"line_number": 1, "category": "style", "severity": "warning",
         "message": "Function name should follow snake_case convention"},
    ],
    "correct_verdict": "request_changes",
    "has_critical_issues": False,
}

def test_perfect_review_scores_high():
    action = CodeReviewAction(
        comments=[ReviewComment(
            line_number=1, category=ReviewCategory.STYLE, severity=Severity.WARNING,
            message="Function name should follow snake_case convention",
        )],
        overall_verdict="request_changes",
        summary="Style issue detected.",
        confidence=0.9,
    )
    reward = grade_action(action, SAMPLE_GT, "easy")
    assert reward.total >= 0.7, f"Expected >= 0.7, got {reward.total}"

def test_empty_review_scores_low():
    action = CodeReviewAction(
        comments=[],
        overall_verdict="approve",
        summary="Looks good.",
        confidence=0.5,
    )
    reward = grade_action(action, SAMPLE_GT, "easy")
    assert reward.total < 0.5, f"Expected < 0.5, got {reward.total}"

def test_reward_always_in_range():
    import random
    for _ in range(20):
        action = CodeReviewAction(
            comments=[ReviewComment(
                line_number=random.randint(1,10),
                category=random.choice(list(ReviewCategory)),
                severity=random.choice(list(Severity)),
                message="Some random comment here that might or might not match",
            )],
            overall_verdict=random.choice(["approve", "request_changes", "comment_only"]),
            summary="Test.",
            confidence=random.random(),
        )
        reward = grade_action(action, SAMPLE_GT, "easy")
        assert 0.0 <= reward.total <= 1.0

def test_paraphrased_correct_comment_gets_credit():
    action = CodeReviewAction(
        comments=[ReviewComment(
            line_number=1, category=ReviewCategory.STYLE, severity=Severity.WARNING,
            message="Naming convention is inconsistent; prefer snake_case for this function.",
        )],
        overall_verdict="request_changes",
        summary="Naming style issue found.",
        confidence=0.85,
    )
    reward = grade_action(action, SAMPLE_GT, "easy")
    assert reward.issue_detection_score > 0.0
    assert reward.total >= 0.6

def test_empty_review_with_request_changes_not_overrewarded():
    action = CodeReviewAction(
        comments=[],
        overall_verdict="request_changes",
        summary="Likely issues present.",
        confidence=0.5,
    )
    reward = grade_action(action, SAMPLE_GT, "easy")
    assert reward.total < 0.5, f"Expected < 0.5, got {reward.total}"

if __name__ == "__main__":
    test_perfect_review_scores_high()
    test_empty_review_scores_low()
    test_reward_always_in_range()
    test_paraphrased_correct_comment_gets_credit()
    test_empty_review_with_request_changes_not_overrewarded()
    print("All grader tests passed.")
