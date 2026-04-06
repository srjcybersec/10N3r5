"""
Graders for each task difficulty.

Scoring philosophy:
  - Binary correct/incorrect is NOT enough. We reward partial matches.
  - Issue detection: each real issue caught = points. Missing issues = deductions.
  - False positives: flagging things that are NOT issues = penalty.
  - Severity accuracy: critical missed vs. info missed are penalized differently.
  - Verdict accuracy: approving buggy code is worse than rejecting clean code.

Each grader returns a CodeReviewReward with total in [0.0, 1.0].
"""
from __future__ import annotations
import re
from .models import (
    CodeReviewAction,
    CodeReviewReward,
    ReviewComment,
    Severity,
)

_SEVERITY_WEIGHT = {
    Severity.INFO: 0.5,
    Severity.WARNING: 1.0,
    Severity.ERROR: 2.0,
    Severity.CRITICAL: 3.0,
}


def _normalize(val: float, max_val: float) -> float:
    if max_val == 0:
        return 1.0
    return max(0.0, min(1.0, val / max_val))


def _comment_matches_issue(comment: ReviewComment, issue: dict, line_tolerance: int = 2) -> bool:
    """
    Fuzzy match: a comment matches a ground truth issue if:
    - The category matches, AND
    - The line number is within tolerance (or both are None), AND
    - The comment message shares meaningful keywords with the issue message.
    """
    if comment.category.value != issue["category"]:
        return False

    # Line number check (with tolerance for off-by-one)
    gt_line = issue.get("line_number")
    if gt_line is not None and comment.line_number is not None:
        if abs(comment.line_number - gt_line) > line_tolerance:
            return False

    # Keyword overlap check (at least 2 meaningful keywords must match)
    comment_words = set(re.findall(r"\b[a-z_]{4,}\b", comment.message.lower()))
    issue_words = set(re.findall(r"\b[a-z_]{4,}\b", issue["message"].lower()))
    overlap = comment_words & issue_words
    return len(overlap) >= 2


def grade_action(
    action: CodeReviewAction,
    ground_truth: dict,
    task_difficulty: str = "easy",
) -> CodeReviewReward:
    gt_issues = ground_truth.get("ground_truth_issues", [])
    correct_verdict = ground_truth.get("correct_verdict", "request_changes")
    has_critical = ground_truth.get("has_critical_issues", False)

    # ── 1. Issue detection score ────────────────────────────────────────────
    matched_issues = set()
    for comment in action.comments:
        for i, issue in enumerate(gt_issues):
            if i not in matched_issues and _comment_matches_issue(comment, issue):
                matched_issues.add(i)

    total_issue_weight = sum(_SEVERITY_WEIGHT[Severity(i["severity"])] for i in gt_issues) or 1.0
    caught_weight = sum(
        _SEVERITY_WEIGHT[Severity(gt_issues[i]["severity"])]
        for i in matched_issues
    )
    issue_detection_score = _normalize(caught_weight, total_issue_weight)

    # ── 2. False positive penalty ────────────────────────────────────────────
    unmatched_comments = [
        c
        for j, c in enumerate(action.comments)
        if not any(
            _comment_matches_issue(c, issue)
            for issue in gt_issues
        )
    ]
    fp_penalty_raw = len(unmatched_comments) * 0.15
    false_positive_score = max(0.0, 1.0 - fp_penalty_raw)

    # ── 3. Severity accuracy ─────────────────────────────────────────────────
    severity_scores = []
    for comment in action.comments:
        for issue in gt_issues:
            if _comment_matches_issue(comment, issue):
                gt_sev = Severity(issue["severity"])
                pred_sev = comment.severity
                weight_diff = abs(
                    _SEVERITY_WEIGHT[gt_sev] - _SEVERITY_WEIGHT[pred_sev]
                )
                max_diff = max(_SEVERITY_WEIGHT.values()) - min(_SEVERITY_WEIGHT.values())
                severity_scores.append(1.0 - weight_diff / max_diff)
                break
    severity_accuracy = sum(severity_scores) / len(severity_scores) if severity_scores else (1.0 if not gt_issues else 0.5)

    # ── 4. Verdict accuracy ──────────────────────────────────────────────────
    verdict_correct = action.overall_verdict == correct_verdict
    # Penalize harder for approving code with critical security issues
    if has_critical and action.overall_verdict == "approve":
        verdict_accuracy = 0.0
    elif verdict_correct:
        verdict_accuracy = 1.0
    else:
        verdict_accuracy = 0.3  # Partial credit for "comment_only" when "request_changes" expected

    # ── 5. Weighted total ────────────────────────────────────────────────────
    weights = {"easy": (0.4, 0.2, 0.2, 0.2), "medium": (0.35, 0.2, 0.2, 0.25), "hard": (0.3, 0.25, 0.15, 0.3)}
    w = weights.get(task_difficulty, weights["easy"])
    total = (
        w[0] * issue_detection_score
        + w[1] * false_positive_score
        + w[2] * severity_accuracy
        + w[3] * verdict_accuracy
    )

    explanation = (
        f"Caught {len(matched_issues)}/{len(gt_issues)} issues "
        f"(detection={issue_detection_score:.2f}). "
        f"False positives: {len(unmatched_comments)} "
        f"(fp_score={false_positive_score:.2f}). "
        f"Severity accuracy={severity_accuracy:.2f}. "
        f"Verdict={'correct' if verdict_correct else 'wrong'} "
        f"(verdict_score={verdict_accuracy:.2f}). "
        f"Total={total:.3f}."
    )

    return CodeReviewReward(
        total=round(total, 4),
        issue_detection_score=round(issue_detection_score, 4),
        false_positive_penalty=round(1.0 - false_positive_score, 4),
        severity_accuracy=round(severity_accuracy, 4),
        verdict_accuracy=round(verdict_accuracy, 4),
        explanation=explanation,
    )
