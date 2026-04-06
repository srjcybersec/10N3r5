"""
Typed models for the Code Review RL Environment.
All models inherit from pydantic BaseModel and are fully serializable.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ReviewCategory(str, Enum):
    STYLE = "style"
    LOGIC = "logic"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ReviewComment(BaseModel):
    line_number: Optional[int] = Field(None, description="Line number the comment applies to (1-indexed). None means file-level.")
    category: ReviewCategory
    severity: Severity
    message: str = Field(..., description="Human-readable review comment.")
    suggestion: Optional[str] = Field(None, description="Optional suggested fix.")


class CodeReviewAction(BaseModel):
    """
    The action an agent takes: submit a code review.

    An agent receives a diff/code snippet in its observation and must
    produce a list of ReviewComments plus an overall verdict.
    """
    comments: List[ReviewComment] = Field(default_factory=list, description="List of review comments on the code.")
    overall_verdict: str = Field(..., description="One of: 'approve', 'request_changes', 'comment_only'")
    summary: str = Field(..., description="A brief summary of the review (1-3 sentences).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's self-reported confidence in the review.")


class CodeReviewObservation(BaseModel):
    """
    What the agent sees each step.
    """
    task_id: str
    task_description: str
    code_snippet: str = Field(..., description="The code or diff to review.")
    language: str = Field(..., description="Programming language of the code.")
    context: Optional[str] = Field(None, description="Additional context: what the PR is trying to do.")
    step: int
    max_steps: int
    previous_comments: List[ReviewComment] = Field(default_factory=list, description="Comments from previous steps (for multi-step refinement).")
    feedback: Optional[str] = Field(None, description="Feedback on the previous action, if any.")


class CodeReviewReward(BaseModel):
    """
    Reward signal. Provides dense partial-credit signal over the full episode.
    """
    total: float = Field(..., ge=0.0, le=1.0, description="Normalized total reward for this step.")
    issue_detection_score: float = Field(..., ge=0.0, le=1.0, description="How many real issues were caught.")
    false_positive_penalty: float = Field(..., ge=0.0, le=1.0, description="Penalty for flagging things that are not issues.")
    severity_accuracy: float = Field(..., ge=0.0, le=1.0, description="How accurately severity was assigned.")
    verdict_accuracy: float = Field(..., ge=0.0, le=1.0, description="Whether approve/reject was correct.")
    explanation: str = Field(..., description="Human-readable breakdown of reward components.")


class EnvironmentState(BaseModel):
    task_id: str
    step: int
    done: bool
    current_observation: Optional[CodeReviewObservation] = None
    cumulative_reward: float
    episode_rewards: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
