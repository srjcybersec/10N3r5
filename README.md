---
title: Code Review RL Environment
emoji: "🧠"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Team Name: 10N3r5

# Team Members: Prakhar Mehta (TL), Shivansh Rohit Jindal & Shreyas Tekawade


# Code Review RL Environment

An **OpenEnv-style** reinforcement learning environment where agents learn to perform **automated code review** at three difficulty levels: style and formatting, logic bugs, and security vulnerabilities. The design targets **dense partial-credit rewards**, **multi-step refinement** (feedback between steps), and a simple **HTTP API** suitable for LLM or RL training loops.

## Why it matters

Code review is a universal bottleneck: it blends pattern matching, reasoning, and risk prioritization. This environment turns that into a **measurable benchmark** with structured observations, actions, and graded feedback—useful for evaluating agents that must read code, cite lines, and choose an appropriate merge verdict.

## Environment design

### Observation space

| Field | Type | Description |
|--------|------|-------------|
| `task_id` | string | Task family (`style_review`, `logic_bug_review`, `security_review`). |
| `task_description` | string | Instructions for the reviewer role. |
| `code_snippet` | string | Code to review (may be a full snippet or diff-like text). |
| `language` | string | Language hint (e.g. `python`, `javascript`). |
| `context` | string (optional) | Short PR / intent description. |
| `step` | integer | Current step index (0-based at reset). |
| `max_steps` | integer | Maximum steps per episode (3). |
| `previous_comments` | array | Comments from the prior step (for refinement). |
| `feedback` | string (optional) | Grader explanation from the previous step. |

### Action space

| Field | Type | Description |
|--------|------|-------------|
| `comments` | array of `ReviewComment` | Line-level (or file-level) findings. |
| `overall_verdict` | enum string | `approve`, `request_changes`, or `comment_only`. |
| `summary` | string | Short narrative summary (1–3 sentences). |
| `confidence` | float | Self-reported confidence in \([0, 1]\). |

Each `ReviewComment` includes optional `line_number`, `category` (`style`, `logic`, `security`, `performance`, `documentation`), `severity` (`info` … `critical`), `message`, and optional `suggestion`.

### Reward function

Per step, the grader returns a **`CodeReviewReward`** with:

- **issue_detection_score** — weighted recall against labeled ground-truth issues (severity-weighted).
- **false_positive_penalty** — penalty mass for comments that do not match any ground-truth issue (stored as penalty amount; higher means more false positives).
- **severity_accuracy** — how close predicted severities are to ground truth on matched issues.
- **verdict_accuracy** — alignment with the expected `overall_verdict`; approving critically vulnerable snippets is penalized heavily when `has_critical_issues` is true.

**Total** is a difficulty-dependent weighted mix of the above, clipped to \([0, 1]\). See `server/graders.py` for the exact weights.

## Tasks

**1. Style review (`style_review`, easy)** — Focus on naming, formatting, PEP 8–style Python issues, and basic JavaScript style. Issues are mostly visible without deep execution reasoning.

**2. Logic bug review (`logic_bug_review`, medium)** — Off-by-one errors, wrong interval boundaries, broken binary search or memoization, pagination mistakes, etc. Expect to reason about control flow and invariants.

**3. Security review (`security_review`, hard)** — CWE-aligned patterns: SQL injection, path traversal, hardcoded secrets, insecure deserialization (`pickle`), SSRF, weak secret comparison, plus **clean** secure examples that should be **approved**.

## Baseline scores

Latest measured runs with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Model | Mean step reward (avg over steps) | Notes |
|------|--------|-----------------------------------|--------|
| style_review | `Qwen/Qwen2.5-72B-Instruct` | `0.600` | Full 3-task run (`success=true`) |
| logic_bug_review | `Qwen/Qwen2.5-72B-Instruct` | `0.375` | Full 3-task run before task-aware tuning |
| security_review | `Qwen/Qwen2.5-72B-Instruct` | `0.805` | Full 3-task run (`success=true`) |
| logic_bug_review | `Qwen/Qwen2.5-72B-Instruct` | `0.922` | Logic-only rerun after prompt/task-focus tuning (`success=true`) |

Latest full-run average observed: `0.593` (style + logic + security, before logic tuning rerun).

Success in the inference script is defined as mean reward \(\geq 0.6\) per task (configurable via `SUCCESS_THRESHOLD` in `inference.py`).

## Quick start

```bash
cd code-review-env
pip install -r requirements.txt
python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860` for the UI and `http://localhost:7860/docs` for Swagger.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Body: `{ "task_id": "...", "seed": null }`. Starts a new episode; returns initial observation. |
| `POST` | `/step` | Body: `{ "action": { ... CodeReviewAction } }`. Returns observation, reward, `done`, and `info`. |
| `GET` | `/state` | Full `EnvironmentState` for debugging. |
| `GET` | `/tasks` | Lists tasks, difficulties, and descriptions. |
| `GET` | `/health` | Liveness / version JSON. |
| `GET` | `/` | Serves the minimal dashboard (`ui/index.html`). |

## Citation and license

If you use this environment in research or benchmarks, please cite the repository URL and the OpenEnv / hackathon context as appropriate.

Licensed under the **MIT License** (see repository root if a `LICENSE` file is added; otherwise treat as MIT by author intent for hackathon submission).

## Project layout

- `server/` — FastAPI app, env, graders, Pydantic models, tasks, and `data/*.json` case banks.
- `ui/` — Static dashboard.
- `tests/` — Smoke tests for env and grader.
- `inference.py` — Example loop calling the HTTP API and an OpenAI-compatible LLM.
