"""
Inference Script — Code Review RL Environment
==============================================

Runs a model against all 3 tasks and emits structured stdout logs.

Environment variables:
  API_BASE_URL   LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key (no default — set in Space secrets or shell)
  LOCAL_IMAGE_NAME  Optional; only if using from_docker_image() workflows
  INFERENCE_DEBUG   Set to 1 for [DEBUG] lines; omit for stdout limited to [START]/[STEP]/[END]
  TASK_NAME      One of: style_review, logic_bug_review, security_review, or 'all'
"""

import os
import json
import textwrap
import requests
from typing import List, Optional
from openai import OpenAI

# Rubric: defaults only for API_BASE_URL and MODEL_NAME; HF_TOKEN must not use getenv(..., default).
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "all")
BENCHMARK = "code_review_env"
MAX_STEPS = 3
SUCCESS_THRESHOLD = 0.6

# Set INFERENCE_DEBUG=1 for extra stderr-style diagnostics; keep unset for strict stdout (START/STEP/END only).
_INFERENCE_DEBUG = os.getenv("INFERENCE_DEBUG", "").lower() in ("1", "true", "yes")

TASKS = ["style_review", "logic_bug_review", "security_review"]

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert senior software engineer performing a code review.

Your task is to analyze the provided code snippet and return a JSON object with this exact schema:
{
  "comments": [
    {
      "line_number": <integer or null>,
      "category": "<style|logic|security|performance|documentation>",
      "severity": "<info|warning|error|critical>",
      "message": "<specific, actionable comment>",
      "suggestion": "<optional concrete fix>"
    }
  ],
  "overall_verdict": "<approve|request_changes|comment_only>",
  "summary": "<1-3 sentence summary of the review>",
  "confidence": <float 0.0 to 1.0>
}

Rules:
- Only flag real issues. Do not invent problems.
- Be specific: include the line number whenever possible.
- If the code is clean, return an empty comments list and verdict "approve".
- Return ONLY the JSON object. No preamble, no markdown, no explanation.
""").strip()


def _task_review_focus(task_id: str) -> str:
    if task_id == "logic_bug_review":
        return (
            "Task focus: logic bug review.\n"
            "Before approving, explicitly verify: loop bounds, off-by-one behavior, boundary conditions, "
            "termination/progress guarantees, and return-value correctness. "
            "If any one of these is wrong, use overall_verdict='request_changes'."
        )
    if task_id == "security_review":
        return (
            "Task focus: security review.\n"
            "Before approving, explicitly verify: SQL injection safety, path handling/traversal, secrets management, "
            "deserialization safety, SSRF controls, and secret comparison timing safety. "
            "If any vulnerability exists, use overall_verdict='request_changes' and severity at least 'error'."
        )
    return (
        "Task focus: style review.\n"
        "Check naming conventions, formatting consistency, and language style-guide compliance."
    )


def _parse_action_json(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action for log readability
    action_short = action[:80].replace("\n", " ") + ("..." if len(action) > 80 else "")
    print(f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def call_env(endpoint: str, method: str = "POST", payload: dict = None) -> dict:
    url = f"{ENV_BASE_URL}/{endpoint}"
    resp = requests.request(method, url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_agent_action(client: OpenAI, obs: dict, step: int) -> dict:
    task_id = obs.get("task_id", "")
    code = obs.get("code_snippet", "")
    lang = obs.get("language", "")
    context = obs.get("context", "")
    feedback = obs.get("feedback", "")
    prev_comments = obs.get("previous_comments", [])
    task_focus = _task_review_focus(task_id)

    user_prompt = textwrap.dedent(f"""
    Task ID: {task_id}
    {task_focus}

    Language: {lang}
    Context: {context}

    Code to review:
    ```{lang}
    {code}
    ```
    """).strip()

    if feedback:
        user_prompt += f"\n\nFeedback on your previous review:\n{feedback}"
    if prev_comments:
        user_prompt += f"\n\nYour previous comments (refine if needed):\n{json.dumps(prev_comments, indent=2)}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        action = _parse_action_json(completion.choices[0].message.content or "")

        # Second-pass safeguard for harder tasks when model is over-approving with no findings.
        if task_id in {"logic_bug_review", "security_review"} and not action.get("comments") and action.get("overall_verdict") == "approve":
            second_pass_prompt = user_prompt + (
                "\n\nRe-check with a strict adversarial mindset. "
                "If there is any plausible bug/vulnerability, return request_changes with concrete line-level comments."
            )
            completion_2 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": second_pass_prompt},
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            action = _parse_action_json(completion_2.choices[0].message.content or "")
        return action
    except Exception as e:
        if _INFERENCE_DEBUG:
            print(f"[DEBUG] Model error: {e}", flush=True)
        return {
            "comments": [],
            "overall_verdict": "comment_only",
            "summary": "Unable to produce review.",
            "confidence": 0.0,
        }


def run_task(client: OpenAI, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = call_env("reset", payload={"task_id": task_id})
        obs = reset_resp["observation"]
        done = reset_resp.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_agent_action(client, obs, step)
            action_str = json.dumps(action_dict)

            step_resp = call_env("step", payload={"action": action_dict})
            obs = step_resp["observation"]
            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", False)
            error = step_resp.get("info", {}).get("error", None)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        if _INFERENCE_DEBUG:
            print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "")

    tasks_to_run = TASKS if TASK_NAME == "all" else [TASK_NAME]

    all_scores = {}
    for task_id in tasks_to_run:
        score = run_task(client, task_id)
        all_scores[task_id] = score
        if _INFERENCE_DEBUG:
            print(f"[DEBUG] {task_id} final score: {score:.3f}", flush=True)

    if _INFERENCE_DEBUG:
        print(f"\n[DEBUG] === SUMMARY ===", flush=True)
        for t, s in all_scores.items():
            print(f"[DEBUG] {t}: {s:.3f}", flush=True)
        avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        print(f"[DEBUG] Average: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
