"""
FastAPI server exposing the OpenEnv HTTP interface.

Endpoints:
  POST /reset          — Start a new episode, return initial observation
  POST /step           — Take an action, return (obs, reward, done, info)
  GET  /state          — Return current environment state
  GET  /tasks          — List available tasks
  GET  /health         — Health check
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from .models import CodeReviewAction, CodeReviewObservation, EnvironmentState
from .env import CodeReviewEnv, TASK_REGISTRY

app = FastAPI(
    title="Code Review RL Environment",
    description="An OpenEnv-compliant environment for training agents to review code like a senior engineer.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global env instance (fine for demo; use sessions for production)
_env: Optional[CodeReviewEnv] = None


class ResetRequest(BaseModel):
    task_id: str = "style_review"
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    observation: CodeReviewObservation
    done: bool = False


class StepRequest(BaseModel):
    action: CodeReviewAction


class StepResponse(BaseModel):
    observation: CodeReviewObservation
    reward: float
    done: bool
    info: Dict[str, Any]


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    global _env
    try:
        _env = CodeReviewEnv(task_id=request.task_id, seed=request.seed)
        obs = _env.reset()
        return ResetResponse(observation=obs, done=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        obs, reward, done, info = _env.step(request.action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvironmentState)
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": tid,
                "difficulty": TASK_REGISTRY[tid][1],
                "description": TASK_REGISTRY[tid][0].DESCRIPTION if hasattr(TASK_REGISTRY[tid][0], "DESCRIPTION") else "",
            }
            for tid in TASK_REGISTRY
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# Serve the minimal UI
try:
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

    @app.get("/")
    def root():
        return FileResponse("ui/index.html")
except Exception:

    @app.get("/")
    def root_fallback():
        return {"message": "Code Review RL Environment is running. See /docs for API."}


if __name__ == "__main__":
    uvicorn.run("server.main:app", host="0.0.0.0", port=7860, reload=False)
