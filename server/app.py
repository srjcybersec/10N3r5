"""
OpenEnv multi-mode entry: ASGI app and console script target.

Validators expect `server/app.py` and a `[project.scripts]` entry named `server`.
"""
from __future__ import annotations

from .main import app

__all__ = ["app", "serve"]


def serve() -> None:
    """Console script entry: `server` → run uvicorn on port 7860."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
