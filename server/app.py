"""
OpenEnv multi-mode entry: ASGI app and console script target.

Validators expect `server/app.py` with `main()`, `[project.scripts] server = server.app:main`,
and `if __name__ == "__main__": main()`.
"""
from __future__ import annotations

from .main import app

__all__ = ["app", "main"]


def main() -> None:
    """Console script entry: `server` → run uvicorn on port 7860."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
