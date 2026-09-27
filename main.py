"""PFC Assistant API entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from api.app import create_app

app = create_app(
    evaluation_log_path=os.getenv(
        "EVALUATION_LOG_PATH",
        "data/evaluation/interactions.jsonl",
    ),
)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
