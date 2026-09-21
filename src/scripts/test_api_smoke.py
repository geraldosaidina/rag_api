"""Real HTTP smoke test for Slice 3.

Starts the FastAPI app with the real RAG stack and issues HTTP requests.

Run from repository root:
  uv run python src/scripts/test_api_smoke.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from api.app import create_app


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("Creating FastAPI app with real RAG lifecycle initialization...")
    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health")
        print(f"GET /health -> {health.status_code} {health.json()}")
        if health.status_code != 200:
            return 1

        ready = client.get("/ready")
        print(f"GET /ready -> {ready.status_code} {ready.json()}")
        if ready.status_code != 200:
            print("API is not ready; aborting smoke test.")
            return 1

        supported = {
            "question": "Como funciona Retrieval-Augmented Generation?"
        }
        print("\nPOST /api/v1/ask (supported question)")
        print(json.dumps(supported, ensure_ascii=False))
        response = client.post("/api/v1/ask", json=supported)
        print(f"status={response.status_code}")
        payload = response.json()
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
        if response.status_code != 200:
            return 1
        if not payload.get("answer"):
            print("Missing answer.")
            return 1
        if payload.get("diagnostics", {}).get("insufficient_evidence") is True:
            print("Unexpected insufficient_evidence for supported question.")
            return 1

        unsupported = {
            "question": "Qual é a capital da Austrália segundo os PFCs indexados?"
        }
        print("\nPOST /api/v1/ask (unsupported question)")
        print(json.dumps(unsupported, ensure_ascii=False))
        response2 = client.post("/api/v1/ask", json=unsupported)
        print(f"status={response2.status_code}")
        payload2 = response2.json()
        print(json.dumps(payload2, ensure_ascii=False, indent=2)[:2000])
        if response2.status_code != 200:
            return 1

        print("\nHTTP smoke test completed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
