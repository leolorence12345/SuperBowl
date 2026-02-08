#!/usr/bin/env bash
# Run the FastAPI backend from the NFL project root so it can import understanding.py
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
uvicorn api:app --reload --port 8000
