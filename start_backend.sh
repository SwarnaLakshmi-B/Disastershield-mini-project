#!/usr/bin/env bash
# helpful starter script for local dev. Run from project root with:
#    ./backend/start_backend.sh
#
# It will try to activate project venv at ../.venv if found, then start the dev server.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"

if [ -f "$VENV/bin/activate" ]; then
  echo "Activating venv at $VENV"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
else
  echo "No venv found at $VENV - continuing without venv (you can create one at project root)."
fi

echo "Starting backend (Flask) on http://0.0.0.0:5000"
python "$ROOT/backend/app.py"


