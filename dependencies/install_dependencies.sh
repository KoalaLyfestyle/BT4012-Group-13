#!/bin/bash
# Auto-install dependencies from requirements.txt for this project
REQ_FILE="$(dirname "$0")/requirements.txt"
if [ -f "$REQ_FILE" ]; then
  echo "Installing dependencies from $REQ_FILE..."
  "$(dirname "$0")/../.venv/bin/python" -m pip install -r "$REQ_FILE"
else
  echo "requirements.txt not found in dependencies folder."
  exit 1
fi
