#!/usr/bin/env bash
# Starts both the Flask prediction server and the Next.js dev server.
set -e
cd "$(dirname "$0")"

# Use the plant-disease conda env if available, otherwise fall back to system python3
PYTHON="/Volumes/JadSSD/conda_envs/plant-disease/bin/python3"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(which python3)"
fi
echo "Using Python: $PYTHON"

# ── Python prediction server ──
echo "Starting prediction server on port 8001…"
"$PYTHON" predict_server.py &
PYTHON_PID=$!

# Wait until Flask is accepting connections (max 120 s — TF model load takes time)
echo "Waiting for prediction server to be ready (this may take ~30 s while TensorFlow loads)…"
for i in $(seq 1 120); do
  if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "Prediction server ready."
    break
  fi
  sleep 1
done

# ── Next.js ──
echo "Starting Next.js on http://localhost:3000"
npm run dev &
NEXT_PID=$!

# Kill both on Ctrl-C or script exit
trap 'echo "Shutting down…"; kill "$PYTHON_PID" "$NEXT_PID" 2>/dev/null; exit' SIGINT SIGTERM EXIT

wait $NEXT_PID
