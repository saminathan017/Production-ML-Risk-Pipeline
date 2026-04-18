#!/bin/bash
# NeuralNexus — one-command launcher
# Usage: ./start.sh [--train] [--no-open]
#   --train     force full retrain even if a model already exists
#   --no-open   skip auto-opening the browser

set -e

FORCE_TRAIN=false
OPEN_BROWSER=true
PYTHON="./venv/bin/python"

for arg in "$@"; do
  case $arg in
    --train)   FORCE_TRAIN=true ;;
    --no-open) OPEN_BROWSER=false ;;
  esac
done

echo ""
echo "  NeuralNexus — ML Risk Intelligence Platform"
echo "  ─────────────────────────────────────────────"
echo ""

# Check virtualenv
if [ ! -d "venv" ]; then
  echo "  ERROR: virtual environment not found."
  echo "  Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
echo "  [ok] virtual environment"

# Run pipeline if no model exists or --train flag is set
MODEL_EXISTS=$(ls artifacts/models/*.joblib 2>/dev/null | head -1 || true)

if [ -z "$MODEL_EXISTS" ] || [ "$FORCE_TRAIN" = true ]; then
  echo "  [..] running pipeline: prepare -> train -> evaluate"
  echo ""
  $PYTHON scripts/download_or_prepare_data.py
  $PYTHON scripts/train_model.py
  $PYTHON scripts/evaluate_model.py
  echo ""
  echo "  [ok] pipeline complete"
else
  echo "  [ok] model found, skipping pipeline  (use --train to retrain)"
fi

echo ""
echo "  Dashboard  ->  http://localhost:8000"
echo "  API docs   ->  http://localhost:8000/docs"
echo ""
echo "  Ctrl+C to stop"
echo ""

if [ "$OPEN_BROWSER" = true ]; then
  (sleep 2 && open http://localhost:8000 2>/dev/null || true) &
fi

$PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000
