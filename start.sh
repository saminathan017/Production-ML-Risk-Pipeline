#!/bin/bash

# ML Risk Pipeline - One-Click Start Script
# Starts the server and opens the frontend in your browser

set -e

echo "🚀 Starting ML Risk Pipeline..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if model exists
if [ ! -f "artifacts/models/model_v20251230_1602.joblib" ]; then
    echo "⚠️  No trained model found!"
    echo "Running quick setup..."
    ./venv/bin/python scripts/download_or_prepare_data.py
    ./venv/bin/python scripts/train_model.py
    ./venv/bin/python scripts/evaluate_model.py
fi

echo "✓ Model ready"
echo ""
echo "🌐 Starting server on http://localhost:8000"
echo "📊 API docs at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait a moment then open browser
(sleep 2 && open http://localhost:8000) &

# Start the server
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
