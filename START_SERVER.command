#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

# ML Risk Pipeline - Double-Click to Start
echo "🚀 ML Risk Pipeline - Starting..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Running first-time setup..."
    echo ""
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo ""
    echo "✓ Virtual environment created"
fi

# Check if model exists
if [ ! -f "artifacts/models/model_v20251230_1602.joblib" ]; then
    echo "⚠️  No trained model found!"
    echo "Running setup pipeline..."
    echo ""
    ./venv/bin/python scripts/download_or_prepare_data.py
    ./venv/bin/python scripts/train_model.py
    ./venv/bin/python scripts/evaluate_model.py
    echo ""
fi

echo "✓ Everything ready!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Server starting at: http://localhost:8000"
echo "📊 API Documentation: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Browser will open automatically in 2 seconds..."
echo ""
echo "⚠️  Press Ctrl+C to stop the server"
echo ""

# Wait then open browser
(sleep 2 && open http://localhost:8000) &

# Start the server
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
