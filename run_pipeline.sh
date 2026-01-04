#!/bin/bash

# Production ML Pipeline - Quick Start Script
# This script runs the complete pipeline from start to finish

set -e  # Exit on error

echo "=========================================="
echo "ML Risk Pipeline - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Step 1: Prepare data
echo "📊 Step 1: Preparing data..."
./venv/bin/python scripts/download_or_prepare_data.py
echo ""

# Step 2: Train models
echo "🎯 Step 2: Training models..."
./venv/bin/python scripts/train_model.py
echo ""

# Step 3: Evaluate models
echo "📈 Step 3: Evaluating models..."
./venv/bin/python scripts/evaluate_model.py
echo ""

# Step 4: Run smoke tests
echo "🧪 Step 4: Running smoke tests..."
./venv/bin/python scripts/run_smoke_test.py
echo ""

# Done
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start the API server:"
echo "   ./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "2. Open your browser:"
echo "   http://localhost:8000"
echo ""
echo "3. View API docs:"
echo "   http://localhost:8000/docs"
echo ""
