# ML Risk Pipeline

A production-grade machine learning pipeline for risk scoring with automated training, inference, and real-time monitoring capabilities. Built with FastAPI and scikit-learn, featuring a modern web UI and comprehensive API.

## 🌟 Features

- **Production ML Pipeline**: Complete end-to-end ML workflow from data preprocessing to deployment
- **Real-time Predictions**: Fast, scalable inference API with sub-100ms latency
- **Model Management**: Version-controlled model registry with automatic tracking
- **Performance Monitoring**: Real-time monitoring of predictions, latency, and feature drift
- **Modern Web UI**: Interactive dashboard for predictions and monitoring
- **RESTful API**: Comprehensive FastAPI backend with automatic OpenAPI documentation
- **Model Evaluation**: Detailed metrics, confusion matrices, and performance plots
- **Configurable**: Environment-based configuration with sensible defaults

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **ML/Data**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Configuration**: Pydantic Settings
- **AI Integration**: OpenAI (optional, for explanations)

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd ml_risk_pipeline

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Note: OpenAI API key is optional
```

### 3. Run the Application

```bash
# One-click start (prepares data, trains model, starts server)
./start.sh

# Or run individually:
python scripts/download_or_prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The application will be available at:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 📖 API Documentation

### Health Check

```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "app_name": "ML Risk Pipeline",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Make Prediction

```bash
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [0.5, 1.2, -0.3, 0.8, 1.5]
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.78,
  "risk_level": "high",
  "model_version": "model_v20251230_1602",
  "latency_ms": 12.5
}
```

### Get Model Information

```bash
GET /api/model
```

**Response:**
```json
{
  "model_version": "model_v20251230_1602",
  "model_type": "Random Forest",
  "trained_at": "2025-12-30T16:02:15",
  "feature_count": 5,
  "model_params": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  }
}
```

### Get Model Metrics

```bash
GET /api/metrics
```

**Response:**
```json
{
  "model_version": "model_v20251230_1602",
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90,
    "roc_auc": 0.95
  },
  "evaluated_at": "2025-12-30T16:05:30",
  "dataset_size": 200
}
```

### Get Monitoring Statistics

```bash
GET /api/monitoring/stats
```

**Response:**
```json
{
  "total_predictions": 1234,
  "average_latency_ms": 15.2,
  "prediction_distribution": {
    "0": 587,
    "1": 647
  },
  "feature_statistics": {...}
}
```

## 📁 Project Structure

```
ml_risk_pipeline/
│
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── config.py                 # Configuration management
│   │
│   ├── api/                      # API routes
│   │   ├── __init__.py
│   │   └── routes.py             # API endpoints
│   │
│   ├── ml/                       # Machine learning components
│   │   ├── __init__.py
│   │   ├── train.py              # Model training logic
│   │   ├── inference.py          # Inference engine
│   │   ├── preprocessing.py      # Data preprocessing
│   │   ├── monitoring.py         # Prediction monitoring
│   │   └── registry.py           # Model registry
│   │
│   ├── utils/                    # Utility modules
│   │   ├── __init__.py
│   │   ├── logging.py            # Logging configuration
│   │   └── schemas.py            # Pydantic schemas
│   │
│   └── frontend/                 # Web UI
│       ├── index.html            # Main UI
│       └── styles.css            # Styling
│
├── scripts/                      # Standalone scripts
│   ├── download_or_prepare_data.py  # Data preparation
│   ├── train_model.py            # Model training script
│   ├── evaluate_model.py         # Model evaluation
│   └── run_smoke_test.py         # Integration tests
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data files (gitignored)
│   └── processed/                # Processed data (gitignored)
│
├── artifacts/                    # Model artifacts
│   ├── models/                   # Trained models (gitignored)
│   ├── metrics/                  # Evaluation metrics (gitignored)
│   ├── plots/                    # Visualization plots (gitignored)
│   └── monitoring/               # Monitoring logs (gitignored)
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Example environment variables
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── start.sh                      # One-click start script
└── README.md                     # This file
```

## ⚙️ Configuration

Configuration is managed through environment variables. Copy `.env.example` to `.env` and customize:

### Application Settings

- `APP_NAME`: Application name (default: "ML Risk Pipeline")
- `APP_VERSION`: Version string (default: "1.0.0")
- `DEBUG`: Enable debug mode (default: true)

### API Settings

- `API_HOST`: Server host (default: "0.0.0.0")
- `API_PORT`: Server port (default: 8000)

### Model Settings

- `DEFAULT_MODEL_TYPE`: ML model type (default: "random_forest")
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)

### OpenAI Settings (Optional)

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: "gpt-3.5-turbo")
- `OPENAI_MAX_TOKENS`: Max tokens per request (default: 500)
- `OPENAI_TEMPERATURE`: Temperature for generation (default: 0.7)

## 🔧 Scripts

### Data Preparation

```bash
python scripts/download_or_prepare_data.py
```

Downloads or generates synthetic risk scoring dataset and saves to `data/raw/`.

### Model Training

```bash
python scripts/train_model.py
```

Trains a new model using processed data and saves it with version control to `artifacts/models/`.

### Model Evaluation

```bash
python scripts/evaluate_model.py
```

Evaluates the latest model and generates:
- Performance metrics (JSON)
- Confusion matrix plot
- ROC curve
- Feature importance chart

### Smoke Testing

```bash
python scripts/run_smoke_test.py
```

Runs comprehensive integration tests to verify:
- API endpoints
- Model predictions
- Data processing
- Monitoring functionality

## 🧪 Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run smoke tests
python scripts/run_smoke_test.py
```

## 📊 Monitoring

The pipeline includes built-in monitoring for:

- **Prediction Tracking**: All predictions are logged with timestamps
- **Latency Monitoring**: Response times tracked per request
- **Feature Distribution**: Statistical tracking of input features
- **Prediction Distribution**: Class balance monitoring
- **Model Performance**: Ongoing evaluation metrics

Monitoring data is stored in `artifacts/monitoring/` as JSONL files.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with FastAPI for high-performance API
- scikit-learn for robust machine learning
- OpenAI integration for intelligent explanations

---

**Made with ❤️ for production ML workflows**
