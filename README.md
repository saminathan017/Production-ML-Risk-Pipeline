# NeuralNexus

A production-grade machine learning pipeline for binary risk classification. Covers the full workflow — data preparation, model training, evaluation, a REST + WebSocket inference API, real-time monitoring, and a web dashboard.

Built with FastAPI and scikit-learn. Runs entirely on your local machine with no cloud or Docker required.

---

## Table of Contents

- [About](#about)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [ML Pipeline](#ml-pipeline)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Scripts](#scripts)
- [Contributing](#contributing)
- [License](#license)

---

## About

NeuralNexus was built to understand what it takes to ship a machine learning model beyond a notebook. The goal was a system where you can train, evaluate, version, and serve a model — and then watch it work in real time through a dashboard.

The dataset used is the [scikit-learn breast cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) — 569 samples, 30 features, binary labels (malignant / benign). It serves as a stand-in for any binary risk classification problem. The labels follow scikit-learn's convention: `0 = malignant`, `1 = benign`.

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| API framework | FastAPI | 0.115.0 |
| ASGI server | Uvicorn | 0.32.0 |
| ML | scikit-learn | 1.5.0 |
| Data | pandas, numpy | 2.2.3, 2.0.0 |
| Serialization | joblib | 1.4.0 |
| Validation | Pydantic v2 | 2.10.0 |
| Plots | matplotlib, seaborn | 3.9.0, 0.13.2 |
| Config | python-dotenv | 1.0.1 |
| Frontend | Vanilla JS, Chart.js | — |

Python 3.9 or higher is required.

---

## Project Structure

```
ml_risk_pipeline/
│
├── app/
│   ├── main.py                  # FastAPI app — CORS, routes, startup
│   ├── config.py                # Settings from .env, directory paths
│   │
│   ├── api/
│   │   └── routes.py            # All 11 endpoints + WebSocket
│   │
│   ├── ml/
│   │   ├── data_loader.py       # Load CSV or fall back to sklearn dataset
│   │   ├── preprocessing.py     # StandardScaler, stratified 80/20 split
│   │   ├── train.py             # Three model trainers (LogReg, RF, GB)
│   │   ├── inference.py         # Singleton inference engine
│   │   ├── evaluate.py          # Metrics, confusion matrix, ROC curve
│   │   ├── monitoring.py        # Append-only JSONL prediction logger
│   │   └── registry.py          # Model versioning and registry
│   │
│   ├── utils/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── logging.py           # Logger setup
│   │
│   └── frontend/
│       ├── index.html           # Single-page dashboard
│       ├── styles.css           # Dark theme with neon accents
│       └── app.js               # Charts, particle canvas, WebSocket client
│
├── scripts/
│   ├── download_or_prepare_data.py   # Prepare and split the dataset
│   ├── train_model.py                # Train all three model variants
│   ├── evaluate_model.py             # Compute metrics and save plots
│   └── run_smoke_test.py             # Integration tests against live API
│
├── data/
│   ├── raw/                     # Raw input files (gitignored)
│   └── processed/               # Scaled train/test splits (gitignored)
│
├── artifacts/
│   ├── models/                  # Saved .joblib models + registry.json
│   ├── metrics/                 # Per-model evaluation JSON
│   ├── plots/                   # Confusion matrix and ROC curve PNGs
│   └── monitoring/              # Daily prediction logs (JSONL)
│
├── requirements.txt
├── .env.example
├── start.sh
└── README.md
```

---

## Getting Started

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/neuralnexus.git
cd neuralnexus

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Defaults work as-is — edit only if you want to change the port, seed, etc.
```

### 3. Start

```bash
./start.sh
```

This script checks for a trained model. If none exists, it runs the full pipeline (prepare → train → evaluate) before starting the server. If a model already exists, it skips straight to the server.

```bash
./start.sh --train     # force a full retrain even if a model exists
./start.sh --no-open   # skip auto-opening the browser
```

Once running:

| URL | Description |
|---|---|
| `http://localhost:8000` | Web dashboard |
| `http://localhost:8000/docs` | Swagger UI (interactive API docs) |
| `http://localhost:8000/redoc` | ReDoc API reference |

### Manual step-by-step (alternative to start.sh)

```bash
python scripts/download_or_prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
python -m uvicorn app.main:app --reload
```

---

## API Reference

The API has 11 endpoints. All are prefixed with `/api`.

### Endpoints overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/predict` | Single prediction |
| `POST` | `/api/predict/batch` | Batch prediction (up to 100 samples) |
| `GET` | `/api/model` | Active model metadata |
| `GET` | `/api/models/list` | All registered models |
| `GET` | `/api/metrics` | Evaluation metrics for the active model |
| `GET` | `/api/feature-importance` | Top feature importances |
| `GET` | `/api/monitoring/stats` | Aggregated prediction statistics |
| `GET` | `/api/predictions/recent` | Last N prediction log entries |
| `GET` | `/api/system/status` | Runtime info (uptime, platform, connections) |
| `WS` | `/api/ws/live` | WebSocket — streams every new prediction |

---

### `GET /api/health`

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "healthy",
  "app_name": "NeuralNexus",
  "version": "2.0.0",
  "model_loaded": true
}
```

---

### `POST /api/predict`

Accepts a 30-element feature vector matching the breast cancer dataset feature order. See `data/processed/metadata.json` for the full feature name list.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
      0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
      0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
      0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
  }'
```

```json
{
  "prediction": 0,
  "probability": 0.042,
  "risk_level": "High",
  "model_version": "model_v20251230_1602",
  "latency_ms": 3.7
}
```

**Risk level thresholds** — based on the probability of class 1 (benign):

| Probability of benign | Risk level |
|---|---|
| < 0.30 | High |
| 0.30 – 0.70 | Medium |
| > 0.70 | Low |

---

### `POST /api/predict/batch`

```bash
curl -X POST http://localhost:8000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
       0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
       0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
       0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
      [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
       0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146,
       0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2,
       0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
    ]
  }'
```

```json
{
  "results": [
    {"prediction": 0, "probability": 0.042, "risk_level": "High", "latency_ms": 3.1},
    {"prediction": 1, "probability": 0.963, "risk_level": "Low",  "latency_ms": 1.8}
  ],
  "count": 2,
  "total_latency_ms": 4.9,
  "model_version": "model_v20251230_1602"
}
```

Maximum 100 samples per request.

---

### `GET /api/feature-importance`

Returns the top 20 feature importances. For Gradient Boosting and Random Forest this comes from `feature_importances_`. For Logistic Regression it uses the absolute value of `coef_`.

```bash
curl http://localhost:8000/api/feature-importance
```

```json
{
  "feature_importance": [
    {"feature": "worst concave points", "importance": 0.142},
    {"feature": "worst perimeter",      "importance": 0.118},
    {"feature": "mean concave points",  "importance": 0.097}
  ],
  "model_type": "gradient_boosting",
  "total_features": 30
}
```

---

### `WS /api/ws/live`

Connect once and receive a JSON message for every prediction made through `/api/predict`. A heartbeat is sent every 25 seconds to keep the connection open.

```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/live');

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  // msg.event: "connected" | "prediction" | "heartbeat"
};
```

A prediction event:

```json
{
  "event": "prediction",
  "timestamp": "2026-04-18T14:32:01.123456",
  "prediction": 0,
  "probability": 0.042,
  "risk_level": "High",
  "model_version": "model_v20251230_1602",
  "latency_ms": 3.7
}
```

---

## ML Pipeline

### Dataset

| Property | Value |
|---|---|
| Source | `sklearn.datasets.load_breast_cancer` |
| Samples | 569 |
| Features | 30 |
| Train / Test | 455 / 114 (80 / 20, stratified) |
| Class 0 | 212 (malignant) |
| Class 1 | 357 (benign) |
| Hash | `f8540222611e6e86` |

The 30 features are the mean, standard error, and "worst" (largest) value of 10 cell nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

### Models trained

Three variants are trained and saved during `train_model.py`. The evaluate script scores all three on the held-out test set and registers the best one by F1 score as `approved` in `artifacts/models/registry.json`. The inference engine loads the approved model on startup.

| Model | Role |
|---|---|
| Logistic Regression | Baseline — interpretable, fast |
| Random Forest | Default — good out-of-the-box performance |
| Gradient Boosting | Best performer in most runs |

### Preprocessing

Features are scaled with `StandardScaler` fitted on the training set only, then applied to both splits. The fitted scaler is saved as `artifacts/models/preprocessor.joblib` and loaded at inference time so predictions use the same scaling as training.

### Evaluation results (current approved model)

Model: Gradient Boosting — `model_v20251230_1602`

| Metric | Score |
|---|---|
| Accuracy | 93.86% |
| Precision | 95.77% |
| Recall | 94.44% |
| F1 | 95.10% |
| ROC-AUC | 98.64% |

Evaluated on 114 held-out test samples.

---

## Dashboard

The frontend is a single-page app in `app/frontend/`. No build step, no framework — plain HTML, CSS, and JavaScript.

**Panels and charts:**

- Header bar — accuracy, ROC-AUC, total predictions, average latency, model name, live clock
- Prediction terminal — paste a feature vector or use a preset (malignant / benign / random), run inference, see the result with a confidence bar
- Model intelligence panel — algorithm, version, training date, feature count, per-metric progress bars
- Radar chart — all five metrics plotted together
- Risk doughnut chart — Low / Medium / High breakdown from the prediction history
- Bar chart — accuracy, precision, recall, F1, ROC-AUC side by side
- Feature importance panel — top 10 features with animated fill bars
- Live prediction feed — updates in real time via WebSocket

The background runs two canvas animations: a particle field that draws connecting lines between nearby nodes, and a layered neural network diagram with animated data packets moving through it.

---

## Configuration

Copy `.env.example` to `.env`. All settings have defaults that work without any changes.

```env
APP_NAME=NeuralNexus
APP_VERSION=2.0.0
DEBUG=true

API_HOST=0.0.0.0
API_PORT=8000

DEFAULT_MODEL_TYPE=random_forest
RANDOM_SEED=42
```

`DEFAULT_MODEL_TYPE` sets the model used when training a single variant. Valid values: `logistic_regression`, `random_forest`, `gradient_boosting`.

`RANDOM_SEED` is passed to the train/test split and to all sklearn estimators that accept a `random_state` parameter.

---

## Monitoring

Every call to `POST /api/predict` appends one line to a daily log file:

```
artifacts/monitoring/predictions_YYYYMMDD.jsonl
```

Each log entry:

```json
{
  "timestamp": "2026-04-18T14:32:01.123456",
  "model_version": "model_v20251230_1602",
  "prediction": 0,
  "probability": 0.042,
  "risk_level": "High",
  "latency_ms": 3.7,
  "feature_stats": {
    "mean": 0.431,
    "std": 0.298,
    "min": 0.006,
    "max": 2.019,
    "count": 30
  }
}
```

`GET /api/monitoring/stats` aggregates recent log entries into prediction distribution, risk breakdown, and latency percentiles (mean, median, p95, p99).

`GET /api/predictions/recent` returns the last N raw log entries, newest first.

---

## Scripts

| Script | What it does |
|---|---|
| `download_or_prepare_data.py` | Loads the breast cancer dataset, fits and applies StandardScaler, saves train/test splits and `metadata.json` to `data/processed/` |
| `train_model.py` | Trains all three model variants, saves timestamped `.joblib` files to `artifacts/models/` |
| `evaluate_model.py` | Scores each model on the test set, saves a metrics JSON file and plots (confusion matrix, ROC curve) to `artifacts/` |
| `run_smoke_test.py` | Integration test — the server must be running; hits key API endpoints and checks the responses |

---

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes and test them: `python scripts/run_smoke_test.py`
4. Commit with a clear message: `git commit -m "add: what you changed and why"`
5. Push and open a pull request

If you find a bug, open an issue with the steps to reproduce it.

---

## License

MIT. See [LICENSE](LICENSE) for the full text.

---

*Built to learn production ML engineering — from data pipelines and model registries to REST APIs, WebSockets, and real-time dashboards.*
