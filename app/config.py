"""
Configuration management for NeuralNexus ML Risk Intelligence Platform.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings — loaded from .env, with sane defaults."""

    app_name: str = "NeuralNexus"
    app_version: str = "2.0.0"
    debug: bool = True

    default_model_type: str = "random_forest"
    random_seed: int = 42

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}


# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR      = BASE_DIR / "artifacts"
MODELS_DIR         = ARTIFACTS_DIR / "models"
METRICS_DIR        = ARTIFACTS_DIR / "metrics"
PLOTS_DIR          = ARTIFACTS_DIR / "plots"
MONITORING_DIR     = ARTIFACTS_DIR / "monitoring"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR, MONITORING_DIR):
    _d.mkdir(parents=True, exist_ok=True)

settings = Settings()
