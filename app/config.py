"""
Configuration management for the ML Risk Pipeline.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration (Optional)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 500
    openai_temperature: float = 0.7
    
    # Application Configuration
    app_name: str = "ML Risk Pipeline"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # Model Configuration
    default_model_type: str = "random_forest"
    random_seed: int = 42
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MONITORING_DIR = ARTIFACTS_DIR / "monitoring"

# Ensure directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    MONITORING_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
