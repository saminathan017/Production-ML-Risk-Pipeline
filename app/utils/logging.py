"""
Logging utilities for the ML Risk Pipeline.
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


# Default application logger
app_logger = setup_logger("ml_risk_pipeline")
