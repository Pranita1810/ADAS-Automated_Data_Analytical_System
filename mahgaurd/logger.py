"""
Mahgaurd - Centralized Logging Subsystem for ADAS.
Provides unified, thread-safe, rotating file and console logging across all services.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Base directory for mahgaurd logs
MAHGAURD_DIR = Path(__file__).resolve().parent
LOGS_DIR = MAHGAURD_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG_FILE = LOGS_DIR / "adas.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "ADAS", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a configured logger with console and rotating file output.

    Args:
        name: The component or module name (e.g., 'ADAS.AI', 'ADAS.Engine', 'ADAS.API').
        log_file: Optional path to a specific log file. Defaults to mahgaurd/logs/adas.log.
        level: Logging level (default logging.INFO).

    Returns:
        logging.Logger: Pre-configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger was already created
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Central Rotating File Handler (Max 10MB per file, up to 5 backups)
    target_log_file = log_file or DEFAULT_LOG_FILE
    try:
        file_handler = RotatingFileHandler(
            target_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if file cannot be opened (e.g. read-only filesystem)
        logger.warning(f"Failed to initialize rotating file handler at {target_log_file}: {e}")

    # Do not propagate to root logger to avoid double logging
    logger.propagate = False
    return logger
