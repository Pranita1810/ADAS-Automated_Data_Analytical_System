"""
Mahgaurd: Observability, Metrics & Centralized Logging Subsystem for ADAS.
"""

from mahgaurd.logger import get_logger, DEFAULT_LOG_FILE, LOGS_DIR
from mahgaurd.observability import (
    timed,
    observe,
    metrics,
    MetricsCollector,
    get_system_health,
    get_metrics_summary,
    get_recent_logs,
)

# Helper function
def get_metrics():
    return metrics.get_summary()

__all__ = [
    "get_logger",
    "DEFAULT_LOG_FILE",
    "LOGS_DIR",
    "timed",
    "observe",
    "metrics",
    "MetricsCollector",
    "get_system_health",
    "get_metrics",
    "get_recent_logs",
]
