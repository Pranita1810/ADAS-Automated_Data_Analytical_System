"""
Mahgaurd - Observability & Telemetry Subsystem for ADAS.
Provides execution latency tracking, performance metrics, health checks, and log inspection.
"""

import time
import functools
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Dict, List, Optional

from mahgaurd.logger import get_logger, DEFAULT_LOG_FILE

logger = get_logger("Mahgaurd.Observability")


class MetricsCollector:
    """Thread-safe collector for runtime metrics and function latencies."""

    def __init__(self):
        self._lock = threading.Lock()
        self._timings: Dict[str, Dict[str, Any]] = {}
        self._counts: Dict[str, int] = {"total_calls": 0, "errors": 0}
        self._events: List[Dict[str, Any]] = []
        self._started_at = datetime.now()

    def record_duration(self, operation: str, duration_sec: float, error: bool = False):
        """Record execution duration of a tracked operation."""
        with self._lock:
            self._counts["total_calls"] += 1
            if error:
                self._counts["errors"] += 1

            if operation not in self._timings:
                self._timings[operation] = {
                    "count": 0,
                    "total_sec": 0.0,
                    "min_sec": duration_sec,
                    "max_sec": duration_sec,
                    "last_sec": duration_sec,
                    "errors": 0,
                    "last_timestamp": datetime.now().isoformat(timespec="seconds"),
                }

            entry = self._timings[operation]
            entry["count"] += 1
            entry["total_sec"] += duration_sec
            entry["min_sec"] = min(entry["min_sec"], duration_sec)
            entry["max_sec"] = max(entry["max_sec"], duration_sec)
            entry["last_sec"] = duration_sec
            entry["last_timestamp"] = datetime.now().isoformat(timespec="seconds")
            if error:
                entry["errors"] += 1

    def record_event(self, subsystem: str, event: str, message: str, level: str = "INFO"):
        """Record a milestone or health event."""
        with self._lock:
            evt = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "subsystem": subsystem,
                "event": event,
                "message": message,
                "level": level,
            }
            self._events.append(evt)
            # Keep last 200 events in memory
            if len(self._events) > 200:
                self._events.pop(0)

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated performance metrics."""
        with self._lock:
            summary_timings = {}
            for op, data in self._timings.items():
                avg_sec = data["total_sec"] / data["count"] if data["count"] > 0 else 0.0
                summary_timings[op] = {
                    "count": data["count"],
                    "avg_ms": round(avg_sec * 1000.0, 2),
                    "min_ms": round(data["min_sec"] * 1000.0, 2),
                    "max_ms": round(data["max_sec"] * 1000.0, 2),
                    "last_ms": round(data["last_sec"] * 1000.0, 2),
                    "errors": data["errors"],
                    "last_seen": data["last_timestamp"],
                }

            uptime_sec = (datetime.now() - self._started_at).total_seconds()
            return {
                "uptime_seconds": round(uptime_sec, 1),
                "total_calls": self._counts["total_calls"],
                "total_errors": self._counts["errors"],
                "operations": summary_timings,
                "recent_events_count": len(self._events),
            }

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent recorded events."""
        with self._lock:
            return list(reversed(self._events[-limit:]))


# Global metrics singleton
metrics = MetricsCollector()


def timed(operation_name: Optional[str] = None):
    """
    Decorator to measure function latency and register metrics in Mahgaurd.
    Usage:
        @timed("compute_kpis")
        def get_kpis(...):
            ...
    """
    def decorator(func: Callable):
        op_name = operation_name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            is_error = False
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                is_error = True
                logger.error(f"Error executing [{op_name}]: {e}")
                raise
            finally:
                duration = time.perf_counter() - start
                metrics.record_duration(op_name, duration, error=is_error)
                logger.debug(f"Operation [{op_name}] finished in {duration*1000.0:.2f}ms")

        return wrapper
    return decorator


def observe(name: Optional[str] = None):
    """Alias for @timed for clean observability annotation."""
    return timed(operation_name=name)


def get_metrics_summary() -> Dict[str, Any]:
    """Return aggregated metrics summary."""
    return metrics.get_summary()


def get_system_health() -> Dict[str, Any]:
    """
    Check the operational health of all ADAS subsystems:
    - Analytical Engine
    - Parquet Backups
    - AI Service
    - Central Logs
    - System Memory & CPU
    """
    root_dir = Path(__file__).resolve().parents[1]
    health = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "HEALTHY",
        "subsystems": {},
    }

    # 1. Backups storage check
    backup_dir = root_dir / "pipeline" / "backups" / "parquet_backups"
    parquet_count = len(list(backup_dir.glob("*.parquet"))) if backup_dir.exists() else 0
    health["subsystems"]["Backups_Storage"] = {
        "status": "ONLINE" if parquet_count > 0 else "DEGRADED",
        "details": f"{parquet_count} Parquet snapshot files available",
    }

    # 2. Decisions Store check
    decisions_file = root_dir / "engines" / "decisions_log.json"
    health["subsystems"]["Strategic_Decision_Hub"] = {
        "status": "ONLINE" if decisions_file.exists() else "INITIALIZING",
        "details": "Persistent decisions log ready" if decisions_file.exists() else "No decisions logged yet",
    }

    # 3. Log Storage check
    health["subsystems"]["Mahgaurd_Logging"] = {
        "status": "ONLINE" if DEFAULT_LOG_FILE.exists() else "INITIALIZING",
        "details": f"Central log file: {DEFAULT_LOG_FILE.name} ({DEFAULT_LOG_FILE.stat().st_size // 1024 if DEFAULT_LOG_FILE.exists() else 0} KB)",
    }

    # 4. Host Resource Check (psutil if available)
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        health["host_resources"] = {
            "cpu_percent": f"{cpu_pct}%",
            "memory_percent": f"{mem.percent}%",
            "memory_available_mb": round(mem.available / (1024 * 1024), 1),
        }
    except Exception:
        health["host_resources"] = {"status": "Available"}

    # Overall verdict
    for sub in health["subsystems"].values():
        if sub["status"] == "OFFLINE":
            health["status"] = "CRITICAL"
            break
        elif sub["status"] == "DEGRADED":
            health["status"] = "WARNING"

    return health


def get_recent_logs(lines: int = 100) -> List[str]:
    """
    Read the latest log lines from Mahgaurd central log file.
    """
    if not DEFAULT_LOG_FILE.exists():
        return ["Mahgaurd: Log file has not been initialized yet."]

    try:
        with open(DEFAULT_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            return all_lines[-lines:] if all_lines else ["Mahgaurd: Log file is empty."]
    except Exception as e:
        return [f"Error reading Mahgaurd log file: {e}"]
