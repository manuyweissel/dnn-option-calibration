"""
Structured logging utilities for experiment tracking.

Provides console and JSON file logging with context managers for timing operations.
Follows structured logging best practices for machine-parseable log streams.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import torch


def setup_logging(
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    log_name: str = "experiment",
) -> logging.Logger:
    """
    Set up structured logger with console and optional file output.
    
    Args:
        log_file: Path to log file (None = console only).
        level: Logging level (default: INFO).
        log_name: Logger name.
        
    Returns:
        Configured logger instance.
        
    Example:
        >>> logger = setup_logging("runs/exp001/logs/run.log")
        >>> logger.info("Starting experiment")
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers
    
    # Console handler with readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler with JSON format (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_fmt = JSONFormatter()
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON lines."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_obj = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        
        return json.dumps(log_obj)


@contextmanager
def timer(logger: logging.Logger, operation: str, extra: Optional[dict[str, Any]] = None):
    """
    Context manager for timing operations.
    
    Args:
        logger: Logger instance.
        operation: Operation description.
        extra: Optional extra fields for logging.
        
    Yields:
        None
        
    Example:
        >>> with timer(logger, "data_loading"):
        ...     data = load_data()
        # Logs: "data_loading completed in 1.234s"
    """
    start = time.perf_counter()
    extra = extra or {}
    logger.info(f"{operation} started", extra={"operation": operation, **extra})
    
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info(
            f"{operation} completed",
            extra={"operation": operation, "duration_s": f"{duration:.3f}", **extra},
        )


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system and environment information.
    
    Args:
        logger: Logger instance.
        
    Example:
        >>> log_system_info(logger)
        # Logs Python version, PyTorch version, CUDA availability, etc.
    """
    import platform
    
    info = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    
    logger.info("System information", extra=info)


def calculate_duration_ms(start_time: float) -> int:
    """
    Calculate duration in milliseconds from start time.
    
    Args:
        start_time: Start time from time.perf_counter().
        
    Returns:
        Duration in milliseconds (integer).
        
    Example:
        >>> start = time.perf_counter()
        >>> # ... do work ...
        >>> duration_ms = calculate_duration_ms(start)
    """
    return int((time.perf_counter() - start_time) * 1000)

