# Logging Configuration for Stock Dashboard
# Implements comprehensive logging as per project_context.md Level 6

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from config.settings import LOGGING_CONFIG, LOGS_DIR

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""

    def format(self, record):
        # Add timestamp if not present
        if not hasattr(record, 'timestamp'):
            record.timestamp = datetime.utcnow().isoformat()

        # Create structured log entry
        log_entry = {
            "timestamp": record.timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)

def setup_logging():
    """Initialize logging configuration"""

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(LOGGING_CONFIG["log_level"], logging.INFO)
    root_logger.setLevel(log_level)

    # Create formatters
    if LOGGING_CONFIG["log_format"] == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if LOGGING_CONFIG["log_rotation"] == "daily":
        file_handler = logging.handlers.TimedRotatingFileHandler(
            LOGS_DIR / "stock_dashboard.log",
            when="midnight",
            interval=1,
            backupCount=LOGGING_CONFIG["log_retention_days"]
        )
    else:
        file_handler = logging.handlers.RotatingFileHandler(
            LOGS_DIR / "stock_dashboard.log",
            maxBytes=LOGGING_CONFIG["log_max_size_mb"] * 1024 * 1024,
            backupCount=10
        )

    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Create specific loggers for different components
    loggers = {
        "yfinance_api": logging.getLogger("yfinance_api"),
        "cache": logging.getLogger("cache"),
        "validation": logging.getLogger("validation"),
        "performance": logging.getLogger("performance"),
    }

    # Ensure all loggers use the root configuration
    for logger in loggers.values():
        logger.setLevel(log_level)

    return loggers

# Global logger instances
loggers = setup_logging()

def log_api_call(operation, ticker, start_time, end_time=None, success=True, error=None, extra_data=None):
    """Log API call with timing and result"""
    duration = None
    if end_time:
        duration = (end_time - start_time).total_seconds()

    log_data = {
        "operation": operation,
        "ticker": ticker,
        "duration_seconds": duration,
        "success": success,
    }

    if error:
        log_data["error"] = str(error)
        log_data["error_type"] = type(error).__name__

    if extra_data:
        log_data.update(extra_data)

    if success:
        loggers["yfinance_api"].info("API call completed", extra={"extra_data": log_data})
    else:
        loggers["yfinance_api"].error("API call failed", extra={"extra_data": log_data})

def log_cache_operation(operation, ticker, cache_hit=False, cache_age_hours=None):
    """Log cache operations"""
    log_data = {
        "operation": operation,
        "ticker": ticker,
        "cache_hit": cache_hit,
        "cache_age_hours": cache_age_hours,
    }

    loggers["cache"].info("Cache operation", extra={"extra_data": log_data})

def log_performance_metric(metric_name, value, unit="seconds"):
    """Log performance metrics"""
    log_data = {
        "metric": metric_name,
        "value": value,
        "unit": unit,
    }

    loggers["performance"].info("Performance metric", extra={"extra_data": log_data})
