# Stock Dashboard Configuration Settings
# Based on project_context.md requirements

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Parameters
API_CONFIG = {
    "max_retries": 3,
    "retry_backoff_factor": 2,  # exponential backoff
    "request_delay": 1.0,  # seconds between calls
    "timeout": 30,  # seconds per request
    "default_period": "1y",  # 1 year of data
    "rate_limit_per_hour": 2000,  # approximate Yahoo Finance limit
}

# Cache Parameters
CACHE_CONFIG = {
    "cache_directory": str(CACHE_DIR),
    "cache_expiration_hours": 24,
    "cache_format": "csv",  # csv or json
    "metadata_storage": "json",
    "cache_size_limit_gb": 1.0,
    "cache_cleanup_interval_hours": 6,
}

# Logging Parameters
LOGGING_CONFIG = {
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_format": "structured",  # structured or simple
    "log_rotation": "daily",  # daily or size-based
    "log_max_size_mb": 100,
    "log_retention_days": 30,
    "error_alert_threshold_percent": 5,
}

# UI Display Parameters
UI_CONFIG = {
    "stocks_per_page": 10,
    "price_update_frequency_seconds": 60,
    "chart_time_ranges": ["1d", "1mo", "3mo", "6mo", "1y", "max"],
    "decimal_places_prices": 2,
    "refresh_on_focus": True,
    "timezone": "Asia/Kolkata",  # Indian timezone
    "currency": "INR",  # Indian Rupee
    "market_hours": "09:15-15:30 IST",  # NSE/BSE trading hours
}

# Data Validation Parameters
VALIDATION_CONFIG = {
    "required_columns": ["Open", "High", "Low", "Close", "Volume"],
    "price_validation": {
        "min_price": 0.01,
        "max_price": 100000.00,
    },
    "volume_validation": {
        "min_volume": 0,
        "max_volume": 1000000000,  # 1 billion
    },
    "nan_threshold_percent": 10,  # max allowed NaN values
}

# Ticker Configuration
TICKER_CONFIG = {
    "default_suffix": ".NS",  # Default suffix for Indian stocks
    "valid_suffixes": [".NS", ".BO"],
    "index_mappings": {
        # Indian Indices
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "NIFTY_50": "^NSEI",
        "NSEI": "^NSEI",
        "NIFTYBANK": "^NSEBANK",
        "BANKNIFTY": "^NSEBANK",
        "NIFTYIT": "^NSMIDCPIT",
        "NIFTYITINDEX": "^NSMIDCPIT",
        "NIFTYAUTO": "^CNXAUTO",
        "NIFTYFMCG": "^CNXFMCG",
        "NIFTYPHARMA": "^CNXPHARMA",
        "NIFTYPSU": "^CNXPSU",
        "NIFTYMETAL": "^CNXMETAL",
        "NIFTYREALTY": "^CNXREALTY",
        "NIFTYENERGY": "^CNXENERGY",
        "NIFTYINFRA": "^CNXINFRA",
        "NIFTYSMALLCAP": "^NSEMDCP50",
        "NIFTYMIDCAP": "^NSEMDCP100",

        # Global Indices
        "DOW": "^DJI",
        "DOWJONES": "^DJI",
        "SP500": "^GSPC",
        "S&P500": "^GSPC",
        "NASDAQ": "^IXIC",
        "NASDAQ100": "^NDX",
        "FTSE100": "^FTSE",
        "DAX": "^GDAXI",
        "NIKKEI": "^N225",
        "HANG SENG": "^HSI",
        "SHANGHAI": "000001.SS",
        "SENSEX": "^BSESN",
        "BSESN": "^BSESN",
    },
    "common_mappings": {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
        "BHARTIARTL": "BHARTIARTL.NS",
        "ITC": "ITC.NS",
        "KOTAKBANK": "KOTAKBANK.NS",
        "LT": "LT.NS",
    }
}

# Data Source Configuration
DATA_SOURCE_CONFIG = {
    "default_source": "smart_fallback",  # "yfinance", "kaggle", or "smart_fallback"
    "kaggle_data_directory": "kaggle_data/archive",
    "fallback_to_yfinance": True,  # If Kaggle data not available, fallback to yfinance
    "cache_kaggle_data": True,  # Cache loaded Kaggle data in memory
    "smart_fallback": {
        "current_data_priority": "yfinance",  # yfinance first for latest data
        "historical_data_priority": "kaggle",  # kaggle first for historical data
        "pattern_analysis_priority": "yfinance",  # use yfinance for pattern analysis (needs current data)
        "backtest_priority": "kaggle",  # always use kaggle for backtesting
        "max_kaggle_age_days": 7,  # consider kaggle data "old" after this many days (reduced from 30)
    }
}

# Pattern Detection Configuration
PATTERN_CONFIG = {
    "trend_lookback": 5,  # Number of candles to check for trend
    "doji": {
        "body_threshold_pct": 10,  # 10% of range
        "dragonfly_lower_shadow_multiplier": 2,
        "gravestone_upper_shadow_multiplier": 2,
        "long_legged_shadow_multiplier": 1.5
    },
    "hammer": {
        "lower_shadow_multiplier": 2,  # >= 2x body
        "upper_shadow_max_pct": 30,  # <= 0.3x body
        "min_body_pct": 1  # > 1% of range
    },
    "shooting_star": {
        "upper_shadow_multiplier": 2,  # >= 2x body
        "lower_shadow_max_pct": 30,  # <= 0.3x body
        "min_body_pct": 1  # > 1% of range
    },
    "marubozu": {
        "shadow_max_pct": 1,  # <= 1% of range
        "min_body_pct": 5  # > 5% of range
    }
}

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "development":
    LOGGING_CONFIG["log_level"] = "DEBUG"
    API_CONFIG["request_delay"] = 0.5  # faster for development
