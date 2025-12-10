"""
Validation functions for extracted query components.

This module provides functions to validate tickers, timeframes, patterns, and query types
against expected formats and external data sources.
"""

import re
from typing import Optional
import logging
# Import yfinance functions conditionally
try:
    from src.yfinance_wrapper import validate_ticker as format_validate_ticker, InvalidTickerError
    from src.yfinance_wrapper import get_stock_info, YFinanceError
    YFINANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: yfinance not available ({e}), using format validation only")
    YFINANCE_AVAILABLE = False
    # Fallback format validation
    def format_validate_ticker(ticker: str, exchange: str = "nse") -> str:
        """Simple format validation fallback when yfinance not available."""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker")
        ticker = ticker.upper().strip()
        if not ticker.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Invalid ticker format")
        return ticker

    class InvalidTickerError(Exception):
        pass

    def get_stock_info(ticker: str, exchange: str = "nse") -> dict:
        """Mock function when yfinance not available."""
        raise Exception("yfinance not available")

    class YFinanceError(Exception):
        pass

# Configuration for validation
VALIDATION_CONFIG = {
    "valid_timeframes": ["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1mo"],
    "valid_patterns": ["doji", "hammer", "shooting_star", "marubozu"],
    "valid_query_types": ["ohlcv", "pattern"]
}


def validate_ticker(ticker: str, check_yfinance: bool = True, exchange: str = "nse") -> bool:
    """
    Validate ticker symbol by checking format and availability on yfinance.

    Args:
        ticker (str): The ticker symbol to validate
        check_yfinance (bool): Whether to check against yfinance (default: True)

    Returns:
        bool: True if ticker is valid and available on yfinance, False otherwise

    Examples:
        >>> validate_ticker("INFY")
        True
        >>> validate_ticker("INVALID123")
        False
    """
    logging.debug(
        f"DEBUG: validate_ticker() - Function called with ticker: {ticker}, "
        f"check_yfinance: {check_yfinance}, exchange: {exchange}"
    )

    if not ticker or not isinstance(ticker, str):
        logging.debug(f"DEBUG: validate_ticker() - Invalid ticker input: {ticker} (type: {type(ticker)})")
        return False

    try:
        logging.debug(f"DEBUG: validate_ticker() - Starting format/normalization for ticker: {ticker}")
        # First, validate format and normalize via shared helper
        formatted_ticker = format_validate_ticker(ticker, exchange=exchange)
        logging.debug(
            f"DEBUG: validate_ticker() - Format/normalization passed, "
            f"formatted ticker: {formatted_ticker}"
        )

        # For testing purposes, skip yfinance validation if requested
        if not check_yfinance:
            logging.debug("DEBUG: validate_ticker() - Skipping yfinance validation as requested")
            return True

        logging.debug(
            f"DEBUG: validate_ticker() - Attempting yfinance validation for "
            f"ticker: {formatted_ticker} with exchange: {exchange}"
        )
        # Check if data is available by attempting to fetch basic info
        try:
            info = get_stock_info(formatted_ticker, exchange=exchange)
            logging.debug(
                f"DEBUG: validate_ticker() - Yfinance validation successful for "
                f"ticker: {formatted_ticker} on exchange: {exchange}"
            )
            # If we get here without exception, ticker is valid
            return True
        except YFinanceError:
            logging.debug(f"DEBUG: validate_ticker() - Yfinance validation failed (YFinanceError) for ticker: {formatted_ticker}")
            # Could not fetch data, ticker might be invalid
            return False

    except InvalidTickerError:
        logging.debug(f"DEBUG: validate_ticker() - Format validation failed (InvalidTickerError) for ticker: {ticker}")
        return False
    except Exception:
        logging.debug(f"DEBUG: validate_ticker() - Unexpected exception during validation for ticker: {ticker}")
        # Any other error means validation failed
        return False


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate timeframe against allowed values.

    Args:
        timeframe (str): The timeframe string to validate

    Returns:
        bool: True if timeframe is valid, False otherwise

    Examples:
        >>> validate_timeframe("1d")
        True
        >>> validate_timeframe("2d")
        False
    """
    logging.debug(f"DEBUG: validate_timeframe() - Function called with timeframe: {timeframe}")

    if not timeframe or not isinstance(timeframe, str):
        logging.debug(f"DEBUG: validate_timeframe() - Invalid timeframe input: {timeframe} (type: {type(timeframe)})")
        return False

    valid_timeframes = VALIDATION_CONFIG["valid_timeframes"]
    result = timeframe.lower() in valid_timeframes
    logging.debug(f"DEBUG: validate_timeframe() - Validation result: {result} (valid options: {valid_timeframes})")
    return result


def validate_pattern(pattern: str) -> bool:
    """
    Validate pattern name against allowed candlestick patterns.

    Args:
        pattern (str): The pattern name to validate

    Returns:
        bool: True if pattern is valid, False otherwise

    Examples:
        >>> validate_pattern("doji")
        True
        >>> validate_pattern("invalid_pattern")
        False
    """
    logging.debug(f"DEBUG: validate_pattern() - Function called with pattern: {pattern}")

    if not pattern or not isinstance(pattern, str):
        logging.debug(f"DEBUG: validate_pattern() - Invalid pattern input: {pattern} (type: {type(pattern)})")
        return False

    valid_patterns = VALIDATION_CONFIG["valid_patterns"]
    result = pattern.lower() in valid_patterns
    logging.debug(f"DEBUG: validate_pattern() - Validation result: {result} (valid options: {valid_patterns})")
    return result


def validate_query_type(query_type: str) -> bool:
    """
    Validate query type against allowed values.

    Args:
        query_type (str): The query type to validate

    Returns:
        bool: True if query type is valid, False otherwise

    Examples:
        >>> validate_query_type("pattern")
        True
        >>> validate_query_type("invalid_type")
        False
    """
    logging.debug(f"DEBUG: validate_query_type() - Function called with query_type: {query_type}")

    if not query_type or not isinstance(query_type, str):
        logging.debug(f"DEBUG: validate_query_type() - Invalid query_type input: {query_type} (type: {type(query_type)})")
        return False

    valid_query_types = VALIDATION_CONFIG["valid_query_types"]
    result = query_type.lower() in valid_query_types
    logging.debug(f"DEBUG: validate_query_type() - Validation result: {result} (valid options: {valid_query_types})")
    return result


def validate_all_components(ticker: Optional[str], timeframe: Optional[str],
                          pattern: Optional[str], query_type: Optional[str],
                          check_yfinance: bool = True, exchange: str = "nse") -> dict:
    """
    Validate all extracted components and return validation results.

    Args:
        ticker: The extracted ticker symbol
        timeframe: The extracted timeframe
        pattern: The extracted pattern
        query_type: The detected query type
        check_yfinance: Whether to validate ticker against yfinance

    Returns:
        dict: Validation results for each component

    Example:
        >>> validate_all_components("INFY", "1d", "doji", "pattern")
        {'ticker': True, 'timeframe': True, 'pattern': True, 'query_type': True}
    """
    logging.debug(
        "DEBUG: validate_all_components() - Function called with "
        f"ticker: {ticker}, timeframe: {timeframe}, pattern: {pattern}, "
        f"query_type: {query_type}, check_yfinance: {check_yfinance}, "
        f"exchange: {exchange}"
    )

    result = {
        "ticker": validate_ticker(ticker, check_yfinance, exchange) if ticker else False,
        "timeframe": validate_timeframe(timeframe) if timeframe else False,
        "pattern": validate_pattern(pattern) if pattern else False,
        "query_type": validate_query_type(query_type) if query_type else False
    }

    logging.debug(f"DEBUG: validate_all_components() - Validation results: {result}")
    return result
