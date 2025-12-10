# Data Fetcher for Candlestick Pattern Analysis
# Robust module with caching, error handling, and rate limiting for pattern detection

import yfinance as yf
import yfinance.exceptions as yf_exceptions
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import re
import time
import logging

from config.settings import API_CONFIG, DATA_SOURCE_CONFIG
from datetime import datetime, timedelta
import pandas as pd
from src.cache_manager import cache_manager
from src.logging_config import log_api_call
from src.kaggle_data_fetcher import kaggle_data_fetcher

class DataFetcherError(Exception):
    """Base exception for data fetching operations"""
    pass

class InvalidTickerError(DataFetcherError):
    """Raised when ticker symbol is invalid"""
    pass

class RateLimitError(DataFetcherError):
    """Raised when rate limit is exceeded"""
    pass

class NetworkError(DataFetcherError):
    """Raised when network issues occur"""
    pass

class NoDataError(DataFetcherError):
    """Raised when no data is available"""
    pass

def validate_candle_data(row) -> bool:
    """Validate OHLC candle data for consistency"""
    logging.debug(f"DEBUG: validate_candle_data() - Function called with row data")

    try:
        open_price = float(row['Open'])
        high_price = float(row['High'])
        low_price = float(row['Low'])
        close_price = float(row['Close'])
        volume = int(row['Volume'])
        logging.debug(f"DEBUG: validate_candle_data() - Extracted values: O={open_price}, H={high_price}, L={low_price}, C={close_price}, V={volume}")

        # Basic validations
        if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
            logging.debug("DEBUG: validate_candle_data() - Failed basic validation: prices must be positive")
            return False

        # OHLC relationship: High >= Open, High >= Close, Low <= Open, Low <= Close
        if not (high_price >= open_price and high_price >= close_price and
                low_price <= open_price and low_price <= close_price):
            logging.debug("DEBUG: validate_candle_data() - Failed OHLC relationship validation")
            return False

        # Volume should be reasonable (not zero for active stocks)
        if volume < 0:
            logging.debug("DEBUG: validate_candle_data() - Failed volume validation: volume cannot be negative")
            return False

        logging.debug("DEBUG: validate_candle_data() - All validations passed")
        return True
    except (ValueError, KeyError, TypeError) as e:
        logging.debug(f"DEBUG: validate_candle_data() - Exception during validation: {type(e).__name__}: {e}")
        return False

def retry_with_backoff(func, max_retries=None, backoff_factor=None, *args, **kwargs):
    """Execute function with exponential backoff retry logic"""
    logging.debug(f"DEBUG: retry_with_backoff() - Function called with max_retries={max_retries}, backoff_factor={backoff_factor}")

    if max_retries is None:
        max_retries = API_CONFIG["max_retries"]
        logging.debug(f"DEBUG: retry_with_backoff() - Using default max_retries: {max_retries}")
    if backoff_factor is None:
        backoff_factor = API_CONFIG["retry_backoff_factor"]
        logging.debug(f"DEBUG: retry_with_backoff() - Using default backoff_factor: {backoff_factor}")

    last_exception = None

    for attempt in range(max_retries + 1):
        logging.debug(f"DEBUG: retry_with_backoff() - Attempt {attempt + 1}/{max_retries + 1}")
        try:
            result = func(*args, **kwargs)
            logging.debug("DEBUG: retry_with_backoff() - Function executed successfully")
            return result
        except Exception as e:
            last_exception = e
            logging.debug(f"DEBUG: retry_with_backoff() - Exception on attempt {attempt + 1}: {type(e).__name__}: {e}")

            # Don't retry on certain errors
            if isinstance(e, InvalidTickerError):
                logging.debug("DEBUG: retry_with_backoff() - Not retrying InvalidTickerError")
                raise e

            if attempt < max_retries:
                wait_time = backoff_factor ** attempt
                logging.debug(f"DEBUG: retry_with_backoff() - Waiting {wait_time:.2f} seconds before retry")
                time.sleep(wait_time)
            else:
                logging.debug("DEBUG: retry_with_backoff() - Max retries exceeded, raising last exception")
                raise e

    logging.debug("DEBUG: retry_with_backoff() - Unexpected end of function, raising last exception")
    raise last_exception

def safe_yfinance_call(operation_name: str, ticker: str, call_func, *args, **kwargs):
    """Wrapper for yfinance calls with comprehensive error handling"""
    logging.debug(f"DEBUG: safe_yfinance_call() - Function called with operation: {operation_name}, ticker: {ticker}")
    start_time = datetime.utcnow()
    logging.debug(f"DEBUG: safe_yfinance_call() - Start time: {start_time}")

    try:
        # Add delay between requests to respect rate limits
        delay = API_CONFIG["request_delay"]
        logging.debug(f"DEBUG: safe_yfinance_call() - Applying request delay: {delay} seconds")
        time.sleep(delay)

        # Execute the yfinance call
        logging.debug("DEBUG: safe_yfinance_call() - Executing yfinance call")
        result = call_func(*args, **kwargs)
        logging.debug("DEBUG: safe_yfinance_call() - Yfinance call completed successfully")

        # Log successful call
        log_api_call(operation_name, ticker, start_time, datetime.utcnow(), success=True)
        logging.debug("DEBUG: safe_yfinance_call() - API call logged as successful")

        return result

    except yf_exceptions.YFRateLimitError as e:
        logging.debug(f"DEBUG: safe_yfinance_call() - YFRateLimitError caught: {e}")
        log_api_call(operation_name, ticker, start_time, datetime.utcnow(),
                    success=False, error=e)
        logging.debug("DEBUG: safe_yfinance_call() - Raising RateLimitError")
        raise RateLimitError(f"Rate limit exceeded for {ticker}") from e

    except Exception as e:
        logging.debug(f"DEBUG: safe_yfinance_call() - Exception caught: {type(e).__name__}: {e}")
        # Handle various yfinance exceptions
        error_type = type(e).__name__
        error_message = str(e).lower()

        if ("No data found" in error_message or "not found" in error_message or
            "delisted" in error_message or "No data returned" in error_message):
            error = InvalidTickerError(f"Ticker {ticker} not found")
            logging.debug("DEBUG: safe_yfinance_call() - Categorized as InvalidTickerError")
        elif "timeout" in error_message or "connection" in error_message:
            error = NetworkError(f"Network error for {ticker}")
            logging.debug("DEBUG: safe_yfinance_call() - Categorized as NetworkError")
        else:
            error = DataFetcherError(f"Unexpected error for {ticker}: {str(e)}")
            logging.debug("DEBUG: safe_yfinance_call() - Categorized as DataFetcherError")

        log_api_call(operation_name, ticker, start_time, datetime.utcnow(),
                    success=False, error=error)
        logging.debug("DEBUG: safe_yfinance_call() - API call logged as failed")

        logging.debug(f"DEBUG: safe_yfinance_call() - Raising error: {type(error).__name__}")
        raise error from e

class DataFetcher:
    """Robust data fetcher for candlestick pattern analysis with caching, error handling, and smart fallback routing"""

    def __init__(self):
        # Supported timeframes mapping to yfinance format
        self.timeframe_mapping = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "1d": "1d", "1w": "1wk"
        }

        # Initialize smart fallback tracking
        self._last_kaggle_data_date = None
        self._update_kaggle_data_freshness()

    def _update_kaggle_data_freshness(self):
        """Update the freshness of Kaggle data by checking latest available dates"""
        try:
            # Check a few major stocks to determine data freshness
            sample_stocks = ['TCS', 'RELIANCE', 'INFY']
            latest_dates = []

            for stock in sample_stocks:
                try:
                    info = kaggle_data_fetcher.get_data_info(stock)
                    latest_dates.append(pd.to_datetime(info['date_range']['end']))
                except:
                    continue

            if latest_dates:
                self._last_kaggle_data_date = max(latest_dates)
                logging.debug(f"Kaggle data freshness updated: latest date is {self._last_kaggle_data_date}")

        except Exception as e:
            logging.warning(f"Could not determine Kaggle data freshness: {e}")
            self._last_kaggle_data_date = None

    def _is_request_for_current_data(self, timeframe: str, context: str = "general") -> bool:
        """
        Determine if the request is for current/latest data that needs real-time source

        Args:
            timeframe: The timeframe requested
            context: Context of the request ("latest", "pattern", "backtest", "historical")

        Returns:
            True if this is a current data request, False for historical
        """
        # Latest candle requests are always for current data
        if context == "latest":
            return True

        # Very short timeframes in recent context are current
        if timeframe in ["1m", "5m", "15m"] and context in ["pattern", "latest"]:
            return True

        # Backtesting is always historical
        if context == "backtest":
            return False

        # Check if Kaggle data is too old for this timeframe
        if self._last_kaggle_data_date:
            days_old = (datetime.now() - self._last_kaggle_data_date).days
            max_age = DATA_SOURCE_CONFIG["smart_fallback"]["max_kaggle_age_days"]

            # For daily and higher timeframes, Kaggle data might still be usable
            if timeframe in ["1d", "1w"] and days_old <= max_age:
                return False

        return False

    def _determine_smart_data_source(self, ticker: str, timeframe: str, context: str = "general") -> str:
        """
        Determine the best data source using smart fallback logic

        Args:
            ticker: Stock symbol
            timeframe: Timeframe requested
            context: Request context ("latest", "pattern", "backtest", "historical")

        Returns:
            Best data source ("yfinance" or "kaggle")
        """
        config = DATA_SOURCE_CONFIG["smart_fallback"]

        # Backtesting always uses Kaggle (has the extensive historical data needed)
        if context == "backtest":
            return config["backtest_priority"]

        # Pattern analysis - use yfinance for current pattern detection
        if context == "pattern":
            return config["pattern_analysis_priority"]

        # Latest/current data requests prioritize yfinance
        if context == "latest" or self._is_request_for_current_data(timeframe, context):
            return config["current_data_priority"]

        # Historical data requests - check if kaggle data is fresh enough
        if self._last_kaggle_data_date:
            days_old = (datetime.now() - self._last_kaggle_data_date).days
            max_age = config["max_kaggle_age_days"]
            if days_old <= max_age:
                return config["historical_data_priority"]
            else:
                logging.warning(f"Kaggle data is {days_old} days old (max age: {max_age}), falling back to yfinance")
                return config["current_data_priority"]
        else:
            # No kaggle data available, use yfinance
            return config["current_data_priority"]

    def _try_data_sources_with_fallback(self, primary_source: str, ticker: str, timeframe: str,
                                       fetch_function, *args, **kwargs):
        """
        Try primary data source, fallback to secondary if needed

        Args:
            primary_source: Preferred data source ("yfinance" or "kaggle")
            ticker: Stock symbol
            timeframe: Timeframe
            fetch_function: The function to call for fetching
            *args, **kwargs: Arguments for the fetch function

        Returns:
            Fetched data or raises exception if both sources fail
        """
        sources_to_try = [primary_source]

        # Add fallback source if different from primary
        if primary_source == "yfinance":
            sources_to_try.append("kaggle")
        else:
            sources_to_try.append("yfinance")

        last_exception = None

        for source in sources_to_try:
            try:
                logging.debug(f"Trying data source: {source} for {ticker} {timeframe}")
                return fetch_function(data_source=source, *args, **kwargs)
            except Exception as e:
                logging.warning(f"Data source {source} failed for {ticker} {timeframe}: {e}")
                last_exception = e
                continue

        # If we get here, both sources failed
        logging.error(f"All data sources failed for {ticker} {timeframe}")
        raise last_exception

    def validate_ticker(self, ticker: str) -> str:
        """Validate and normalize ticker symbol for Indian market"""
        logging.debug(f"DEBUG: DataFetcher.validate_ticker() - Function called with ticker: '{ticker}'")

        if not ticker or not isinstance(ticker, str):
            logging.debug("DEBUG: DataFetcher.validate_ticker() - Invalid input, raising InvalidTickerError")
            raise InvalidTickerError("Ticker symbol must be a non-empty string")

        ticker = ticker.upper().strip()
        logging.debug(f"DEBUG: DataFetcher.validate_ticker() - After upper/strip: '{ticker}'")

        # Basic validation - alphanumeric and common special chars
        if not re.match(r'^[A-Z0-9.-]+(?:\.NS|\.BO)?$', ticker):
            logging.debug(f"DEBUG: DataFetcher.validate_ticker() - Regex validation failed for: '{ticker}'")
            # If no exchange suffix, add .NS for Indian context
            if '.' not in ticker:
                ticker = f"{ticker}.NS"
                logging.debug(f"DEBUG: DataFetcher.validate_ticker() - Added .NS suffix: '{ticker}'")
            else:
                logging.debug(f"DEBUG: DataFetcher.validate_ticker() - Invalid format with dot, raising error: '{ticker}'")
                raise InvalidTickerError(f"Invalid ticker format: {ticker}")

        logging.debug(f"DEBUG: DataFetcher.validate_ticker() - Validation successful, returning: '{ticker}'")
        return ticker

    def get_latest_candle(self, ticker: str, timeframe: str = "1d", data_source: str = None, context: str = "latest") -> Dict:
        """
        Fetch latest candle data for pattern analysis with caching, error handling, and smart fallback

        Args:
            ticker: Stock symbol (e.g., "INFY", "RELIANCE")
            timeframe: Timeframe ("1m", "5m", "15m", "30m", "1h", "1d", "1w")
            data_source: Data source to use ("yfinance", "kaggle", "smart_fallback", or None for default)
            context: Request context ("latest", "pattern", "backtest", "historical") for smart routing

        Returns:
            Dict with OHLCV data for latest candle
        """
        logging.debug(f"DEBUG: DataFetcher.get_latest_candle() - Function called with ticker: '{ticker}', timeframe: '{timeframe}', data_source: '{data_source}', context: '{context}'")

        # Determine data source to use
        if data_source is None:
            data_source = DATA_SOURCE_CONFIG["default_source"]

        # Apply smart fallback logic
        if data_source == "smart_fallback":
            data_source = self._determine_smart_data_source(ticker, timeframe, context)
            logging.debug(f"DEBUG: DataFetcher.get_latest_candle() - Smart fallback determined data source: '{data_source}'")

        logging.debug(f"DEBUG: DataFetcher.get_latest_candle() - Final data source: '{data_source}'")

        # Route to appropriate data fetcher
        if data_source == "kaggle":
            try:
                return kaggle_data_fetcher.get_latest_candle(ticker, timeframe)
            except Exception as e:
                if DATA_SOURCE_CONFIG["fallback_to_yfinance"]:
                    logging.warning(f"DEBUG: DataFetcher.get_latest_candle() - Kaggle fetch failed, falling back to yfinance: {e}")
                    data_source = "yfinance"
                else:
                    raise e

        # Default to yfinance or fallback
        ticker = self.validate_ticker(ticker)
        logging.debug(f"DEBUG: DataFetcher.get_latest_candle() - Ticker validated: '{ticker}'")

        if timeframe not in self.timeframe_mapping:
            logging.debug(f"DEBUG: DataFetcher.get_latest_candle() - Unsupported timeframe: '{timeframe}'")
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Simple in-memory cache for candle data (not using DataFrame cache manager)
        cache_key = f"latest_candle_{ticker}_{timeframe}"
        logging.debug(f"DEBUG: DataFetcher.get_latest_candle() - Checking cache with key: '{cache_key}'")

        if hasattr(self, '_candle_cache') and cache_key in self._candle_cache:
            cached_data, cache_time = self._candle_cache[cache_key]
            cache_age_seconds = (datetime.utcnow() - cache_time).seconds
            logging.debug(f"DEBUG: DataFetcher.get_latest_candle() - Cache hit, age: {cache_age_seconds} seconds")
            # Cache for 5 minutes
            if cache_age_seconds < 300:
                logging.debug("DEBUG: DataFetcher.get_latest_candle() - Using cached data")
                return cached_data
            else:
                logging.debug("DEBUG: DataFetcher.get_latest_candle() - Cache expired, fetching fresh data")

        # Cache miss - fetch from API
        logging.debug("DEBUG: DataFetcher.get_latest_candle() - Cache miss, proceeding with API fetch")

        def _fetch_candle():
            logging.debug("DEBUG: DataFetcher._fetch_candle() - Inner function called")
            stock = yf.Ticker(ticker)

            # Special handling for weekly timeframe
            if timeframe == "1w":
                logging.debug("DEBUG: DataFetcher._fetch_candle() - Weekly timeframe detected, using daily data aggregation")
                # Fetch 2 years of daily data to get sufficient weekly candles
                df_daily = stock.history(period="2y", interval="1d")
                logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Daily data fetched for weekly aggregation, shape: {df_daily.shape}")

                if df_daily.empty:
                    logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Empty daily dataframe, raising NoDataError")
                    raise NoDataError(f"No daily data available for {ticker} to create weekly candles")

                # Resample daily data to weekly (Monday to Friday)
                # Use 'W' for week ending on Sunday, but adjust to Friday for Indian market
                df = df_daily.resample('W-FRI').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                # Filter out incomplete weeks (very recent partial weeks)
                if len(df) > 0:
                    # Remove the last candle if it's from today (incomplete week)
                    today = pd.Timestamp.today(tz='UTC').normalize()  # Use UTC timezone
                    last_candle_date = df.index[-1]
                    if last_candle_date.tz is None:
                        last_candle_date = last_candle_date.tz_localize('UTC')
                    last_candle_date = last_candle_date.normalize()

                    # If the last candle is from this week and today is not Friday, remove it
                    if last_candle_date >= today - pd.Timedelta(days=today.weekday()):
                        # Check if we're in the middle of the current week
                        days_since_friday = (today.weekday() - 4) % 7  # 4 = Friday
                        if days_since_friday > 0:  # Not Friday yet
                            df = df[:-1]  # Remove incomplete week
                            logging.debug("DEBUG: DataFetcher._fetch_candle() - Removed incomplete current week")

                logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Weekly data aggregated, shape: {df.shape}")
            else:
                yf_interval = self.timeframe_mapping[timeframe]
                logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Using yfinance interval: '{yf_interval}'")

                # Get appropriate period for timeframe
                period = "2d" if timeframe in ["1m", "5m", "15m", "30m"] else "5d"
                logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Using period: '{period}' for timeframe '{timeframe}'")

                df = stock.history(period=period, interval=yf_interval)
                logging.debug(f"DEBUG: DataFetcher._fetch_candle() - History fetched, dataframe shape: {df.shape}")

            if df.empty:
                logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Empty dataframe, raising NoDataError")
                raise NoDataError(f"No data available for {ticker} in timeframe {timeframe}")

            # Validate the data
            latest_row = df.iloc[-1]
            if not validate_candle_data(latest_row):
                logging.debug("DEBUG: DataFetcher._fetch_candle() - Data validation failed, raising NoDataError")
                raise NoDataError(f"Invalid candle data for {ticker}")

            latest = df.iloc[-1]
            logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Latest candle extracted: {latest.name}")

            # Calculate change from previous candle if available
            change = change_pct = prev_close = None
            if len(df) >= 2:
                prev_close = float(df.iloc[-2]['Close'])
                change = float(latest['Close']) - prev_close
                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Change calculated: {change:.2f} ({change_pct:.2f}%)")
            else:
                logging.debug("DEBUG: DataFetcher._fetch_candle() - Not enough data for change calculation")

            candle_data = {
                "timestamp": latest.name.strftime('%Y-%m-%d'),
                "open": round(float(latest['Open']), 2),
                "high": round(float(latest['High']), 2),
                "low": round(float(latest['Low']), 2),
                "close": round(float(latest['Close']), 2),
                "volume": int(latest['Volume']),
                "change": round(change, 2) if change else None,
                "change_percent": round(change_pct, 2) if change_pct else None,
                "previous_close": round(prev_close, 2) if prev_close else None
            }
            logging.debug(f"DEBUG: DataFetcher._fetch_candle() - Candle data prepared: {candle_data}")

            # Add more precise timestamp for intraday data
            if timeframe in ["1m", "5m", "15m", "30m", "1h"]:
                candle_data["timestamp"] = latest.name.strftime('%Y-%m-%d %H:%M:%S')
                logging.debug("DEBUG: DataFetcher._fetch_candle() - Added precise timestamp for intraday data")

            return candle_data

        # Fetch with retry logic and error handling
        logging.debug("DEBUG: DataFetcher.get_latest_candle() - Starting fetch with retry logic")
        candle_data = retry_with_backoff(lambda: safe_yfinance_call("get_latest_candle", ticker, _fetch_candle))
        logging.debug("DEBUG: DataFetcher.get_latest_candle() - Fetch completed successfully")

        # Cache the result (simple in-memory cache)
        logging.debug(f"DEBUG: DataFetcher.get_latest_candle() - Caching result with key: '{cache_key}'")
        if not hasattr(self, '_candle_cache'):
            self._candle_cache = {}
        self._candle_cache[cache_key] = (candle_data, datetime.utcnow())

        logging.debug("DEBUG: DataFetcher.get_latest_candle() - Returning candle data")
        return candle_data


# Global instance
data_fetcher = DataFetcher()
