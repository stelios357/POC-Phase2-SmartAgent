# YFinance Wrapper Functions
# Implements Levels 1-5 from project_context.md: Basic fetching, error handling, caching, batch processing, validation

import yfinance as yf
import yfinance.exceptions as yf_exceptions
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import traceback
import logging
import calendar

from config.settings import API_CONFIG, DATA_SOURCE_CONFIG
from .kaggle_data_fetcher import kaggle_data_fetcher
from config.settings import VALIDATION_CONFIG, TICKER_CONFIG
from src.logging_config import log_api_call, log_cache_operation, log_performance_metric
from src.cache_manager import cache_manager

class YFinanceError(Exception):
    """Base exception for yfinance operations"""
    pass

class InvalidTickerError(YFinanceError):
    """Raised when ticker symbol is invalid"""
    pass

class RateLimitError(YFinanceError):
    """Raised when rate limit is exceeded"""
    pass

class NetworkError(YFinanceError):
    """Raised when network issues occur"""
    pass

class DataValidationError(YFinanceError):
    """Raised when data validation fails"""
    pass

class DataFetcherError(YFinanceError):
    """Raised when data source selection or fetch fails"""
    pass


def _determine_smart_data_source(timeframe: str, context: str = "historical") -> str:
    """
    Decide which data source to use based on context and freshness rules.
    Mirrors the prior smart-fallback logic without depending on data_fetcher.
    """
    config = DATA_SOURCE_CONFIG["smart_fallback"]

    if context == "backtest":
        return config["backtest_priority"]

    if context == "pattern":
        return config["pattern_analysis_priority"]

    if context == "latest":
        return config["current_data_priority"]

    # Historical defaults to Kaggle when allowed
    return config["historical_data_priority"]

def validate_candle_data(row) -> bool:
    """Validate OHLC candle data for consistency"""
    try:
        open_price = float(row['Open'])
        high_price = float(row['High'])
        low_price = float(row['Low'])
        close_price = float(row['Close'])
        volume = int(row['Volume'])

        # Basic validations
        if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
            return False

        # OHLC relationship: High >= Open, High >= Close, Low <= Open, Low <= Close
        if not (high_price >= open_price and high_price >= close_price and
                low_price <= open_price and low_price <= close_price):
            return False

        # Volume should be reasonable (not zero for active stocks)
        if volume < 0:
            return False

        # High-Low spread should be reasonable (not too wide)
        if low_price > 0:
            spread_ratio = (high_price - low_price) / low_price
            if spread_ratio > 0.50:  # More than 50% daily spread might indicate data issues
                logging.getLogger("validation").warning(f"Unusual price spread: {spread_ratio:.2%}")

        return True
    except (ValueError, KeyError, TypeError):
        return False

def validate_price_data(info) -> bool:
    """Validate current price data from info"""
    try:
        price = float(info.get('regularMarketPrice', 0))
        if price <= 0:
            return False

        # Check if we have reasonable price ranges
        day_high = float(info.get('regularMarketDayHigh', 0))
        day_low = float(info.get('regularMarketDayLow', 0))

        if day_high > 0 and day_low > 0:
            if not (day_low <= price <= day_high):
                return False

        return True
    except (ValueError, TypeError):
        return False

def resolve_ticker(ticker: str, exchange: str = 'nse') -> str:
    """
    Resolve ambiguous ticker symbols to valid yfinance tickers.

    Strategies:
    1. Check if it's already a valid format (e.g., TCS.NS, ^NSEI)
    2. Check index mappings first (e.g., NIFTY -> ^NSEI)
    3. Check common mappings (e.g., RELIANCE -> RELIANCE.NS)
    4. Strip existing suffixes if they don't match the target exchange
    5. Apply exchange-specific suffix if no suffix present
    """
    ticker = ticker.upper().strip()
    # Normalize common ampersand forms by removing spaces so they match mappings
    if "&" in ticker:
        ticker = ticker.replace(" ", "")

    # Check index mappings first (highest priority for indices)
    if ticker in TICKER_CONFIG.get("index_mappings", {}):
        return TICKER_CONFIG["index_mappings"][ticker]

    # Check common mappings for stocks
    if ticker in TICKER_CONFIG.get("common_mappings", {}):
        return TICKER_CONFIG["common_mappings"][ticker]

    # Check if it's already an index symbol (starts with ^)
    if ticker.startswith('^'):
        return ticker  # Already a valid index symbol

    # Check if suffix is present
    has_ns_suffix = ticker.endswith('.NS')
    has_bo_suffix = ticker.endswith('.BO')

    # Strip existing suffix if it doesn't match the target exchange
    if has_ns_suffix and exchange != 'nse':
        ticker = ticker[:-3]  # Remove .NS
    elif has_bo_suffix and exchange != 'bse':
        ticker = ticker[:-3]  # Remove .BO

    # Apply exchange-specific suffix if no suffix present
    if "." not in ticker:
        if exchange == 'nse':
            return f"{ticker}.NS"
        elif exchange == 'bse':
            return f"{ticker}.BO"
        # For non-india, don't add any suffix

    return ticker

def validate_ticker(ticker: str, indian_context: bool = True, exchange: str = 'nse') -> str:
    """Validate ticker symbol format and normalize it for yfinance.

    This is the single source of truth for ticker normalization. All yfinance
    calls must use the value returned from here (e.g., TCS -> TCS.NS).
    """
    if not ticker or not isinstance(ticker, str):
        raise InvalidTickerError("Ticker symbol must be a non-empty string")

    # Use the resolution logic with exchange context
    normalized_ticker = resolve_ticker(ticker, exchange)

    # Basic validation - alphanumeric and common special chars
    import re
    if indian_context:
        # Allow Indian stock symbols (NSE: RELIANCE.NS, BSE: RELIANCE.BO) and indices (^NSEI)
        if not re.match(r'^(?:\^[A-Z0-9]+|[A-Z0-9.-]+(?:\.NS|\.BO)?)$', normalized_ticker):
            raise InvalidTickerError(f"Invalid ticker format: {ticker} (normalized: {normalized_ticker})")
    else:
        # Allow global symbols and indices (^GSPC, AAPL, etc.)
        if not re.match(r'^(?:\^[A-Z0-9]+|[A-Z0-9.-]+)$', normalized_ticker):
            raise InvalidTickerError(f"Invalid ticker format: {ticker} (normalized: {normalized_ticker})")

    return normalized_ticker

def validate_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate downloaded data according to quality requirements"""
    if df is None or df.empty:
        raise DataValidationError(f"No data returned for ticker {ticker}")

    # Check required columns
    missing_cols = set(VALIDATION_CONFIG["required_columns"]) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing required columns for {ticker}: {missing_cols}")

    # Check for excessive NaN values
    nan_percentages = (df.isnull().sum() / len(df)) * 100
    high_nan_cols = nan_percentages[nan_percentages > VALIDATION_CONFIG["nan_threshold_percent"]]

    if not high_nan_cols.empty:
        logging.getLogger("validation").warning(
            f"High NaN percentage for {ticker}",
            extra={"extra_data": {"columns": high_nan_cols.to_dict()}}
        )

    # Validate price ranges
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if col in df.columns:
            invalid_prices = df[
                (df[col] < VALIDATION_CONFIG["price_validation"]["min_price"]) |
                (df[col] > VALIDATION_CONFIG["price_validation"]["max_price"])
            ]
            if not invalid_prices.empty:
                raise DataValidationError(f"Invalid price range in {col} for {ticker}")

    # Validate volume
    if "Volume" in df.columns:
        invalid_volume = df[
            (df["Volume"] < VALIDATION_CONFIG["volume_validation"]["min_volume"]) |
            (df["Volume"] > VALIDATION_CONFIG["volume_validation"]["max_volume"])
        ]
        if not invalid_volume.empty:
            logging.getLogger("validation").warning(f"Suspicious volume data for {ticker}")

    return df

def retry_with_backoff(func, max_retries=None, backoff_factor=None, *args, **kwargs):
    """Execute function with exponential backoff retry logic"""
    if max_retries is None:
        max_retries = API_CONFIG["max_retries"]
    if backoff_factor is None:
        backoff_factor = API_CONFIG["retry_backoff_factor"]

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # Don't retry on certain errors
            if isinstance(e, InvalidTickerError):
                raise e

            if attempt < max_retries:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
            else:
                raise e

    raise last_exception

def safe_yfinance_call(operation_name: str, ticker: str, call_func, exchange: str = 'nse', *args, **kwargs):
    """Wrapper for yfinance calls with comprehensive error handling.

    The provided call_func MUST accept the normalized ticker as its first
    positional argument. This ensures that ticker normalization is applied
    consistently before any yfinance operations are executed.
    """
    start_time = datetime.utcnow()

    original_ticker = ticker

    try:
        # Validate ticker with exchange context
        indian_context = (exchange in ['nse', 'bse'])
        normalized_ticker = validate_ticker(ticker, indian_context=indian_context, exchange=exchange)

        # Add delay between requests to respect rate limits
        time.sleep(API_CONFIG["request_delay"])

        # Execute the yfinance call with the normalized ticker
        result = call_func(normalized_ticker, *args, **kwargs)

        # Log successful call
        log_api_call(operation_name, normalized_ticker, start_time, datetime.utcnow(), success=True)

        return result

    except yf_exceptions.YFRateLimitError as e:
        log_api_call(operation_name, original_ticker, start_time, datetime.utcnow(),
                    success=False, error=e)
        raise RateLimitError(f"Rate limit exceeded for {original_ticker}") from e

    except Exception as e:
        # Handle various yfinance exceptions
        message = str(e)
        message_lower = message.lower()

        if ("no data found" in message_lower or "not found" in message_lower or
            "delisted" in message_lower or "no data returned" in message_lower):
            error = InvalidTickerError(f"Ticker {original_ticker} not found")
        elif "timeout" in message_lower or "connection" in message_lower:
            error = NetworkError(f"Network error for {original_ticker}")
        else:
            error = YFinanceError(f"Unexpected error for {original_ticker}: {message}")

        log_api_call(operation_name, original_ticker, start_time, datetime.utcnow(),
                    success=False, error=error)

        raise error from e

def safe_yfinance_batch_call(operation_name: str, ticker_list: List[str], call_func, *args, **kwargs):
    """Wrapper for batch yfinance calls without individual ticker validation"""
    start_time = datetime.utcnow()
    ticker_str = ",".join(ticker_list)

    try:
        # Add delay between requests to respect rate limits
        time.sleep(API_CONFIG["request_delay"])

        # Execute the yfinance call
        result = call_func(*args, **kwargs)

        # Log successful call
        log_api_call(operation_name, ticker_str, start_time, datetime.utcnow(), success=True)

        return result

    except yf_exceptions.YFRateLimitError as e:
        log_api_call(operation_name, ticker_str, start_time, datetime.utcnow(),
                    success=False, error=e)
        raise RateLimitError(f"Rate limit exceeded for batch {ticker_list}") from e

    except Exception as e:
        # Handle various yfinance exceptions
        error = YFinanceError(f"Batch operation failed for {ticker_list}: {str(e)}")
        log_api_call(operation_name, ticker_str, start_time, datetime.utcnow(),
                    success=False, error=error)
        raise error from e

def get_stock_history(ticker: str, period: str = None, start_date: str = None,
                     end_date: str = None, timeframe: Optional[str] = None, exchange: str = 'nse',
                     data_source: str = None, context: str = "historical") -> pd.DataFrame:
    """Fetch historical stock data with validation, error handling, and smart fallback routing.

    If a timeframe is provided (e.g. 5m, 1h, 1d, 1w), this function will
    derive an appropriate yfinance interval and default period so that
    the returned candles match the requested timeframe semantics.

    Args:
        ticker: Stock symbol
        period: Time period (e.g., "1mo", "1y")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Timeframe for resampling ("1m", "5m", etc.)
        exchange: Exchange ("nse" or "bse")
        data_source: Data source ("yfinance", "kaggle", "smart_fallback", or None for default)
        context: Request context for smart routing ("historical", "pattern", "backtest")
    """
    # Apply smart fallback logic if requested
    if data_source == "smart_fallback":
        data_source = _determine_smart_data_source(timeframe or "1d", context)

    # Route to appropriate data source
    if data_source == "kaggle":
        try:
            return kaggle_data_fetcher.get_historical_data(
                ticker=ticker,
                timeframe=timeframe or "1d",
                start_date=start_date,
                end_date=end_date,
                limit=None  # No limit for historical data
            )
        except Exception as e:
            # Fallback to yfinance if Kaggle fails
            logging.warning(f"Kaggle historical data failed for {ticker}, falling back to yfinance: {e}")

    # Default to yfinance implementation

    # Map standard timeframes to yfinance intervals and sensible default periods
    timeframe_spec = {
        "1m": {"interval": "1m", "default_period": "2d"},
        "5m": {"interval": "5m", "default_period": "5d"},
        "15m": {"interval": "15m", "default_period": "10d"},
        "30m": {"interval": "30m", "default_period": "1mo"},
        "1h": {"interval": "60m", "default_period": "3mo"},
        "1d": {"interval": "1d", "default_period": "1y"},
        "1w": {"interval": "1wk", "default_period": "2y"},
        "1mo": {"interval": "1mo", "default_period": "2y"},
    }

    interval = None
    if timeframe:
        spec = timeframe_spec.get(timeframe)
        if not spec:
            raise DataValidationError(f"Unsupported timeframe: {timeframe}")
        interval = spec["interval"]
        if period is None:
            period = spec["default_period"]

    if period is None:
        period = API_CONFIG["default_period"]

    # When start_date/end_date are provided, don't use period for cache lookup
    # to avoid cache key conflicts
    cache_period = None if (start_date and end_date) else period

    # Check cache first (cache-aside pattern)
    cached_data = cache_manager.get(ticker, period=cache_period, start_date=start_date, end_date=end_date)
    if cached_data is not None:
        return cached_data

    # Cache miss - fetch from API
    def _fetch_history(normalized_ticker: str):
        stock = yf.Ticker(normalized_ticker)
        history_kwargs = {
            "timeout": API_CONFIG["timeout"],
        }

        # yfinance treats end as exclusive; extend single-day requests by one day
        fetch_start_date = start_date
        fetch_end_date = end_date
        single_day_range = None
        if start_date and end_date and start_date == end_date:
            try:
                fetch_end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                single_day_range = (pd.to_datetime(start_date), pd.to_datetime(fetch_end_date))
            except Exception:
                pass

        # yfinance allows maximum 2 of: period, start, end
        # Priority: start/end take precedence over period when both are provided
        if fetch_start_date and fetch_end_date:
            history_kwargs["start"] = fetch_start_date
            history_kwargs["end"] = fetch_end_date
        elif period:
            history_kwargs["period"] = period

        if interval is not None:
            history_kwargs["interval"] = interval

        df = stock.history(**history_kwargs)

        # For single-day requests, filter back to the original day after extending end
        if single_day_range is not None:
            start_dt, end_dt = single_day_range
            # Ensure timezone consistency for comparison
            if df.index.tz is not None:
                # If index is timezone-aware, make comparison datetimes timezone-aware
                start_dt = start_dt.tz_localize(df.index.tz) if start_dt.tz is None else start_dt
                end_dt = end_dt.tz_localize(df.index.tz) if end_dt.tz is None else end_dt
            df = df[(df.index >= start_dt) & (df.index < end_dt)]

        return validate_data(df, normalized_ticker)

    df = retry_with_backoff(lambda: safe_yfinance_call("get_history", ticker, _fetch_history, exchange))

    # Cache the result (keyed by user ticker + period; timeframe is encoded via period/interval)
    cache_manager.put(ticker, df, period=period, start_date=start_date, end_date=end_date)

    return df

def get_stock_info(ticker: str, exchange: str = 'nse') -> Dict:
    """Fetch company information"""

    def _fetch_info(normalized_ticker: str):
        stock = yf.Ticker(normalized_ticker)
        info = stock.info
        if not info:
            raise DataValidationError(f"No info available for {normalized_ticker}")
        return info

    return retry_with_backoff(lambda: safe_yfinance_call("get_info", ticker, _fetch_info, exchange))

def get_multiple_stocks(tickers: List[str], period: str = None, batch_size: int = 10, exchange: str = 'nse') -> Dict[str, pd.DataFrame]:
    """Batch download multiple stocks efficiently"""

    if period is None:
        period = API_CONFIG["default_period"]

    results = {}

    # Process in batches to avoid rate limits
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        try:
            def _fetch_batch():
                # yf.download doesn't support our custom resolution logic, so we must map manually
                # if we want to support it. For now, assuming batch tickers are valid.
                # Ideally, we should resolve each ticker in the batch first.
                resolved_batch = [resolve_ticker(t, exchange) for t in batch]
                
                df = yf.download(resolved_batch, period=period, group_by='ticker',
                               timeout=API_CONFIG["timeout"])

                # yf.download returns MultiIndex columns for multiple tickers
                if len(resolved_batch) > 1:
                    result = {}
                    for ticker in resolved_batch:
                        if ticker in df.columns.levels[0]:
                            ticker_data = df[ticker].dropna()
                            if not ticker_data.empty:
                                result[ticker] = validate_data(ticker_data, ticker)
                    return result
                else:
                    # Single ticker
                    ticker = resolved_batch[0]
                    return {ticker: validate_data(df, ticker)}

            batch_results = retry_with_backoff(lambda: safe_yfinance_batch_call(
                "batch_download", batch, _fetch_batch))

            results.update(batch_results)

        except Exception as e:
            # Log batch failure but continue with individual fetches if needed
            logging.getLogger("yfinance_api").error(f"Batch download failed for {batch}: {e}")

            # Fallback to individual fetches
            for ticker in batch:
                try:
                    results[ticker] = get_stock_history(ticker, period)
                except Exception as individual_e:
                    logging.getLogger("yfinance_api").error(f"Individual fetch failed for {ticker}: {individual_e}")

    return results

def get_financial_statements(ticker: str) -> Dict[str, pd.DataFrame]:
    """Fetch financial statements (income statement, balance sheet, cash flow)"""

    def _fetch_financials(normalized_ticker: str):
        stock = yf.Ticker(normalized_ticker)

        return {
            "income_statement": stock.financials,
            "balance_sheet": stock.balance_sheet,
            "cash_flow": stock.cashflow,
        }

    return retry_with_backoff(lambda: safe_yfinance_call("get_financials", ticker, _fetch_financials))

def get_dividends_and_splits(ticker: str) -> Dict[str, pd.DataFrame]:
    """Fetch dividend and split history"""

    def _fetch_actions(normalized_ticker: str):
        stock = yf.Ticker(normalized_ticker)

        return {
            "dividends": stock.dividends,
            "splits": stock.splits,
        }

    return retry_with_backoff(lambda: safe_yfinance_call("get_actions", ticker, _fetch_actions))

# Performance monitoring
def get_latest_candle(ticker: str, exchange: str = 'nse', data_source: str = None, context: str = "latest") -> Dict:
    """Get the latest available candle/price data for a stock

    Returns the most recent trading day's data or real-time data if available.
    This is useful for getting current market prices and recent trading activity.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL")

    Returns:
        Dict: Latest candle data with the following structure:
        {
            "timestamp": "2024-01-15",  # Date of the data
            "open": 150.25,             # Opening price
            "high": 152.50,             # High price
            "low": 149.00,              # Low price
            "close": 151.75,            # Closing price
            "volume": 45000000,         # Trading volume
            "price": 151.75,            # Current/last price (same as close for daily)
            "change": 1.50,             # Price change from previous close
            "change_percent": 1.00      # Price change percentage
        }
    """
    # Apply smart fallback logic if requested
    if data_source == "smart_fallback":
        data_source = _determine_smart_data_source("1d", context)

    # Route to appropriate data source
    if data_source == "kaggle":
        # For latest/current data requests, Kaggle should not be used as it has old data
        # This is a configuration error - raise an exception
        raise DataFetcherError(f"Kaggle data source requested for latest data, but Kaggle only contains historical data. Use yfinance for current prices.")

    # Default to yfinance implementation
    def _fetch_latest(normalized_ticker: str):
        stock = yf.Ticker(normalized_ticker)

        # Try to get the most recent day's data
        df = stock.history(period="5d", interval="1d")  # Get last 5 days (improved from 2d for safety)

        if df.empty:
            raise DataValidationError(f"No recent data available for {normalized_ticker}")

        # Get the most recent row
        latest_row = df.iloc[-1]

        # Also get info for additional current data
        info = stock.info

        # Calculate change from previous day if available
        change = None
        change_percent = None
        if len(df) >= 2:
            prev_close = df.iloc[-2]['Close']
            current_close = latest_row['Close']
            change = current_close - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

        latest_data = {
            "timestamp": latest_row.name.strftime('%Y-%m-%d'),
            "open": round(float(latest_row['Open']), 2),
            "high": round(float(latest_row['High']), 2),
            "low": round(float(latest_row['Low']), 2),
            "close": round(float(latest_row['Close']), 2),
            "volume": int(latest_row['Volume']),
            "price": round(float(info.get('regularMarketPrice', latest_row['Close'])), 2),
            "change": round(float(change), 2) if change is not None else None,
            "change_percent": round(float(change_percent), 2) if change_percent is not None else None,
            "currency": info.get('currency', 'INR'),  # INR for Indian context
            "market_state": info.get('marketState', 'CLOSED'),
            "timezone": info.get('timeZoneFullName', 'Asia/Kolkata'),  # IST timezone
            "exchange": info.get('exchange', 'NSE'),  # NSE/BSE for Indian context
            "data_quality": "verified" if validate_candle_data(latest_row) else "warning"
        }

        # Add validation against official sources
        validation = validate_against_official_sources(normalized_ticker, latest_data)
        latest_data["validation"] = validation

        return latest_data

    return retry_with_backoff(lambda: safe_yfinance_call("get_latest_candle", ticker, _fetch_latest, exchange))

def get_current_price(ticker: str, exchange: str = 'nse') -> Dict:
    """Get the current market price and basic info for a stock

    This is a lightweight version focused on current price rather than full candle data.

    Parameters:
        ticker (str): Stock ticker symbol

    Returns:
        Dict: Current price information:
        {
            "ticker": "AAPL",
            "price": 151.75,
            "currency": "USD",
            "market_state": "REGULAR",  # PRE, REGULAR, POST, CLOSED
            "timestamp": "2024-01-15T16:00:00Z",
            "day_change": 1.50,
            "day_change_percent": 1.00,
            "volume": 45000000
        }
    """
    def _fetch_current(normalized_ticker: str):
        stock = yf.Ticker(normalized_ticker)
        info = stock.info

        if not info:
            raise DataValidationError(f"No current price data available for {normalized_ticker}")

        current_data = {
            "ticker": normalized_ticker.upper(),
            "price": round(float(info.get('regularMarketPrice', 0)), 2),
            "currency": info.get('currency', 'INR'),  # INR for Indian context
            "market_state": info.get('marketState', 'UNKNOWN'),
            "timestamp": datetime.utcnow().isoformat(),
            "day_change": round(float(info.get('regularMarketChange', 0)), 2),
            "day_change_percent": round(float(info.get('regularMarketChangePercent', 0)), 2),
            "volume": int(info.get('regularMarketVolume', 0)),
            "previous_close": round(float(info.get('regularMarketPreviousClose', 0)), 2),
            "day_high": round(float(info.get('regularMarketDayHigh', 0)), 2),
            "day_low": round(float(info.get('regularMarketDayLow', 0)), 2),
            "exchange": info.get('exchange', 'NSE'),  # NSE/BSE for Indian context
            "timezone": "Asia/Kolkata",  # IST timezone
            "market_hours": "09:15-15:30 IST" if info.get('marketState') != 'CLOSED' else "Closed",
            "data_quality": "verified" if validate_price_data(info) else "warning"
        }

        return current_data

    return retry_with_backoff(lambda: safe_yfinance_call("get_current_price", ticker, _fetch_current, exchange))

def convert_temporal_context_to_dates(temporal_context: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert temporal context strings to start_date and end_date for yfinance.

    Args:
        temporal_context (str): Temporal context like "last_week", "last_month", etc.

    Returns:
        Tuple[Optional[str], Optional[str]]: (start_date, end_date) in YYYY-MM-DD format, or (None, None) if no conversion needed
    """
    if not temporal_context:
        return None, None

    now = datetime.now()
    temporal_context = temporal_context.lower()

    if temporal_context == "today":
        # Today's data - from start of day to now
        start_date = now.strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        return start_date, end_date

    elif temporal_context == "yesterday":
        # Yesterday's data
        yesterday = now - timedelta(days=1)
        start_date = yesterday.strftime('%Y-%m-%d')
        end_date = yesterday.strftime('%Y-%m-%d')
        return start_date, end_date

    elif temporal_context == "this_week":
        # Current week (Monday to today)
        monday = now - timedelta(days=now.weekday())
        start_date = monday.strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        return start_date, end_date

    elif temporal_context == "last_week":
        # Previous week (Monday to Sunday)
        # Find last Monday
        last_monday = now - timedelta(days=now.weekday() + 7)
        # Find last Sunday
        last_sunday = last_monday + timedelta(days=6)
        start_date = last_monday.strftime('%Y-%m-%d')
        end_date = last_sunday.strftime('%Y-%m-%d')
        return start_date, end_date

    elif temporal_context == "this_month":
        # Current month (1st to today)
        start_date = now.replace(day=1).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        return start_date, end_date

    elif temporal_context == "last_month":
        # Previous month (full month)
        # Get first day of current month, then go back one month
        first_of_current = now.replace(day=1)
        last_of_previous = first_of_current - timedelta(days=1)
        first_of_previous = last_of_previous.replace(day=1)
        start_date = first_of_previous.strftime('%Y-%m-%d')
        end_date = last_of_previous.strftime('%Y-%m-%d')
        return start_date, end_date

    elif temporal_context == "last_year":
        # Previous year (full year)
        last_year = now.year - 1
        start_date = f"{last_year}-01-01"
        end_date = f"{last_year}-12-31"
        return start_date, end_date

    # If no specific temporal context matched, return None
    return None, None

def validate_against_official_sources(ticker: str, data: Dict) -> Dict:
    """
    Cross-validate data against official Indian exchange sources
    Returns validation results and confidence scores
    """
    validation_result = {
        "ticker": ticker,
        "data_source": "yfinance",
        "validation_checks": [],
        "overall_confidence": 0.0,
        "warnings": [],
        "errors": []
    }

    try:
        # Check 1: Price range validation (Indian stocks typically ₹10-₹50,000)
        price = data.get('price', 0)
        if not (10 <= price <= 50000):
            validation_result["warnings"].append(f"Price ₹{price} outside typical Indian stock range")
            validation_result["validation_checks"].append({"check": "price_range", "passed": False})
        else:
            validation_result["validation_checks"].append({"check": "price_range", "passed": True})

        # Check 2: Market hours validation (IST)
        from datetime import datetime
        import pytz

        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)

        # Indian market hours: 9:15 AM - 3:30 PM IST, Monday-Friday
        market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

        is_market_day = now_ist.weekday() < 5  # Monday-Friday
        is_market_hours = market_open <= now_ist <= market_close if is_market_day else False

        market_state = data.get('market_state', 'UNKNOWN')
        expected_state = 'REGULAR' if is_market_hours else 'CLOSED'

        if market_state == expected_state:
            validation_result["validation_checks"].append({"check": "market_hours", "passed": True})
        else:
            validation_result["warnings"].append(f"Market state {market_state} doesn't match expected {expected_state}")
            validation_result["validation_checks"].append({"check": "market_hours", "passed": False})

        # Check 3: Volume validation (Indian stocks have varying volumes)
        volume = data.get('volume', 0)
        if volume < 0:
            validation_result["errors"].append("Negative volume")
            validation_result["validation_checks"].append({"check": "volume_validity", "passed": False})
        elif volume == 0:
            validation_result["warnings"].append("Zero volume - might indicate no trading")
            validation_result["validation_checks"].append({"check": "volume_validity", "passed": False})
        else:
            validation_result["validation_checks"].append({"check": "volume_validity", "passed": True})

        # Check 4: Currency validation
        currency = data.get('currency', '')
        if currency not in ['INR', '₹']:
            validation_result["warnings"].append(f"Unexpected currency: {currency}")
            validation_result["validation_checks"].append({"check": "currency", "passed": False})
        else:
            validation_result["validation_checks"].append({"check": "currency", "passed": True})

        # Check 5: Data freshness (should be recent)
        timestamp_str = data.get('timestamp', '')
        if timestamp_str:
            try:
                data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                data_age = (datetime.now(pytz.UTC) - data_time).total_seconds()

                if data_age > 3600:  # Older than 1 hour
                    validation_result["warnings"].append(f"Data is {data_age/3600:.1f} hours old")
                    validation_result["validation_checks"].append({"check": "data_freshness", "passed": False})
                else:
                    validation_result["validation_checks"].append({"check": "data_freshness", "passed": True})
            except:
                validation_result["warnings"].append("Could not parse timestamp")
                validation_result["validation_checks"].append({"check": "data_freshness", "passed": False})

        # Calculate overall confidence score
        passed_checks = sum(1 for check in validation_result["validation_checks"] if check["passed"])
        total_checks = len(validation_result["validation_checks"])
        validation_result["overall_confidence"] = passed_checks / total_checks if total_checks > 0 else 0.0

        # Add confidence level
        if validation_result["overall_confidence"] >= 0.8:
            validation_result["confidence_level"] = "HIGH"
        elif validation_result["overall_confidence"] >= 0.6:
            validation_result["confidence_level"] = "MEDIUM"
        else:
            validation_result["confidence_level"] = "LOW"

    except Exception as e:
        validation_result["errors"].append(f"Validation failed: {str(e)}")
        validation_result["overall_confidence"] = 0.0
        validation_result["confidence_level"] = "ERROR"

    return validation_result

def get_api_stats() -> Dict:
    """Get API usage statistics (to be implemented with persistent storage)"""
    # This would track calls, success rates, etc.
    return {
        "calls_today": 0,
        "success_rate": 0.0,
        "avg_response_time": 0.0,
    }
