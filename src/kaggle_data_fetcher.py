# Kaggle Data Fetcher for Historical Stock Data
# Provides historical minute-level data from Kaggle dataset as alternative to yfinance

import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
import re

# Define exceptions locally to avoid circular imports
class DataFetcherError(Exception):
    """Base exception for data fetching operations"""
    pass

class InvalidTickerError(DataFetcherError):
    """Raised when ticker symbol is invalid"""
    pass

class NoDataError(DataFetcherError):
    """Raised when no data is available"""
    pass

class KaggleDataFetcher:
    """Data fetcher for historical Kaggle dataset with same interface as DataFetcher"""

    def __init__(self, data_directory: str = "kaggle_data/archive"):
        self.data_directory = data_directory
        self.timeframe_mapping = {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "1d": "1D", "1w": "1W"
        }

        # Cache for loaded dataframes to improve performance
        self._data_cache = {}

        # List of available stocks
        self._available_stocks = self._load_available_stocks()

        logging.debug(f"KaggleDataFetcher initialized with {len(self._available_stocks)} stocks")

    def _load_available_stocks(self) -> List[str]:
        """Load list of available stock symbols from Kaggle data"""
        if not os.path.exists(self.data_directory):
            logging.warning(f"Kaggle data directory not found: {self.data_directory}")
            return []

        stocks = []
        for file in os.listdir(self.data_directory):
            if file.endswith('_minute.csv'):
                stock_symbol = file.replace('_minute.csv', '')
                stocks.append(stock_symbol)

        return sorted(stocks)

    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol to match Kaggle data format"""
        # Remove .NS suffix if present (Kaggle data doesn't use it)
        ticker = ticker.upper().replace('.NS', '').replace('.BO', '')
        return ticker

    def _load_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load stock data from Kaggle CSV file"""
        normalized_ticker = self._normalize_ticker(ticker)

        # Check cache first
        if normalized_ticker in self._data_cache:
            return self._data_cache[normalized_ticker]

        file_path = os.path.join(self.data_directory, f"{normalized_ticker}_minute.csv")

        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Rename columns to match yfinance format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Cache the dataframe
            self._data_cache[normalized_ticker] = df
            logging.debug(f"Loaded and cached data for {normalized_ticker}: {len(df)} rows")
            return df

        except Exception as e:
            logging.error(f"Error loading data for {ticker}: {e}")
            return None

    def validate_ticker(self, ticker: str) -> str:
        """Validate and normalize ticker symbol"""
        if not ticker or not isinstance(ticker, str):
            raise InvalidTickerError("Ticker symbol must be a non-empty string")

        normalized_ticker = self._normalize_ticker(ticker)

        if normalized_ticker not in self._available_stocks:
            raise InvalidTickerError(f"Ticker {ticker} not found in Kaggle dataset")

        return normalized_ticker

    def get_latest_candle(self, ticker: str, timeframe: str = "1d") -> Dict:
        """
        Fetch latest candle data for pattern analysis from historical data

        Args:
            ticker: Stock symbol (e.g., "TCS", "RELIANCE")
            timeframe: Timeframe ("1m", "5m", "15m", "30m", "1h", "1d", "1w")

        Returns:
            Dict with OHLCV data for latest candle
        """
        logging.debug(f"Fetching latest candle for {ticker} in {timeframe} timeframe")

        ticker = self.validate_ticker(ticker)

        if timeframe not in self.timeframe_mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Load the data
        df = self._load_stock_data(ticker)
        if df is None or df.empty:
            raise NoDataError(f"No data available for {ticker}")

        # Resample to requested timeframe if needed
        if timeframe != "1m":
            resample_rule = self.timeframe_mapping[timeframe]
            df_resampled = df.resample(resample_rule).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            df_resampled = df

        if df_resampled.empty:
            raise NoDataError(f"No data available for {ticker} in timeframe {timeframe}")

        # Get the latest candle
        latest_row = df_resampled.iloc[-1]

        # Calculate change from previous candle if available
        change = change_pct = prev_close = None
        if len(df_resampled) >= 2:
            prev_close = float(df_resampled.iloc[-2]['Close'])
            change = float(latest_row['Close']) - prev_close
            change_pct = (change / prev_close * 100) if prev_close != 0 else 0

        candle_data = {
            "timestamp": latest_row.name.strftime('%Y-%m-%d'),
            "open": round(float(latest_row['Open']), 2),
            "high": round(float(latest_row['High']), 2),
            "low": round(float(latest_row['Low']), 2),
            "close": round(float(latest_row['Close']), 2),
            "volume": int(latest_row['Volume']),
            "change": round(change, 2) if change else None,
            "change_percent": round(change_pct, 2) if change_pct else None,
            "previous_close": round(prev_close, 2) if prev_close else None
        }

        # Add more precise timestamp for intraday data
        if timeframe in ["1m", "5m", "15m", "30m", "1h"]:
            candle_data["timestamp"] = latest_row.name.strftime('%Y-%m-%d %H:%M:%S')

        return candle_data

    def get_historical_data(self, ticker: str, timeframe: str = "1d",
                           start_date: Optional[str] = None, end_date: Optional[str] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical data for analysis

        Args:
            ticker: Stock symbol
            timeframe: Timeframe for resampling
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of candles to return

        Returns:
            DataFrame with OHLCV data
        """
        ticker = self.validate_ticker(ticker)

        df = self._load_stock_data(ticker)
        if df is None or df.empty:
            raise NoDataError(f"No data available for {ticker}")

        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]

        # Resample to requested timeframe if needed
        if timeframe != "1m":
            resample_rule = self.timeframe_mapping[timeframe]
            df = df.resample(resample_rule).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        # Apply limit if specified
        if limit and len(df) > limit:
            df = df.tail(limit)

        return df

    def get_available_stocks(self) -> List[str]:
        """Get list of available stock symbols"""
        return self._available_stocks.copy()

    def get_data_info(self, ticker: str) -> Dict:
        """Get information about available data for a stock"""
        ticker = self.validate_ticker(ticker)

        df = self._load_stock_data(ticker)
        if df is None:
            return {}

        return {
            "symbol": ticker,
            "total_records": len(df),
            "date_range": {
                "start": df.index.min().strftime('%Y-%m-%d'),
                "end": df.index.max().strftime('%Y-%m-%d')
            },
            "avg_volume": df['Volume'].mean(),
            "price_range": {
                "min": df['Low'].min(),
                "max": df['High'].max()
            }
        }


# Global instance
kaggle_data_fetcher = KaggleDataFetcher()
