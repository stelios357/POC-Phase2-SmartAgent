# Cache Manager for Stock Data
# Implements Level 3: Performance Optimization - Caching

import pandas as pd
import json
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import logging

from config.settings import CACHE_CONFIG
from src.logging_config import log_cache_operation

class CacheManager:
    """File-based cache manager for stock data"""

    def __init__(self, cache_dir: str = None):
        logging.debug(f"DEBUG: CacheManager.__init__() - Called with cache_dir: {cache_dir}")
        self.cache_dir = Path(cache_dir or CACHE_CONFIG["cache_directory"])
        logging.debug(f"DEBUG: CacheManager.__init__() - Using cache directory: {self.cache_dir}")
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.lock = threading.Lock()

        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)

        # Load existing metadata
        self.metadata = self._load_metadata()

        # Clean up expired entries on initialization
        self._cleanup_expired()

    def _load_metadata(self) -> Dict:
        logging.debug(f"DEBUG: CacheManager._load_metadata() - Called")
        # [Function implementation]
        """Load cache metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                loggers = __import__('src.logging_config', fromlist=['loggers']).loggers
                loggers["cache"].warning(f"Failed to load cache metadata: {e}")
                return {}
        return {}

    def _save_metadata(self):
        logging.debug(f"DEBUG: CacheManager._save_metadata() - Called")
        # [Function implementation]
        """Save cache metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except IOError as e:
            loggers = __import__('src.logging_config', fromlist=['loggers']).loggers
            loggers["cache"].error(f"Failed to save cache metadata: {e}")

    def _generate_cache_key(self, ticker: str, period: str = None,
                           start_date: str = None, end_date: str = None) -> str:
        """Generate unique cache key for request"""
        key_components = [ticker.upper()]

        if period:
            key_components.append(f"period:{period}")
        if start_date:
            key_components.append(f"start:{start_date}")
        if end_date:
            key_components.append(f"end:{end_date}")

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        logging.debug(f"DEBUG: CacheManager._get_cache_path() - Called with key: {cache_key}")
        # [Function implementation]
        """Get file path for cached data"""
        return self.cache_dir / f"{cache_key}.csv"

    def _is_expired(self, timestamp: str) -> bool:
        logging.debug(f"DEBUG: CacheManager._is_expired() - Called with timestamp: {timestamp}")
        # [Function implementation]
        """Check if cache entry is expired"""
        try:
            cache_time = datetime.fromisoformat(timestamp)
            expiry_time = timedelta(hours=CACHE_CONFIG["cache_expiration_hours"])
            return datetime.utcnow() - cache_time > expiry_time
        except (ValueError, TypeError):
            return True  # Consider invalid timestamps as expired

    def _cleanup_expired(self):
        logging.debug(f"DEBUG: CacheManager._cleanup_expired() - Called")
        # [Function implementation]
        """Remove expired cache entries"""
        expired_keys = []

        for key, meta in self.metadata.items():
            if self._is_expired(meta.get("timestamp", "")):
                expired_keys.append(key)

        for key in expired_keys:
            cache_path = self._get_cache_path(key)
            try:
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata[key]
                log_cache_operation("cleanup", key.split("|")[0], cache_hit=False)
            except OSError as e:
                loggers = __import__('src.logging_config', fromlist=['loggers']).loggers
                loggers["cache"].error(f"Failed to remove expired cache file {key}: {e}")

        if expired_keys:
            self._save_metadata()

    def get(self, ticker: str, period: str = None, start_date: str = None,
            end_date: str = None) -> Optional[pd.DataFrame]:
        logging.debug(f"DEBUG: CacheManager.get() - Called with ticker: {ticker}")
        # [Function implementation]
        """Retrieve data from cache if available and fresh"""

        with self.lock:
            cache_key = self._generate_cache_key(ticker, period, start_date, end_date)

            if cache_key not in self.metadata:
                log_cache_operation("miss", ticker, cache_hit=False)
                return None

            meta = self.metadata[cache_key]

            # Check if expired
            if self._is_expired(meta["timestamp"]):
                log_cache_operation("expired", ticker, cache_hit=False)
                self._remove_entry(cache_key)
                return None

            # Load data from file
            cache_path = self._get_cache_path(cache_key)
            try:
                if CACHE_CONFIG["cache_format"] == "csv":
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                else:
                    df = pd.read_json(cache_path, orient='index')

                # Calculate cache age
                cache_time = datetime.fromisoformat(meta["timestamp"])
                age_hours = (datetime.utcnow() - cache_time).total_seconds() / 3600

                log_cache_operation("hit", ticker, cache_hit=True, cache_age_hours=age_hours)

                return df

            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
                loggers = __import__('src.logging_config', fromlist=['loggers']).loggers
                loggers["cache"].warning(f"Failed to read cache file {cache_key}: {e}")
                self._remove_entry(cache_key)
                return None

    def put(self, ticker: str, data: pd.DataFrame, period: str = None,
            start_date: str = None, end_date: str = None):
        logging.debug(f"DEBUG: CacheManager.put() - Called with ticker: {ticker}")
        # [Function implementation]
        """Store data in cache"""

        with self.lock:
            cache_key = self._generate_cache_key(ticker, period, start_date, end_date)
            cache_path = self._get_cache_path(cache_key)

            try:
                # Save data to file
                if CACHE_CONFIG["cache_format"] == "csv":
                    data.to_csv(cache_path)
                else:
                    data.to_json(cache_path, orient='index', date_format='iso')

                # Update metadata
                self.metadata[cache_key] = {
                    "ticker": ticker,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data_points": len(data),
                    "columns": list(data.columns),
                    "period": period,
                    "start_date": start_date,
                    "end_date": end_date,
                    "hash": hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()[:8]
                }

                self._save_metadata()
                log_cache_operation("store", ticker, cache_hit=False)

            except (OSError, ValueError) as e:
                loggers = __import__('src.logging_config', fromlist=['loggers']).loggers
                loggers["cache"].error(f"Failed to write cache file {cache_key}: {e}")

    def _remove_entry(self, cache_key: str):
        logging.debug(f"DEBUG: CacheManager._remove_entry() - Called with key: {cache_key}")
        # [Function implementation]
        """Remove a cache entry"""
        if cache_key in self.metadata:
            del self.metadata[cache_key]

        cache_path = self._get_cache_path(cache_key)
        try:
            if cache_path.exists():
                cache_path.unlink()
        except OSError:
            pass

        self._save_metadata()

    def clear(self, ticker: str = None):
        logging.debug(f"DEBUG: CacheManager.clear() - Called with ticker: {ticker}")
        # [Function implementation]
        """Clear cache entries (all or for specific ticker)"""
        with self.lock:
            if ticker:
                # Remove entries for specific ticker
                keys_to_remove = [
                    key for key, meta in self.metadata.items()
                    if meta.get("ticker", "").upper() == ticker.upper()
                ]
                for key in keys_to_remove:
                    self._remove_entry(key)
            else:
                # Clear all cache
                for cache_key in list(self.metadata.keys()):
                    self._remove_entry(cache_key)

                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.csv"):
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass

    def get_stats(self) -> Dict[str, Any]:
        logging.debug(f"DEBUG: CacheManager.get_stats() - Called")
        # [Function implementation]
        """Get cache statistics"""
        with self.lock:
            total_entries = len(self.metadata)
            total_size = 0

            # Calculate total cache size
            for cache_key in self.metadata.keys():
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    total_size += cache_path.stat().st_size

            expired_count = sum(
                1 for meta in self.metadata.values()
                if self._is_expired(meta.get("timestamp", ""))
            )

            return {
                "total_entries": total_entries,
                "expired_entries": expired_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_hit_ratio": 0.0,  # Would need persistent tracking
                "oldest_entry": min(
                    (datetime.fromisoformat(meta["timestamp"]) for meta in self.metadata.values()
                     if "timestamp" in meta),
                    default=None
                ),
                "newest_entry": max(
                    (datetime.fromisoformat(meta["timestamp"]) for meta in self.metadata.values()
                     if "timestamp" in meta),
                    default=None
                ),
            }

    def warmup(self, tickers: list, period: str = "1y"):
        logging.debug(f"DEBUG: CacheManager.warmup() - Called with {len(tickers)} tickers")
        # [Function implementation]
        """Pre-populate cache for frequently used stocks"""
        from src.yfinance_wrapper import get_multiple_stocks

        loggers = __import__('src.logging_config', fromlist=['loggers']).loggers
        loggers["cache"].info(f"Starting cache warmup for {len(tickers)} tickers")

        try:
            results = get_multiple_stocks(tickers, period)
            for ticker, data in results.items():
                self.put(ticker, data, period=period)
        except Exception as e:
            loggers["cache"].error(f"Cache warmup failed: {e}")

# Global cache instance
cache_manager = CacheManager()
