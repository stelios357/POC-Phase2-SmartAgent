"""
Unified data service that wraps stock data access.

Provides a single interface for history, batch, latest candle, current price,
and info retrieval while handling smart fallback selection between yfinance and
Kaggle. Callers should use this service instead of reaching into individual
fetchers.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from config.settings import DATA_SOURCE_CONFIG
from src import yfinance_wrapper as yf
from src.kaggle_data_fetcher import kaggle_data_fetcher


class DataService:
    """High-level facade for stock data access."""

    def __init__(self) -> None:
        self._last_kaggle_data_date = self._compute_kaggle_freshness()
        self.logger = logging.getLogger(__name__)

    def _compute_kaggle_freshness(self) -> Optional[datetime]:
        """Inspect a few sample stocks to gauge Kaggle data recency."""
        sample_stocks = ["TCS", "RELIANCE", "INFY"]
        latest_dates: List[datetime] = []

        for ticker in sample_stocks:
            try:
                info = kaggle_data_fetcher.get_data_info(ticker)
                date_range = info.get("date_range") or {}
                end_date = date_range.get("end")
                if end_date:
                    latest_dates.append(pd.to_datetime(end_date))  # type: ignore[name-defined]
            except Exception:
                continue

        return max(latest_dates) if latest_dates else None

    def _resolve_data_source(self, requested: Optional[str], timeframe: str, context: str) -> str:
        """Normalize data source selection using smart fallback rules."""
        source = requested or DATA_SOURCE_CONFIG["default_source"]

        if source != "smart_fallback":
            return source

        config = DATA_SOURCE_CONFIG["smart_fallback"]

        if context == "backtest":
            return config["backtest_priority"]

        if context == "pattern":
            return config["pattern_analysis_priority"]

        if context == "latest":
            return config["current_data_priority"]

        # Historical path: prefer Kaggle when fresh enough
        if self._last_kaggle_data_date:
            days_old = (datetime.now() - self._last_kaggle_data_date).days
            if days_old <= config["max_kaggle_age_days"]:
                return config["historical_data_priority"]

        return config["current_data_priority"]

    # Public API -----------------------------------------------------
    def get_history(
        self,
        ticker: str,
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: Optional[str] = None,
        exchange: str = "nse",
        data_source: Optional[str] = None,
        context: str = "historical",
    ):
        source = self._resolve_data_source(data_source, timeframe or "1d", context)
        return yf.get_stock_history(
            ticker=ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            exchange=exchange,
            data_source=source,
            context=context,
        )

    def get_batch_history(self, tickers: List[str], period: Optional[str] = None, exchange: str = "nse"):
        return yf.get_multiple_stocks(tickers, period=period, exchange=exchange)

    def get_latest_candle(
        self,
        ticker: str,
        timeframe: str = "1d",
        exchange: str = "nse",
        data_source: Optional[str] = None,
        context: str = "latest",
    ):
        source = self._resolve_data_source(data_source, timeframe, context)
        return yf.get_latest_candle(ticker, exchange=exchange, data_source=source, context=context)

    def get_current_price(self, ticker: str, exchange: str = "nse"):
        return yf.get_current_price(ticker, exchange=exchange)

    def get_stock_info(self, ticker: str, exchange: str = "nse"):
        return yf.get_stock_info(ticker, exchange=exchange)

    def resolve_ticker(self, ticker: str, exchange: str = "nse") -> str:
        return yf.resolve_ticker(ticker, exchange)

    def convert_temporal_context_to_dates(self, temporal_context: str):
        return yf.convert_temporal_context_to_dates(temporal_context)


data_service = DataService()

