# Candlestick Pattern Analyzer
# Main orchestrator for candlestick pattern detection workflow

import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .data_fetcher import DataFetcherError, InvalidTickerError, NoDataError
from .response_formatter import response_formatter
from .services.data_service import data_service
from .services.pattern_service import pattern_service

class CandlestickAnalyzer:
    """Main analyzer class for candlestick pattern detection"""

    def __init__(self):
        logging.debug("DEBUG: CandlestickAnalyzer.__init__() - Initializing analyzer")
        self.response_formatter = response_formatter
        logging.debug("DEBUG: CandlestickAnalyzer.__init__() - Analyzer initialized with all components")

    def analyze_patterns(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to analyze candlestick patterns with historical context
        """
        logging.debug(f"DEBUG: CandlestickAnalyzer.analyze_patterns() - Function called with query_data: {query_data}")

        try:
            # Extract and validate query parameters
            ticker = query_data.get('ticker', '').strip()
            timeframe = query_data.get('timeframe', '1d').strip()
            query_type = query_data.get('query_type', 'pattern_detection')
            pattern = query_data.get('pattern')
            specific_date = query_data.get('date')  # Check for specific date

            if not ticker:
                return self.response_formatter.format_error_response("MISSING_TICKER", "Ticker symbol is required", query_data)

            # Step 1: Fetch History based on whether we have a specific date or not
            logging.debug(f"DEBUG: CandlestickAnalyzer.analyze_patterns() - Fetching history for context (specific_date: {specific_date})")

            # Use smart fallback for pattern analysis
            from config.settings import DATA_SOURCE_CONFIG
            data_source = DATA_SOURCE_CONFIG["default_source"]

            if specific_date:
                # For specific date queries, fetch data around that date
                # Get 5 days before and after to provide context for pattern analysis
                from datetime import datetime, timedelta
                try:
                    date_obj = datetime.strptime(specific_date, '%Y-%m-%d')
                    start_date = (date_obj - timedelta(days=5)).strftime('%Y-%m-%d')
                    end_date = (date_obj + timedelta(days=5)).strftime('%Y-%m-%d')

                    history_df = data_service.get_history(
                        ticker,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=timeframe,
                        data_source=data_source,
                        context="pattern",
                    )

                    if history_df.empty:
                        raise NoDataError(f"No data available for {ticker} around {specific_date}")

                    # Find the specific date's data
                    target_date_data = history_df[history_df.index.strftime('%Y-%m-%d') == specific_date]

                    if target_date_data.empty:
                        # If exact date not found, try to find the closest trading day
                        date_obj = datetime.strptime(specific_date, '%Y-%m-%d')
                        # Check a few days before and after
                        for offset in range(-3, 4):  # -3 to +3 days
                            check_date = (date_obj + timedelta(days=offset)).strftime('%Y-%m-%d')
                            target_date_data = history_df[history_df.index.strftime('%Y-%m-%d') == check_date]
                            if not target_date_data.empty:
                                logging.debug(f"DEBUG: CandlestickAnalyzer.analyze_patterns() - Using closest trading day {check_date} for requested {specific_date}")
                                specific_date = check_date  # Update to actual trading date
                                break

                        if target_date_data.empty:
                            raise NoDataError(f"No trading data found for {ticker} on or around {specific_date}")

                    target_row = target_date_data.iloc[0]

                except ValueError as e:
                    raise DataFetcherError(f"Invalid date format: {specific_date}") from e
            else:
                # Default behavior: fetch recent history for pattern analysis
                # Adjust period based on timeframe to get enough candles but not too much
                fetch_period = "1mo" # Default for intraday/daily
                if timeframe == "1w":
                    fetch_period = "1y"
                elif timeframe == "1mo":
                    fetch_period = "2y"

                history_df = data_service.get_history(
                    ticker,
                    period=fetch_period,
                    timeframe=timeframe,
                    data_source=data_source,
                    context="pattern",
                )

                if history_df.empty:
                    raise NoDataError(f"No data available for {ticker}")

                target_row = history_df.iloc[-1]  # Use latest for non-specific date queries

            # Get target candle for display (either specific date or latest)
            target_close = float(target_row['Close'])
            prev_close = None
            change = None
            change_pct = None

            if len(history_df) >= 2:
                # Find previous trading day in the dataset
                target_idx = history_df.index.get_loc(target_row.name)
                if target_idx > 0:
                    prev_row = history_df.iloc[target_idx - 1]
                    prev_close = float(prev_row['Close'])
                    change = target_close - prev_close
                    change_pct = (change / prev_close * 100.0) if prev_close != 0 else 0.0

            # Determine currency based on ticker (Indian exchanges use ₹, others use $)
            from config.settings import TICKER_CONFIG
            from .yfinance_wrapper import resolve_ticker

            # Use the same resolution logic as yfinance_wrapper
            resolved_ticker = resolve_ticker(ticker, 'nse').upper()

            is_indian_stock = resolved_ticker.endswith('.NS') or resolved_ticker.endswith('.BO')
            is_indian_index = resolved_ticker.startswith('^') and (
                'NSEI' in resolved_ticker or
                'NSEBANK' in resolved_ticker or
                'BANKNIFTY' in resolved_ticker or
                'NIFTY' in resolved_ticker or
                'BSE' in resolved_ticker
            )

            currency = '₹' if (is_indian_stock or is_indian_index) else '$'

            target_candle = {
                "timestamp": target_row.name.strftime('%Y-%m-%d'),
                "open": float(target_row['Open']),
                "high": float(target_row['High']),
                "low": float(target_row['Low']),
                "close": target_close,
                "volume": int(target_row['Volume']),
                "change": round(change, 4) if change is not None else None,
                "change_percent": round(change_pct, 4) if change_pct is not None else None,
                "previous_close": round(prev_close, 4) if prev_close is not None else None,
                "currency": currency,
            }

            # Step 2: Detect patterns using full history dataframe
            logging.debug("DEBUG: CandlestickAnalyzer.analyze_patterns() - Detecting patterns with history")
            detected_patterns = pattern_service.detect_patterns(history_df, pattern)
            
            # Step 3: Format response
            logging.debug("DEBUG: CandlestickAnalyzer.analyze_patterns() - Formatting response")
            response = self.response_formatter.format_pattern_analysis_response(
                query_data, target_candle, detected_patterns
            )

            return response

        except InvalidTickerError as e:
            logging.debug(f"DEBUG: CandlestickAnalyzer.analyze_patterns() - InvalidTickerError: {e}")
            return self.response_formatter.format_error_response("INVALID_TICKER", str(e), query_data)

        except NoDataError as e:
            logging.debug(f"DEBUG: CandlestickAnalyzer.analyze_patterns() - NoDataError: {e}")
            return self.response_formatter.format_error_response("NO_DATA", str(e), query_data)

        except Exception as e:
            logging.error(f"Unexpected error in analyze_patterns: {e}", exc_info=True)
            return self.response_formatter.format_error_response("INTERNAL_ERROR", f"An unexpected error occurred: {str(e)}", query_data)

    def get_supported_timeframes(self) -> Dict[str, str]:
        return {
            "intraday": ["1m", "5m", "15m", "30m", "1h"],
            "daily": ["1d"],
            "weekly": ["1w"]
        }

    def get_supported_patterns(self) -> list:
        return [
            "Doji (Common, Dragonfly, Gravestone, Long-Legged)",
            "Hammer / Hanging Man",
            "Shooting Star / Inverted Hammer",
            "Marubozu (Bullish/Bearish)"
        ]

def process_query(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    analyzer = CandlestickAnalyzer()
    return analyzer.analyze_patterns(parsed_query)

if __name__ == "__main__":
    # Simple CLI test
    if len(sys.argv) < 3:
        print("Usage: python candlestick_analyzer.py <ticker> <timeframe>")
        sys.exit(1)
        
    logging.basicConfig(level=logging.DEBUG)
    ticker = sys.argv[1]
    timeframe = sys.argv[2]
    
    query = {
        "ticker": ticker,
        "timeframe": timeframe,
        "query_type": "pattern_detection"
    }
    
    analyzer = CandlestickAnalyzer()
    result = analyzer.analyze_patterns(query)
    print(json.dumps(result, indent=2))
