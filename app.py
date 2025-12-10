#!/usr/bin/env python3
# Main Application Entry Point for Candlestick Pattern Analysis
# Provides a simple interface to analyze candlestick patterns from parsed query input

import json
import sys
from typing import Dict, Any
import logging
from src.candlestick_analyzer import process_query

def analyze_from_parsed_input(parsed_query: Dict[str, Any]) -> str:
    """
    Main function to analyze candlestick patterns from parsed query input

    Args:
        parsed_query: Dictionary containing parsed query parameters
            Required keys:
            - ticker: Stock symbol (e.g., "INFY", "RELIANCE")
            - timeframe: Timeframe (e.g., "1d", "1h", "5m")

            Optional keys:
            - query_type: Type of query (default: "pattern_detection")

    Returns:
        JSON string response with pattern analysis results

    Example usage:
        parsed_input = {
            "ticker": "INFY",
            "timeframe": "1d",
            "query_type": "pattern_detection"
        }
        result = analyze_from_parsed_input(parsed_input)
        print(result)
    """
    logging.debug(f"DEBUG: analyze_from_parsed_input() - Called with parsed_query: {parsed_query}")

    try:
        # Process the query using the candlestick analyzer
        logging.debug("DEBUG: analyze_from_parsed_input() - Processing query with candlestick analyzer")
        response = process_query(parsed_query)
        logging.debug(f"DEBUG: analyze_from_parsed_input() - Query processing completed, success: {response.get('success', 'unknown')}")

        # Return as JSON string
        result = json.dumps(response, indent=2, ensure_ascii=False)
        logging.debug("DEBUG: analyze_from_parsed_input() - Returning successful JSON response")
        return result

    except Exception as e:
        logging.debug(f"DEBUG: analyze_from_parsed_input() - Exception caught: {type(e).__name__}: {e}")
        # Return error response
        error_response = {
            "success": False,
            "error": {
                "type": "PROCESSING_ERROR",
                "message": f"Failed to process query: {str(e)}"
            }
        }
        result = json.dumps(error_response, indent=2, ensure_ascii=False)
        logging.debug("DEBUG: analyze_from_parsed_input() - Returning error JSON response")
        return result


def main():
    """Command line interface for testing with sample parsed input"""
    logging.debug("DEBUG: main() - Function called")

    # Example parsed queries for testing
    sample_queries = [
        {
            "ticker": "INFY",
            "timeframe": "1d",
            "query_type": "pattern_detection"
        },
        {
            "ticker": "RELIANCE",
            "timeframe": "1h",
            "query_type": "pattern_detection"
        },
        {
            "ticker": "TCS",
            "timeframe": "1w",
            "query_type": "pattern_detection"
        }
    ]

    if len(sys.argv) > 1:
        logging.debug(f"DEBUG: main() - Using command line arguments: {sys.argv}")
        # Use command line arguments
        if len(sys.argv) < 3:
            logging.debug("DEBUG: main() - Insufficient command line arguments")
            print("Usage: python app.py <ticker> <timeframe>")
            print("Example: python app.py INFY 1d")
            sys.exit(1)

        ticker = sys.argv[1].upper()
        timeframe = sys.argv[2]
        logging.debug(f"DEBUG: main() - Parsed CLI args: ticker={ticker}, timeframe={timeframe}")

        parsed_query = {
            "ticker": ticker,
            "timeframe": timeframe,
            "query_type": "pattern_detection"
        }
    else:
        logging.debug("DEBUG: main() - No command line arguments, using sample query")
        # Use first sample query
        print("Using sample query (INFY, 1d)")
        print("To use custom ticker/timeframe: python app.py <ticker> <timeframe>")
        print()
        parsed_query = sample_queries[0]

    logging.debug(f"DEBUG: main() - Final parsed query: {parsed_query}")
    print(f"Analyzing patterns for {parsed_query['ticker']} on {parsed_query['timeframe']} timeframe...")
    print("-" * 60)

    # Process the query
    logging.debug("DEBUG: main() - Starting query processing")
    result = analyze_from_parsed_input(parsed_query)
    logging.debug("DEBUG: main() - Query processing completed")

    # Print the result
    print(result)
    logging.debug("DEBUG: main() - Result printed, main function completed")


if __name__ == "__main__":
    main()

