"""
Regex-based entity extraction for natural language queries.

This module provides functions to extract stock tickers, timeframes, and candlestick patterns
using regular expressions and pattern matching.
"""

import re
from typing import Dict, Optional, Union, Any
import logging
from datetime import datetime

# Configuration for regex patterns and mappings
REGEX_CONFIG = {
    "ticker_pattern": r"\b\$?[A-Z]{2,5}(?:\.[A-Z]{2})?\b",
    "timeframe_mapping": {
        "1m": ["1m", "1-minute", "1minute"],
        "5m": ["5m", "5-minute", "5minute"],
        "15m": ["15m", "15-minute", "15minute"],
        "30m": ["30m", "30-minute", "30minute"],
        "1h": ["1h", "hourly", "hour", "1-hour", "1hour"],
        "1d": ["1d", "daily", "day"],
        "1w": ["1w", "weekly", "week"],
        "1mo": ["1mo", "monthly", "month", "1-month", "1month"]
    },
    "temporal_contexts": {
        "today": ["today", "current day", "todays"],
        "yesterday": ["yesterday", "previous day"],
        "this_week": ["this week", "current week"],
        "this_month": ["this month", "current month"],
        "last_week": ["last week", "previous week"],
        "last_month": ["last month", "previous month"],
        "last_year": ["last year", "previous year"]
    },
    "patterns": ["doji", "hammer", "shooting_star", "marubozu"],
    "pattern_variations": {
        "doji": ["doji", "dojis"],
        "hammer": ["hammer", "hammers"],
        "shooting_star": ["shooting star", "shooting-star", "shootingstar", "shooting_stars"],
        "marubozu": ["marubozu", "marubozus"]
    },
    "month_names": {
        "jan": "01", "january": "01",
        "feb": "02", "february": "02",
        "mar": "03", "march": "03",
        "apr": "04", "april": "04",
        "may": "05",
        "jun": "06", "june": "06",
        "jul": "07", "july": "07",
        "aug": "08", "august": "08",
        "sep": "09", "september": "09",
        "oct": "10", "october": "10",
        "nov": "11", "november": "11",
        "dec": "12", "december": "12"
    },
    "ordinal_suffixes": ["st", "nd", "rd", "th"]
}


def extract_ticker(query: str) -> Optional[str]:
    """
    Extract ticker symbol from natural language query.

    Args:
        query (str): The input query string

    Returns:
        Optional[str]: Uppercase ticker symbol or None if not found

    Examples:
        >>> extract_ticker("Show me INFY on daily chart")
        'INFY'
        >>> extract_ticker("What about $AAPL price?")
        'AAPL'
    """
    logging.debug(f"DEBUG: extract_ticker() - Function called with query: '{query}'")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: extract_ticker() - Invalid input, returning None")
        return None

    query_upper = query.upper()
    logging.debug(f"DEBUG: extract_ticker() - Query converted to uppercase: '{query_upper}'")

    # First, try to find clear ticker patterns with word boundaries
    ticker_matches = re.findall(REGEX_CONFIG["ticker_pattern"], query)
    logging.debug(f"DEBUG: extract_ticker() - Regex ticker matches: {ticker_matches}")

    # Also look for potential tickers in all-caps words (like RELIANCE)
    all_caps_words = re.findall(r'\b[A-Z]{2,10}\b', query)
    logging.debug(f"DEBUG: extract_ticker() - All caps words found: {all_caps_words}")

    potential_tickers = list(set(ticker_matches + all_caps_words))
    logging.debug(f"DEBUG: extract_ticker() - Combined potential tickers: {potential_tickers}")

    # Filter out common words that aren't tickers
    exclude_words = {"OHLCV", "PRICE", "SHOW", "WHAT", "DAILY", "HOURLY", "WEEKLY", "MINUTE", "CHART", "CANDLE"}
    filtered_matches = [match for match in potential_tickers if match not in exclude_words and len(match) <= 10]
    logging.debug(f"DEBUG: extract_ticker() - Filtered matches (excluded: {exclude_words}): {filtered_matches}")

    if not filtered_matches:
        logging.debug("DEBUG: extract_ticker() - No filtered matches, returning None")
        return None

    # Clean matches (remove $ prefix for consistency)
    cleaned_matches = [match.lstrip('$') for match in filtered_matches]
    logging.debug(f"DEBUG: extract_ticker() - Cleaned matches (removed $ prefix): {cleaned_matches}")

    # If only one match, return it
    if len(cleaned_matches) == 1:
        logging.debug(f"DEBUG: extract_ticker() - Single match found, returning: {cleaned_matches[0]}")
        return cleaned_matches[0]

    logging.debug(f"DEBUG: extract_ticker() - Multiple matches ({len(cleaned_matches)}), applying heuristics")
    # Multiple matches - use heuristics to pick the best one
    # Priority 1: Look for tickers after preposition keywords
    preposition_keywords = ["OF", "FOR", "IN", "ON", "AT"]
    logging.debug(f"DEBUG: extract_ticker() - Priority 1: Checking preposition keywords: {preposition_keywords}")

    for keyword in preposition_keywords:
        keyword_pos = query_upper.find(f" {keyword} ")
        if keyword_pos != -1:
            logging.debug(f"DEBUG: extract_ticker() - Found preposition '{keyword}' at position {keyword_pos}")
            # Look for ticker after the keyword
            after_keyword = query_upper[keyword_pos + len(keyword) + 2:]  # +2 for spaces
            for ticker in cleaned_matches:
                ticker_pos = after_keyword.find(ticker)
                if ticker_pos != -1 and ticker_pos < 30:  # Within reasonable distance
                    logging.debug(f"DEBUG: extract_ticker() - Found ticker '{ticker}' after preposition '{keyword}', returning it")
                    return ticker

    # Priority 2: Look for tickers at word boundaries, preferring longer ones
    # Sort by length (longer = more likely to be ticker)
    sorted_matches = sorted(set(cleaned_matches), key=len, reverse=True)
    logging.debug(f"DEBUG: extract_ticker() - Priority 2: Sorted matches by length: {sorted_matches}")

    # Check if any match appears to be a standalone word (not part of another word)
    for match in sorted_matches:
        # Look for the match as a standalone word
        word_pattern = r'\b' + re.escape(match) + r'\b'
        if re.search(word_pattern, query_upper):
            logging.debug(f"DEBUG: extract_ticker() - Found standalone word match: '{match}', returning it")
            return match

    # Priority 3: Return the longest match if it's reasonable length
    logging.debug("DEBUG: extract_ticker() - Priority 3: Checking for reasonable length matches")
    for match in sorted_matches:
        if 2 <= len(match) <= 5:  # Standard ticker length
            logging.debug(f"DEBUG: extract_ticker() - Found reasonable length match: '{match}' (length {len(match)}), returning it")
            return match

    # Fallback: return the longest match
    result = sorted_matches[0] if sorted_matches else None
    logging.debug(f"DEBUG: extract_ticker() - Fallback: returning longest match: {result}")
    return result


def extract_tickers(query: str) -> list[str]:
    """
    Extract all ticker symbols from natural language query.
    Returns a list of tickers instead of picking just one.

    Args:
        query (str): The input query string

    Returns:
        list[str]: List of uppercase ticker symbols (may be empty)

    Examples:
        >>> extract_tickers("Show me INFY and TCS on daily chart")
        ['INFY', 'TCS']
        >>> extract_tickers("Check price for RELIANCE and TCS 5m")
        ['RELIANCE', 'TCS']
    """
    logging.debug(f"DEBUG: extract_tickers() - Function called with query: '{query}'")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: extract_tickers() - Invalid input, returning empty list")
        return []

    query_upper = query.upper()
    logging.debug(f"DEBUG: extract_tickers() - Query converted to uppercase: '{query_upper}'")

    # First, try to find clear ticker patterns with word boundaries
    ticker_matches = re.findall(REGEX_CONFIG["ticker_pattern"], query)
    logging.debug(f"DEBUG: extract_tickers() - Regex ticker matches: {ticker_matches}")

    # Also look for potential tickers in all-caps words (like RELIANCE)
    all_caps_words = re.findall(r'\b[A-Z]{2,10}\b', query)
    logging.debug(f"DEBUG: extract_tickers() - All caps words found: {all_caps_words}")

    potential_tickers = list(set(ticker_matches + all_caps_words))
    logging.debug(f"DEBUG: extract_tickers() - Combined potential tickers: {potential_tickers}")

    # Filter out common words that aren't tickers
    exclude_words = {"OHLCV", "PRICE", "SHOW", "WHAT", "DAILY", "HOURLY", "WEEKLY", "MINUTE", "CHART", "CANDLE", "AND", "OR", "FOR", "THE", "CHECK"}
    filtered_matches = [match for match in potential_tickers if match not in exclude_words and len(match) <= 10]
    logging.debug(f"DEBUG: extract_tickers() - Filtered matches (excluded: {exclude_words}): {filtered_matches}")

    if not filtered_matches:
        logging.debug("DEBUG: extract_tickers() - No filtered matches, returning empty list")
        return []

    # Clean matches (remove $ prefix for consistency)
    cleaned_matches = [match.lstrip('$') for match in filtered_matches]
    logging.debug(f"DEBUG: extract_tickers() - Cleaned matches (removed $ prefix): {cleaned_matches}")

    # Sort by position in query to maintain order
    # This helps preserve the order tickers appear in the query
    ticker_positions = {}
    for ticker in cleaned_matches:
        pos = query_upper.find(ticker)
        if pos != -1:
            ticker_positions[ticker] = pos

    sorted_tickers = sorted(cleaned_matches, key=lambda x: ticker_positions.get(x, 999))
    logging.debug(f"DEBUG: extract_tickers() - Sorted by position: {sorted_tickers}")

    # Validate each ticker has reasonable length
    valid_tickers = [ticker for ticker in sorted_tickers if 2 <= len(ticker) <= 10]
    logging.debug(f"DEBUG: extract_tickers() - Valid tickers (length 2-10): {valid_tickers}")

    return valid_tickers


def extract_timeframe(query: str) -> Optional[str]:
    """
    Extract timeframe from natural language query.

    Args:
        query (str): The input query string

    Returns:
        Optional[str]: Standard timeframe code or None if not found

    Examples:
        >>> extract_timeframe("Show me INFY on 5-minute chart")
        '5m'
        >>> extract_timeframe("Daily INFY candle")
        '1d'
    """
    logging.debug(f"DEBUG: extract_timeframe() - Function called with query: '{query}'")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: extract_timeframe() - Invalid input, returning None")
        return None

    query_lower = query.lower()
    logging.debug(f"DEBUG: extract_timeframe() - Query converted to lowercase: '{query_lower}'")

    # Check for exact timeframe matches first
    logging.debug("DEBUG: extract_timeframe() - Checking for exact timeframe variations")
    for std_timeframe, variations in REGEX_CONFIG["timeframe_mapping"].items():
        for variation in variations:
            if variation in query_lower:
                logging.debug(f"DEBUG: extract_timeframe() - Found variation '{variation}' for timeframe '{std_timeframe}', returning {std_timeframe}")
                return std_timeframe

    # Look for patterns like "5m", "1h", etc. with regex
    timeframe_pattern = r'\b(\d+[mhdw])\b'
    matches = re.findall(timeframe_pattern, query_lower)
    logging.debug(f"DEBUG: extract_timeframe() - Regex pattern matches: {matches}")

    for match in matches:
        # Map to standard format
        if match in REGEX_CONFIG["timeframe_mapping"]:
            logging.debug(f"DEBUG: extract_timeframe() - Found valid regex match: '{match}', returning it")
            return match

    logging.debug("DEBUG: extract_timeframe() - No timeframe found, returning None")
    return None


def extract_timeframes(query: str) -> list[str]:
    """
    Extract all timeframes from natural language query.
    Returns a list of timeframes instead of picking just one.

    Args:
        query (str): The input query string

    Returns:
        list[str]: List of standard timeframe codes (may be empty)

    Examples:
        >>> extract_timeframes("Check RELIANCE 5m and weekly")
        ['5m', '1w']
        >>> extract_timeframes("Show INFY daily and hourly")
        ['1d', '1h']
    """
    logging.debug(f"DEBUG: extract_timeframes() - Function called with query: '{query}'")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: extract_timeframes() - Invalid input, returning empty list")
        return []

    query_lower = query.lower()
    logging.debug(f"DEBUG: extract_timeframes() - Query converted to lowercase: '{query_lower}'")

    found_timeframes = []
    timeframe_positions = {}

    # Check for exact timeframe matches first
    logging.debug("DEBUG: extract_timeframes() - Checking for exact timeframe variations")
    for std_timeframe, variations in REGEX_CONFIG["timeframe_mapping"].items():
        for variation in variations:
            if variation in query_lower:
                pos = query_lower.find(variation)
                if pos != -1 and std_timeframe not in found_timeframes:
                    found_timeframes.append(std_timeframe)
                    timeframe_positions[std_timeframe] = pos
                    logging.debug(f"DEBUG: extract_timeframes() - Found variation '{variation}' for timeframe '{std_timeframe}' at position {pos}")

    # Look for patterns like "5m", "1h", etc. with regex
    timeframe_pattern = r'\b(\d+[mhdw])\b'
    matches = re.findall(timeframe_pattern, query_lower)
    logging.debug(f"DEBUG: extract_timeframes() - Regex pattern matches: {matches}")

    for match in matches:
        # Map to standard format
        if match in REGEX_CONFIG["timeframe_mapping"] and match not in found_timeframes:
            pos = query_lower.find(match)
            found_timeframes.append(match)
            timeframe_positions[match] = pos
            logging.debug(f"DEBUG: extract_timeframes() - Found regex match '{match}' at position {pos}")

    # Sort by position in query to maintain order
    sorted_timeframes = sorted(found_timeframes, key=lambda x: timeframe_positions.get(x, 999))
    logging.debug(f"DEBUG: extract_timeframes() - Sorted by position: {sorted_timeframes}")

    return sorted_timeframes


def extract_pattern(query: str) -> Optional[str]:
    """
    Extract candlestick pattern name from natural language query.

    Args:
        query (str): The input query string

    Returns:
        Optional[str]: Pattern name or None if not found

    Examples:
        >>> extract_pattern("Is there a doji on INFY?")
        'doji'
        >>> extract_pattern("Show me shooting star pattern")
        'shooting_star'
    """
    logging.debug(f"DEBUG: extract_pattern() - Function called with query: '{query}'")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: extract_pattern() - Invalid input, returning None")
        return None

    query_lower = query.lower()
    logging.debug(f"DEBUG: extract_pattern() - Query converted to lowercase: '{query_lower}'")

    # Check each pattern and its variations
    logging.debug("DEBUG: extract_pattern() - Checking pattern variations")
    for pattern, variations in REGEX_CONFIG["pattern_variations"].items():
        logging.debug(f"DEBUG: extract_pattern() - Checking pattern '{pattern}' with variations: {variations}")
        for variation in variations:
            if variation in query_lower:
                logging.debug(f"DEBUG: extract_pattern() - Found variation '{variation}' for pattern '{pattern}', returning {pattern}")
                return pattern

    logging.debug("DEBUG: extract_pattern() - No pattern found, returning None")
    return None


def extract_temporal_context(query: str) -> Optional[str]:
    """
    Extract temporal context from query (today, yesterday, this week, etc.).

    Args:
        query (str): The input query string

    Returns:
        Optional[str]: Temporal context or None if not found

    Examples:
        >>> extract_temporal_context("show today's TCS data")
        'today'
        >>> extract_temporal_context("TCS this week")
        'this_week'
    """
    logging.debug(f"DEBUG: extract_temporal_context() - Function called with query: '{query}'")

    query_lower = query.lower()
    for context, keywords in REGEX_CONFIG["temporal_contexts"].items():
        for keyword in keywords:
            if keyword in query_lower:
                logging.debug(f"DEBUG: extract_temporal_context() - Found temporal context: '{context}' from keyword '{keyword}'")
                return context

    logging.debug("DEBUG: extract_temporal_context() - No temporal context found")
    return None


def extract_date(query: str) -> Optional[str]:
    """
    Extract specific date from query (e.g., "21st nov 2025", "november 21 2025").

    Args:
        query (str): The input query string

    Returns:
        Optional[str]: Date in YYYY-MM-DD format or None if not found

    Examples:
        >>> extract_date("TCS price on 21st nov 2025")
        '2025-11-21'
        >>> extract_date("INFY data for november 15 2024")
        '2024-11-15'
    """
    logging.debug(f"DEBUG: extract_date() - Function called with query: '{query}'")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: extract_date() - Invalid input, returning None")
        return None

    query_lower = query.lower()
    logging.debug(f"DEBUG: extract_date() - Query converted to lowercase: '{query_lower}'")

    # Pattern 1: "21st nov 2025" or "21 nov 2025"
    pattern1 = r'(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
    match1 = re.search(pattern1, query_lower)
    if match1:
        day, month_name, year = match1.groups()
        month = REGEX_CONFIG["month_names"].get(month_name, month_name)
        try:
            # Validate the date
            date_obj = datetime(int(year), int(month), int(day))
            date_str = date_obj.strftime('%Y-%m-%d')
            logging.debug(f"DEBUG: extract_date() - Found date pattern 1: '{match1.group()}' -> '{date_str}'")
            return date_str
        except ValueError as e:
            logging.debug(f"DEBUG: extract_date() - Invalid date in pattern 1: {e}")
            return None

    # Pattern 2: "nov 21 2025" or "november 21, 2025"
    pattern2 = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})'
    match2 = re.search(pattern2, query_lower)
    if match2:
        month_name, day, year = match2.groups()
        month = REGEX_CONFIG["month_names"].get(month_name, month_name)
        try:
            # Validate the date
            date_obj = datetime(int(year), int(month), int(day))
            date_str = date_obj.strftime('%Y-%m-%d')
            logging.debug(f"DEBUG: extract_date() - Found date pattern 2: '{match2.group()}' -> '{date_str}'")
            return date_str
        except ValueError as e:
            logging.debug(f"DEBUG: extract_date() - Invalid date in pattern 2: {e}")
            return None

    # Pattern 3: "21/11/2025" or "11/21/2025" (ambiguous, assume DD/MM/YYYY)
    pattern3 = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
    match3 = re.search(pattern3, query)
    if match3:
        part1, part2, year = match3.groups()
        try:
            # Try DD/MM/YYYY first
            date_obj = datetime(int(year), int(part2), int(part1))
            date_str = date_obj.strftime('%Y-%m-%d')
            logging.debug(f"DEBUG: extract_date() - Found date pattern 3 (DD/MM/YYYY): '{match3.group()}' -> '{date_str}'")
            return date_str
        except ValueError:
            try:
                # Try MM/DD/YYYY
                date_obj = datetime(int(year), int(part1), int(part2))
                date_str = date_obj.strftime('%Y-%m-%d')
                logging.debug(f"DEBUG: extract_date() - Found date pattern 3 (MM/DD/YYYY): '{match3.group()}' -> '{date_str}'")
                return date_str
            except ValueError as e:
                logging.debug(f"DEBUG: extract_date() - Invalid date in pattern 3: {e}")
                return None

    logging.debug("DEBUG: extract_date() - No date found, returning None")
    return None


def extract_all_regex(query: str) -> Dict[str, Any]:
    """
    Extract all entities (tickers, timeframes, patterns, temporal contexts, dates) using regex.

    Args:
        query (str): The input query string

    Returns:
        Dict[str, Any]: Dictionary with extracted entities

    Example:
        >>> extract_all_regex("Is there a doji on INFY daily chart?")
        {'tickers': ['INFY'], 'ticker': 'INFY', 'timeframes': ['1d'], 'timeframe': '1d', 'pattern': 'doji', 'temporal_context': None, 'date': None}
        >>> extract_all_regex("Check price for RELIANCE and TCS 5m and weekly")
        {'tickers': ['RELIANCE', 'TCS'], 'ticker': 'RELIANCE', 'timeframes': ['5m', '1w'], 'timeframe': '5m', 'pattern': None, 'temporal_context': None, 'date': None}
        >>> extract_all_regex("show today's TCS data")
        {'tickers': ['TCS'], 'ticker': 'TCS', 'timeframes': [], 'timeframe': None, 'pattern': None, 'temporal_context': 'today', 'date': None}
        >>> extract_all_regex("TCS price on 21st nov 2025")
        {'tickers': ['TCS'], 'ticker': 'TCS', 'timeframes': [], 'timeframe': None, 'pattern': None, 'temporal_context': None, 'date': '2025-11-21'}
    """
    logging.debug(f"DEBUG: extract_all_regex() - Function called with query: '{query}'")

    tickers = extract_tickers(query)
    ticker = tickers[0] if tickers else None  # Backward compatibility
    timeframes = extract_timeframes(query)
    timeframe = timeframes[0] if timeframes else None  # Backward compatibility
    pattern = extract_pattern(query)
    temporal_context = extract_temporal_context(query)
    date = extract_date(query)

    result = {
        "tickers": tickers,
        "ticker": ticker,  # Backward compatibility
        "timeframes": timeframes,
        "timeframe": timeframe,  # Backward compatibility
        "pattern": pattern,
        "temporal_context": temporal_context,
        "date": date
    }

    logging.debug(f"DEBUG: extract_all_regex() - Extraction results: {result}")
    return result
