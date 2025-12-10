"""
Intent detection for natural language queries.

This module analyzes queries to determine whether they are asking for OHLCV data
or candlestick pattern detection.
"""

from typing import Dict, Optional, Any
import re
import logging

# Configuration for intent detection
INTENT_CONFIG = {
    "ohlcv_keywords": [
        "price", "ohlcv", "open", "high", "low", "close", "volume",
        "show me", "current", "latest", "candle", "candles", "chart",
        "data", "values", "information", "details", "compare"
    ],
    "pattern_keywords": [
        "doji", "hammer", "shooting star", "shooting-star", "marubozu",
        "pattern", "patterns", "is there", "check for", "looking for",
        "find", "detect", "analyze", "see if", "tell me if"
    ],
    "context_indicators": {
        "pattern": ["pattern", "formation", "signal", "indicator"],
        "ohlcv": ["price", "value", "data", "information", "chart"]
    }
}


def detect_query_intents(query: str, extracted: Dict[str, Any]) -> list[str]:
    """
    Detect all intents of the query based on keywords and extracted entities.
    Returns a list of intents instead of just one.

    Args:
        query (str): The input query string
        extracted (Dict[str, Any]): Extracted entities from regex

    Returns:
        list[str]: List of query types found - can include "ohlcv", "pattern", "current_price"

    Examples:
        >>> detect_query_intents("Show me INFY price", {"ticker": "INFY"})
        ['ohlcv']
        >>> detect_query_intents("Check price and find patterns in INFY", {"ticker": "INFY", "pattern": None})
        ['ohlcv', 'pattern']
        >>> detect_query_intents("TCS current price", {"ticker": "TCS"})
        ['current_price']
    """
    logging.debug(f"DEBUG: detect_query_intents() - Function called with query: '{query}', extracted: {extracted}")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: detect_query_intents() - Invalid input, returning empty list")
        return []

    query_lower = query.lower()
    logging.debug(f"DEBUG: detect_query_intents() - Query converted to lowercase: '{query_lower}'")

    intents = []

    # Priority 1: Check for current price queries (specific combination of current + price)
    current_price_score = 0
    if "current" in query_lower and "price" in query_lower:
        current_price_score = 2  # High priority for specific current price queries
        intents.append("current_price")
        logging.debug("DEBUG: detect_query_intents() - Added 'current_price' intent (high priority)")

    # Check for pattern keywords or explicit pattern
    pattern_score = sum(1 for keyword in INTENT_CONFIG["pattern_keywords"]
                       if keyword.lower() in query_lower)
    has_explicit_pattern = bool(extracted.get("pattern"))
    logging.debug(f"DEBUG: detect_query_intents() - Pattern detection: score={pattern_score}, explicit={has_explicit_pattern}")

    if pattern_score > 0 or has_explicit_pattern:
        intents.append("pattern")
        logging.debug("DEBUG: detect_query_intents() - Added 'pattern' intent")

    # Check for OHLCV keywords (but not if we already have current_price)
    ohlcv_score = sum(1 for keyword in INTENT_CONFIG["ohlcv_keywords"]
                     if keyword.lower() in query_lower)
    has_ticker = bool(extracted.get("ticker") or extracted.get("tickers"))
    has_timeframe = bool(extracted.get("timeframe") or extracted.get("timeframes"))
    logging.debug(f"DEBUG: detect_query_intents() - OHLCV detection: score={ohlcv_score}, ticker={has_ticker}, timeframe={has_timeframe}")

    # Only add OHLCV if we don't have current_price
    if current_price_score == 0 and (ohlcv_score > 0 or (has_ticker and has_timeframe)):
        intents.append("ohlcv")
        logging.debug("DEBUG: detect_query_intents() - Added 'ohlcv' intent")

    # If no specific intents detected but we have ticker/timeframe, assume OHLCV
    if not intents and (has_ticker or has_timeframe):
        intents.append("ohlcv")
        logging.debug("DEBUG: detect_query_intents() - Defaulted to 'ohlcv' intent")

    logging.debug(f"DEBUG: detect_query_intents() - Final intents: {intents}")
    return intents


def detect_query_intent(query: str, extracted: Dict[str, Optional[str]]) -> str:
    """
    Detect the primary intent of the query (backward compatibility).
    Returns the first detected intent for single-intent queries.
    """
    """
    Detect the intent of the query based on keywords and extracted entities.

    Args:
        query (str): The input query string
        extracted (Dict[str, Optional[str]]): Extracted entities from regex

    Returns:
        str: Query type - "ohlcv", "pattern", or "unknown"

    Examples:
        >>> detect_query_intent("Show me INFY price", {"ticker": "INFY"})
        'ohlcv'
        >>> detect_query_intent("Is there a doji on INFY?", {"pattern": "doji"})
        'pattern'
    """
    logging.debug(f"DEBUG: detect_query_intent() - Function called with query: '{query}', extracted: {extracted}")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: detect_query_intent() - Invalid input, returning 'unknown'")
        return "unknown"

    query_lower = query.lower()
    logging.debug(f"DEBUG: detect_query_intent() - Query converted to lowercase: '{query_lower}'")

    # Priority 1: If pattern is explicitly found in extracted entities
    if extracted.get("pattern"):
        logging.debug(f"DEBUG: detect_query_intent() - Priority 1: Pattern found in extracted entities: '{extracted.get('pattern')}', returning 'pattern'")
        return "pattern"

    # Priority 2: Check for pattern keywords
    pattern_score = sum(1 for keyword in INTENT_CONFIG["pattern_keywords"]
                       if keyword.lower() in query_lower)
    logging.debug(f"DEBUG: detect_query_intent() - Priority 2: Pattern keyword score: {pattern_score}")

    # Priority 3: Check for OHLCV keywords
    ohlcv_score = sum(1 for keyword in INTENT_CONFIG["ohlcv_keywords"]
                     if keyword.lower() in query_lower)
    logging.debug(f"DEBUG: detect_query_intent() - Priority 3: OHLCV keyword score: {ohlcv_score}")

    # Priority 4: If only ticker and timeframe found, assume OHLCV
    has_ticker = bool(extracted.get("ticker"))
    has_timeframe = bool(extracted.get("timeframe"))
    has_pattern = bool(extracted.get("pattern"))
    logging.debug(f"DEBUG: detect_query_intent() - Entity presence: ticker={has_ticker}, timeframe={has_timeframe}, pattern={has_pattern}")

    if has_ticker and has_timeframe and not has_pattern:
        logging.debug("DEBUG: detect_query_intent() - Priority 4: Has ticker and timeframe but no pattern, returning 'ohlcv'")
        return "ohlcv"

    # Priority 5: Compare keyword scores
    logging.debug(f"DEBUG: detect_query_intent() - Priority 5: Comparing scores - pattern: {pattern_score}, ohlcv: {ohlcv_score}")
    if pattern_score > ohlcv_score:
        logging.debug("DEBUG: detect_query_intent() - Pattern score higher, returning 'pattern'")
        return "pattern"
    elif ohlcv_score > pattern_score:
        logging.debug("DEBUG: detect_query_intent() - OHLCV score higher, returning 'ohlcv'")
        return "ohlcv"

    # Priority 6: Context-based heuristics
    logging.debug("DEBUG: detect_query_intent() - Priority 6: Applying context-based heuristics")
    if "pattern" in query_lower:
        logging.debug("DEBUG: detect_query_intent() - Found 'pattern' in query, returning 'pattern'")
        return "pattern"
    elif any(word in query_lower for word in ["price", "current", "show"]):
        logging.debug("DEBUG: detect_query_intent() - Found price/current/show keywords, returning 'ohlcv'")
        return "ohlcv"

    # Default fallback
    logging.debug("DEBUG: detect_query_intent() - No clear intent detected, returning 'unknown'")
    return "unknown"


def analyze_context(query: str) -> Dict[str, float]:
    """
    Analyze the context around extracted entities to provide confidence scores.

    Args:
        query (str): The input query string

    Returns:
        Dict[str, float]: Confidence scores for different components

    Examples:
        >>> analyze_context("Is there a doji on INFY daily?")
        {'ticker': 0.95, 'timeframe': 0.90, 'pattern': 0.98}
    """
    logging.debug(f"DEBUG: analyze_context() - Function called with query: '{query}'")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: analyze_context() - Invalid input, returning zero scores")
        return {"ticker": 0.0, "timeframe": 0.0, "pattern": 0.0}

    query_lower = query.lower()
    logging.debug(f"DEBUG: analyze_context() - Query converted to lowercase: '{query_lower}'")

    scores = {"ticker": 0.0, "timeframe": 0.0, "pattern": 0.0}

    # Analyze ticker context
    ticker_indicators = ["of", "for", "in", "on", "show", "price", "check"]
    ticker_score = 0.0
    logging.debug(f"DEBUG: analyze_context() - Analyzing ticker context with indicators: {ticker_indicators}")
    for indicator in ticker_indicators:
        if indicator in query_lower:
            ticker_score += 0.2
            logging.debug(f"DEBUG: analyze_context() - Found ticker indicator '{indicator}', score now: {ticker_score}")
    scores["ticker"] = min(ticker_score, 1.0)
    logging.debug(f"DEBUG: analyze_context() - Final ticker score: {scores['ticker']}")

    # Analyze timeframe context
    timeframe_indicators = ["chart", "candle", "daily", "hourly", "minute", "on"]
    timeframe_score = 0.0
    logging.debug(f"DEBUG: analyze_context() - Analyzing timeframe context with indicators: {timeframe_indicators}")
    for indicator in timeframe_indicators:
        if indicator in query_lower:
            timeframe_score += 0.15
            logging.debug(f"DEBUG: analyze_context() - Found timeframe indicator '{indicator}', score now: {timeframe_score}")
    scores["timeframe"] = min(timeframe_score, 1.0)
    logging.debug(f"DEBUG: analyze_context() - Final timeframe score: {scores['timeframe']}")

    # Analyze pattern context
    pattern_indicators = ["pattern", "is there", "check for", "looking for", "find"]
    pattern_score = 0.0
    logging.debug(f"DEBUG: analyze_context() - Analyzing pattern context with indicators: {pattern_indicators}")
    for indicator in pattern_indicators:
        if indicator in query_lower:
            pattern_score += 0.25
            logging.debug(f"DEBUG: analyze_context() - Found pattern indicator '{indicator}', score now: {pattern_score}")
    scores["pattern"] = min(pattern_score, 1.0)
    logging.debug(f"DEBUG: analyze_context() - Final pattern score: {scores['pattern']}")

    logging.debug(f"DEBUG: analyze_context() - Final context scores: {scores}")
    return scores


def get_intent_confidence(query: str, intent: str) -> float:
    """
    Calculate confidence score for the detected intent.

    Args:
        query (str): The input query string
        intent (str): The detected intent

    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    logging.debug(f"DEBUG: get_intent_confidence() - Function called with query: '{query}', intent: '{intent}'")

    if intent == "unknown":
        logging.debug("DEBUG: get_intent_confidence() - Intent is 'unknown', returning 0.0")
        return 0.0

    query_lower = query.lower()
    confidence = 0.0

    if intent == "pattern":
        logging.debug("DEBUG: get_intent_confidence() - Calculating confidence for pattern intent")
        # Count pattern-related keywords
        pattern_keywords = INTENT_CONFIG["pattern_keywords"]
        matches = sum(1 for keyword in pattern_keywords if keyword in query_lower)
        confidence = min(matches * 0.2, 1.0)
        logging.debug(f"DEBUG: get_intent_confidence() - Pattern matches: {matches}, confidence: {confidence}")
    elif intent == "ohlcv":
        logging.debug("DEBUG: get_intent_confidence() - Calculating confidence for OHLCV intent")
        # Count OHLCV-related keywords
        ohlcv_keywords = INTENT_CONFIG["ohlcv_keywords"]
        matches = sum(1 for keyword in ohlcv_keywords if keyword in query_lower)
        confidence = min(matches * 0.15, 1.0)
        logging.debug(f"DEBUG: get_intent_confidence() - OHLCV matches: {matches}, confidence: {confidence}")

    # Boost confidence if intent is clear from context
    logging.debug(f"DEBUG: get_intent_confidence() - Checking for context boost (current confidence: {confidence})")
    if confidence < 0.5:
        if "pattern" in query_lower and intent == "pattern":
            confidence = max(confidence, 0.8)
            logging.debug("DEBUG: get_intent_confidence() - Applied pattern context boost to 0.8")
        elif any(word in query_lower for word in ["price", "current"]) and intent == "ohlcv":
            confidence = max(confidence, 0.8)
            logging.debug("DEBUG: get_intent_confidence() - Applied OHLCV context boost to 0.8")

    logging.debug(f"DEBUG: get_intent_confidence() - Final confidence: {confidence}")
    return confidence
