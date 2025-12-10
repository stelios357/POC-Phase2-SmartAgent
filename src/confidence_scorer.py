"""
Confidence scoring for query parsing results.

This module calculates confidence scores for extracted entities and overall parsing results.
Scores are based on extraction quality, context analysis, and validation results.
"""

from typing import Dict, Optional, Tuple
import re
import logging
from src.validator import validate_ticker, validate_timeframe, validate_pattern

# Configuration for confidence scoring
CONFIDENCE_CONFIG = {
    "weights": {
        "ticker": 0.5,
        "pattern": 0.3,
        "timeframe": 0.2
    },
    "base_scores": {
        "ticker": {"regex_match": 0.7, "validation": 0.3},
        "timeframe": {"regex_match": 0.8, "context": 0.2},
        "pattern": {"regex_match": 0.8, "context": 0.2}
    },
    "context_multipliers": {
        "ticker": {
            "preceded_by_keywords": 1.1,
            "capitalized": 1.05,
            "dollar_prefix": 1.1
        },
        "timeframe": {
            "near_chart": 1.1,
            "near_candle": 1.05,
            "explicit": 1.15
        },
        "pattern": {
            "question_form": 1.1,
            "explicit_mention": 1.15,
            "near_ticker": 1.05
        }
    }
}


def calculate_confidence(extracted: Dict[str, Optional[str]],
                        context: Dict[str, float],
                        validation_results: Dict[str, bool]) -> float:
    """
    Calculate overall confidence score for the parsing result.

    Args:
        extracted: Dictionary of extracted entities
        context: Context analysis scores
        validation_results: Validation results for each component

    Returns:
        float: Overall confidence score between 0.0 and 1.0

    Examples:
        >>> calculate_confidence(
        ...     {"ticker": "INFY", "timeframe": "1d", "pattern": "doji"},
        ...     {"ticker": 0.9, "timeframe": 0.8, "pattern": 0.95},
        ...     {"ticker": True, "timeframe": True, "pattern": True}
        ... )
        0.915
    """
    logging.debug(f"DEBUG: calculate_confidence() - Function called with extracted: {extracted}, validation_results: {validation_results}")

    ticker_score = score_ticker(extracted.get("ticker"), extracted, validation_results.get("ticker", False))
    logging.debug(f"DEBUG: calculate_confidence() - Ticker score: {ticker_score}")

    timeframe_score = score_timeframe(extracted.get("timeframe"), extracted, validation_results.get("timeframe", False))
    logging.debug(f"DEBUG: calculate_confidence() - Timeframe score: {timeframe_score}")

    pattern_score = score_pattern(extracted.get("pattern"), extracted, validation_results.get("pattern", False))
    logging.debug(f"DEBUG: calculate_confidence() - Pattern score: {pattern_score}")

    # Weighted combination
    weights = CONFIDENCE_CONFIG["weights"]
    overall_score = (
        ticker_score * weights["ticker"] +
        pattern_score * weights["pattern"] +
        timeframe_score * weights["timeframe"]
    )
    logging.debug(f"DEBUG: calculate_confidence() - Weighted calculation: {ticker_score}*{weights['ticker']} + {pattern_score}*{weights['pattern']} + {timeframe_score}*{weights['timeframe']} = {overall_score}")

    final_score = min(max(overall_score, 0.0), 1.0)
    logging.debug(f"DEBUG: calculate_confidence() - Final clamped score: {final_score}")
    return final_score


def score_ticker(ticker: Optional[str], extracted: Dict[str, Optional[str]], is_valid: bool) -> float:
    """
    Calculate confidence score for ticker extraction.

    Args:
        ticker: The extracted ticker symbol
        extracted: All extracted entities
        is_valid: Whether ticker passed validation

    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    logging.debug(f"DEBUG: score_ticker() - Function called with ticker: {ticker}, is_valid: {is_valid}")

    if not ticker:
        logging.debug("DEBUG: score_ticker() - No ticker provided, returning 0.0")
        return 0.0

    base_score = CONFIDENCE_CONFIG["base_scores"]["ticker"]["regex_match"]
    score = base_score
    logging.debug(f"DEBUG: score_ticker() - Initial score: {score} (base: {base_score})")

    # Validation boost
    if is_valid:
        validation_boost = CONFIDENCE_CONFIG["base_scores"]["ticker"]["validation"]
        score += validation_boost
        logging.debug(f"DEBUG: score_ticker() - Added validation boost: +{validation_boost}, score now: {score}")

    # Context analysis
    context_multiplier = 1.0
    logging.debug("DEBUG: score_ticker() - Starting context analysis")

    # Check for capitalization (strong indicator)
    if ticker.isupper():
        cap_multiplier = CONFIDENCE_CONFIG["context_multipliers"]["ticker"]["capitalized"]
        context_multiplier *= cap_multiplier
        logging.debug(f"DEBUG: score_ticker() - Applied capitalization multiplier: {cap_multiplier}")

    # Check for dollar prefix
    if ticker.startswith("$"):
        dollar_multiplier = CONFIDENCE_CONFIG["context_multipliers"]["ticker"]["dollar_prefix"]
        context_multiplier *= dollar_multiplier
        logging.debug(f"DEBUG: score_ticker() - Applied dollar prefix multiplier: {dollar_multiplier}")

    # Check if ticker appears in appropriate context
    # This would require the original query, but we can use heuristics based on extracted entities
    has_other_entities = any(extracted.get(key) for key in ["timeframe", "pattern"])
    if has_other_entities:
        keyword_multiplier = CONFIDENCE_CONFIG["context_multipliers"]["ticker"]["preceded_by_keywords"]
        context_multiplier *= keyword_multiplier
        logging.debug(f"DEBUG: score_ticker() - Applied keyword context multiplier: {keyword_multiplier} (has_other_entities: {has_other_entities})")

    score *= context_multiplier
    final_score = min(max(score, 0.0), 1.0)
    logging.debug(f"DEBUG: score_ticker() - Final score after context multiplier {context_multiplier}: {final_score}")
    return final_score


def score_timeframe(timeframe: Optional[str], extracted: Dict[str, Optional[str]], is_valid: bool) -> float:
    """
    Calculate confidence score for timeframe extraction.

    Args:
        timeframe: The extracted timeframe (may be default)
        extracted: All extracted entities
        is_valid: Whether timeframe passed validation

    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    logging.debug(f"DEBUG: score_timeframe() - Function called with timeframe: {timeframe}, is_valid: {is_valid}")

    # If timeframe is provided and valid, give high confidence
    if timeframe and is_valid:
        logging.debug("DEBUG: score_timeframe() - Timeframe provided and valid, calculating high confidence score")
        score = CONFIDENCE_CONFIG["base_scores"]["timeframe"]["regex_match"]
        score += CONFIDENCE_CONFIG["base_scores"]["timeframe"]["context"]
        logging.debug(f"DEBUG: score_timeframe() - Base score: {score}")

        # Context multiplier based on having other entities
        context_multiplier = 1.0
        has_ticker = bool(extracted.get("ticker"))
        has_pattern = bool(extracted.get("pattern"))
        logging.debug(f"DEBUG: score_timeframe() - Context check: has_ticker={has_ticker}, has_pattern={has_pattern}")

        if has_ticker or has_pattern:
            explicit_multiplier = CONFIDENCE_CONFIG["context_multipliers"]["timeframe"]["explicit"]
            context_multiplier *= explicit_multiplier
            logging.debug(f"DEBUG: score_timeframe() - Applied explicit context multiplier: {explicit_multiplier}")

        score *= context_multiplier
        final_score = min(max(score, 0.0), 1.0)
        logging.debug(f"DEBUG: score_timeframe() - Final score with context: {final_score}")
        return final_score

    # If no timeframe found but we have a query_type, assume default is acceptable
    # This prevents penalizing OHLCV queries that don't specify timeframe
    query_type = extracted.get("query_type")
    logging.debug(f"DEBUG: score_timeframe() - No timeframe provided, checking query_type: {query_type}")

    if query_type and not timeframe:
        # For OHLCV queries, default timeframe is reasonable
        if query_type == "ohlcv":
            logging.debug("DEBUG: score_timeframe() - Returning 0.8 for OHLCV query with default timeframe")
            return 0.8  # High confidence for default in OHLCV context
        else:
            logging.debug("DEBUG: score_timeframe() - Returning 0.6 for pattern query with default timeframe")
            return 0.6  # Medium confidence for default in pattern context

    logging.debug("DEBUG: score_timeframe() - No valid timeframe scenario, returning 0.0")
    return 0.0


def score_pattern(pattern: Optional[str], extracted: Dict[str, Optional[str]], is_valid: bool) -> float:
    """
    Calculate confidence score for pattern extraction.

    Args:
        pattern: The extracted pattern name
        extracted: All extracted entities
        is_valid: Whether pattern passed validation

    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    logging.debug(f"DEBUG: score_pattern() - Function called with pattern: {pattern}, is_valid: {is_valid}")

    if not pattern:
        logging.debug("DEBUG: score_pattern() - No pattern provided, returning 0.0")
        return 0.0

    score = CONFIDENCE_CONFIG["base_scores"]["pattern"]["regex_match"]
    logging.debug(f"DEBUG: score_pattern() - Initial score: {score}")

    # Validation boost
    if is_valid:
        validation_boost = CONFIDENCE_CONFIG["base_scores"]["pattern"]["context"]
        score += validation_boost
        logging.debug(f"DEBUG: score_pattern() - Added validation boost: +{validation_boost}, score now: {score}")

    # Context multiplier
    context_multiplier = 1.0
    has_ticker = bool(extracted.get("ticker"))
    logging.debug(f"DEBUG: score_pattern() - Context check: has_ticker={has_ticker}")

    if has_ticker:
        ticker_multiplier = CONFIDENCE_CONFIG["context_multipliers"]["pattern"]["near_ticker"]
        context_multiplier *= ticker_multiplier
        logging.debug(f"DEBUG: score_pattern() - Applied ticker proximity multiplier: {ticker_multiplier}")

    # Boost for explicit pattern mention (assuming this is checked elsewhere)
    # For now, we'll assume if pattern is found, it has some explicitness
    explicit_multiplier = CONFIDENCE_CONFIG["context_multipliers"]["pattern"]["explicit_mention"]
    context_multiplier *= explicit_multiplier
    logging.debug(f"DEBUG: score_pattern() - Applied explicit mention multiplier: {explicit_multiplier}")

    score *= context_multiplier
    final_score = min(max(score, 0.0), 1.0)
    logging.debug(f"DEBUG: score_pattern() - Final score after context multiplier {context_multiplier}: {final_score}")
    return final_score


def get_component_confidence_scores(extracted: Dict[str, Optional[str]],
                                  validation_results: Dict[str, bool]) -> Dict[str, float]:
    """
    Get confidence scores for each individual component.

    Args:
        extracted: Dictionary of extracted entities
        validation_results: Validation results for each component

    Returns:
        Dict[str, float]: Confidence scores for each component
    """
    logging.debug(f"DEBUG: get_component_confidence_scores() - Function called with extracted: {extracted}, validation_results: {validation_results}")

    ticker_score = score_ticker(extracted.get("ticker"), extracted, validation_results.get("ticker", False))
    timeframe_score = score_timeframe(extracted.get("timeframe"), extracted, validation_results.get("timeframe", False))
    pattern_score = score_pattern(extracted.get("pattern"), extracted, validation_results.get("pattern", False))

    result = {
        "ticker": ticker_score,
        "timeframe": timeframe_score,
        "pattern": pattern_score
    }

    logging.debug(f"DEBUG: get_component_confidence_scores() - Component scores: {result}")
    return result
