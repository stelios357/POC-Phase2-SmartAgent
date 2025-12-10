"""
Main query parser orchestration module.

This module coordinates all components of the natural language query parser:
regex extraction, intent detection, validation, confidence scoring, and optional
spaCy enhancement to provide structured output for financial queries.
"""

import time
from typing import Dict, Optional, Any
import logging

from src.regex_extractor import extract_all_regex, extract_tickers
from src.intent_detector import detect_query_intent, detect_query_intents, analyze_context
from src.validator import validate_all_components
from src.confidence_scorer import calculate_confidence, get_component_confidence_scores
from src.spacy_enhancer import enhance_with_spacy, get_spacy_confidence_boost, is_spacy_available
from src.logging_config import log_performance_metric
from config.settings import TICKER_CONFIG
from src.yfinance_wrapper import resolve_ticker as yf_resolve_ticker

logger = logging.getLogger(__name__)

# Configuration
PARSER_CONFIG = {
    "use_spacy": True,
    "min_confidence_threshold": 0.6,  # Adjusted for reasonable default handling
    "default_timeframe": "1d",
    "max_processing_time_ms": 500
}


def clean_and_normalize_query(query: str) -> str:
    """
    Clean and normalize the input query.

    Args:
        query (str): Raw input query

    Returns:
        str: Cleaned and normalized query
    """
    logging.debug(f"DEBUG: clean_and_normalize_query() - Function called with query: '{query}'")

    if not query or not isinstance(query, str):
        logging.debug(f"DEBUG: clean_and_normalize_query() - Invalid input: {query} (type: {type(query)})")
        return ""

    logging.debug("DEBUG: clean_and_normalize_query() - Starting basic cleaning")
    # Basic cleaning
    cleaned = query.strip()
    logging.debug(f"DEBUG: clean_and_normalize_query() - After strip: '{cleaned}'")

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())
    logging.debug(f"DEBUG: clean_and_normalize_query() - After whitespace normalization: '{cleaned}'")

    # Convert to lowercase for consistent processing
    cleaned = cleaned.lower()
    logging.debug(f"DEBUG: clean_and_normalize_query() - After lowercase conversion: '{cleaned}'")

    logging.debug("DEBUG: clean_and_normalize_query() - Query cleaning completed successfully")
    return cleaned


def build_error_response(error_message: str, suggestions: Optional[list] = None) -> Dict[str, Any]:
    """
    Build standardized error response.

    Args:
        error_message (str): Error message
        suggestions (Optional[list]): List of suggestions

    Returns:
        Dict[str, Any]: Error response dictionary
    """
    logging.debug(f"DEBUG: build_error_response() - Function called with error_message: '{error_message}', suggestions: {suggestions}")

    result = {
        "error": True,
        "message": error_message,
        "suggestions": suggestions or [],
        "tickers": [],
        "ticker": None,  # Backward compatibility
        "timeframes": [],
        "timeframe": None,  # Backward compatibility
        "query_types": [],
        "query_type": None,  # Backward compatibility
        "pattern": None,
        "confidence": 0.0,
        "raw_query": "",
        "extracted_entities": {
            "tickers_found": [],
            "ticker_found": None,  # Backward compatibility
            "timeframes_found": [],
            "timeframe_found": None,  # Backward compatibility
            "pattern_found": None,
            "explicit_ohlcv": False
        }
    }

    logging.debug("DEBUG: build_error_response() - Error response built successfully")
    return result


def parse_query(query: str,
                use_spacy: bool = True,
                validate_ticker: bool = True,
                exchange: str = "nse") -> Dict[str, Any]:
    """
    Parse natural language query into structured financial data request.

    This function orchestrates the entire parsing pipeline:
    1. Clean and normalize query
    2. Extract entities using regex
    3. Detect query intent (ohlcv vs pattern)
    4. Enhance with spaCy NER (optional)
    5. Validate extracted components
    6. Calculate confidence scores
    7. Return structured output

    Args:
        query (str): Natural language query string
        use_spacy (bool): Whether to use spaCy enhancement (default: True)
        validate_ticker (bool): Whether to validate ticker against yfinance (default: True)

    Returns:
        Dict[str, Any]: Structured parsing result with the following keys:
            - tickers: list[str] (uppercase ticker symbols, may be single item)
            - ticker: str | None (first ticker for backward compatibility)
            - timeframes: list[str] (all timeframes found, may be single item)
            - timeframe: str | None (first timeframe for backward compatibility)
            - query_type: str | None (ohlcv|pattern|multi_intent)
            - query_types: list[str] (all detected intents)
            - pattern: str | None (doji|hammer|shooting_star|marubozu)
            - confidence: float (0.0-1.0)
            - raw_query: str (original input)
            - extracted_entities: dict (detailed extraction info)
            - error: bool (True if parsing failed)
            - message: str | None (error message if applicable)

    Examples:
        >>> parse_query("Is there a Doji on the daily chart of INFY?")
        {
            "tickers": ["INFY"],
            "ticker": "INFY",
            "timeframe": "1d",
            "query_type": "pattern",
            "pattern": "doji",
            "confidence": 0.95,
            "raw_query": "Is there a Doji on the daily chart of INFY?",
            "extracted_entities": {...},
            "error": False
        }

        >>> parse_query("Check price for RELIANCE and TCS 5m")
        {
            "tickers": ["RELIANCE", "TCS"],
            "ticker": "RELIANCE",
            "timeframe": "5m",
            "query_type": "ohlcv",
            "pattern": null,
            "confidence": 0.75,
            "raw_query": "Check price for RELIANCE and TCS 5m",
            "extracted_entities": {...},
            "error": False
        }
    """
    logging.debug(
        f"DEBUG: parse_query() - Function called with query: '{query}', "
        f"use_spacy: {use_spacy}, validate_ticker: {validate_ticker}, exchange: {exchange}"
    )
    start_time = time.time()
    logging.debug(f"DEBUG: parse_query() - Start time recorded: {start_time}")

    # Input validation
    if not query or not isinstance(query, str):
        logging.debug(f"DEBUG: parse_query() - Invalid input validation failed: {query} (type: {type(query)})")
        return build_error_response("Invalid query: must be a non-empty string")

    original_query = query
    logging.debug(f"DEBUG: parse_query() - Original query stored: '{original_query}'")

    query = clean_and_normalize_query(query)
    logging.debug(f"DEBUG: parse_query() - Query after cleaning: '{query}'")

    if not query:
        logging.debug("DEBUG: parse_query() - Query is empty after cleaning")
        return build_error_response("Query is empty after cleaning")

    logger.info(f"Parsing query: {original_query}")
    logging.debug("DEBUG: parse_query() - Starting parsing pipeline")

    try:
        logging.debug("DEBUG: parse_query() - Step 1: Starting regex entity extraction")
        # Step 1: Extract entities using regex
        extracted = extract_all_regex(original_query)  # Use original for case sensitivity
        logger.debug(f"Regex extraction: {extracted}")
        logging.debug(f"DEBUG: parse_query() - Regex extraction completed: {extracted}")

        logging.debug("DEBUG: parse_query() - Step 2: Starting query intent detection")
        # Step 2: Detect query intents (can be multiple)
        query_types = detect_query_intents(original_query, extracted)
        query_type = query_types[0] if query_types else "unknown"  # Primary intent for backward compatibility
        logger.debug(f"Detected intents: {query_types}, primary: {query_type}")
        logging.debug(f"DEBUG: parse_query() - Intent detection completed: {query_types} (primary: {query_type})")

        logging.debug("DEBUG: parse_query() - Step 3: Starting context analysis")
        # Step 3: Analyze context for confidence scoring
        context_scores = analyze_context(original_query)
        logger.debug(f"Context scores: {context_scores}")
        logging.debug(f"DEBUG: parse_query() - Context analysis completed: {context_scores}")

        # Step 4: Optional spaCy enhancement
        logging.debug(f"DEBUG: parse_query() - Step 4: Checking spaCy enhancement (use_spacy: {use_spacy})")
        if use_spacy and is_spacy_available():
            logging.debug("DEBUG: parse_query() - spaCy is available and enabled, applying enhancement")
            try:
                extracted = enhance_with_spacy(original_query, extracted)
                spacy_boost = get_spacy_confidence_boost(original_query, extracted)
                logger.debug(f"spaCy enhancement applied, boost: {spacy_boost}")
                logging.debug(f"DEBUG: parse_query() - spaCy enhancement completed, boost: {spacy_boost}")
            except Exception as e:
                logger.warning(f"spaCy enhancement failed: {e}")
                logging.debug(f"DEBUG: parse_query() - spaCy enhancement failed: {e}")
                spacy_boost = 0.0
        else:
            logging.debug(f"DEBUG: parse_query() - spaCy enhancement skipped (use_spacy: {use_spacy}, available: {is_spacy_available()})")
            spacy_boost = 0.0

            # Post-extraction refinement: Check for common names if ticker is missing or ambiguous
        if not extracted.get("ticker"):
            logging.debug("DEBUG: parse_query() - Ticker missing, checking common mappings")
            # Check if any common mapping key is in the query
            # We check mostly single words from the mapping
            # This is a simple heuristic
            query_upper = original_query.upper()
            
            # Simple punctuation cleaning for name matching
            import string
            for char in string.punctuation:
                query_upper = query_upper.replace(char, " ")
            query_upper = " ".join(query_upper.split())
            
            for name, ticker in TICKER_CONFIG.get("common_mappings", {}).items():
                # Check for exact word match to avoid false positives
                # e.g. "L&T" -> "LT"
                # We need to handle special chars in name lookup if needed, but for now exact match
                if f" {name} " in f" {query_upper} ": # simplistic word boundary check
                     logging.debug(f"DEBUG: parse_query() - Found common name '{name}' -> '{ticker}'")
                     symbol_only = ticker.split('.')[0]
                     extracted["ticker"] = symbol_only  # Backward compatibility
                     extracted["tickers"] = [symbol_only]
                     break

        # Step 4.5: Normalize tickers for yfinance compatibility with exchange context
        logging.debug("DEBUG: parse_query() - Step 4.5: Normalizing tickers for yfinance compatibility")
        if extracted.get("tickers"):
            normalized_tickers = []
            for ticker in extracted["tickers"]:
                original_ticker = ticker
                normalized_ticker = yf_resolve_ticker(ticker, exchange=exchange)
                if normalized_ticker != original_ticker:
                    logging.debug(
                        f"DEBUG: parse_query() - Ticker normalized: "
                        f"'{original_ticker}' -> '{normalized_ticker}' for exchange '{exchange}'"
                    )
                normalized_tickers.append(normalized_ticker)
            extracted["tickers"] = normalized_tickers
            extracted["ticker"] = normalized_tickers[0] if normalized_tickers else None  # Update backward compatibility field

        # Step 5: Validate components
        logging.debug("DEBUG: parse_query() - Step 5: Starting component validation")
        validation_results = validate_all_components(
            extracted.get("ticker"),
            extracted.get("timeframe"),
            extracted.get("pattern"),
            query_type,
            check_yfinance=validate_ticker,
            exchange=exchange,
        )
        logging.debug(f"DEBUG: parse_query() - Initial validation results: {validation_results}")

        # Update validation results for defaults
        if not extracted.get("timeframe") and PARSER_CONFIG["default_timeframe"]:
            validation_results["timeframe"] = True  # Default timeframe is always valid
            logging.debug(f"DEBUG: parse_query() - Applied default timeframe validation: {validation_results['timeframe']}")
        logger.debug(f"Validation results: {validation_results}")

        # Step 6: Apply defaults for confidence calculation
        logging.debug("DEBUG: parse_query() - Step 6: Applying defaults for confidence calculation")
        final_timeframes = extracted.get("timeframes", [])
        final_timeframe_for_confidence = extracted.get("timeframe") or PARSER_CONFIG["default_timeframe"]
        final_extracted = extracted.copy()
        final_extracted["timeframe"] = final_timeframe_for_confidence
        logging.debug(f"DEBUG: parse_query() - Final extracted for confidence: {final_extracted}, timeframes: {final_timeframes}")

        # Calculate confidence scores (using final values including defaults)
        logging.debug("DEBUG: parse_query() - Calculating confidence scores")
        base_confidence = calculate_confidence(final_extracted, context_scores, validation_results)
        final_confidence = min(base_confidence + spacy_boost, 1.0)
        logging.debug(f"DEBUG: parse_query() - Confidence calculation: base={base_confidence:.3f}, spacy_boost={spacy_boost:.3f}, final={final_confidence:.3f}")

        component_confidences = get_component_confidence_scores(final_extracted, validation_results)
        logging.debug(f"DEBUG: parse_query() - Component confidences: {component_confidences}")

        # Step 7: Apply defaults and handle missing values for final output
        logging.debug("DEBUG: parse_query() - Step 7: Applying defaults for final output")
        final_tickers = extracted.get("tickers", [])
        final_ticker = extracted.get("ticker")  # Backward compatibility - first ticker
        final_timeframes = extracted.get("timeframes", [])
        final_timeframe = extracted.get("timeframe") or PARSER_CONFIG["default_timeframe"]  # Backward compatibility
        final_pattern = extracted.get("pattern")
        logging.debug(f"DEBUG: parse_query() - Final values: tickers={final_tickers}, ticker={final_ticker}, timeframes={final_timeframes}, timeframe={final_timeframe}, pattern={final_pattern}, query_types={query_types}")

        # Step 8: Build extracted entities info
        logging.debug("DEBUG: parse_query() - Step 8: Building extracted entities info")
        extracted_entities = {
            "tickers_found": final_tickers,
            "ticker_found": extracted.get("ticker"),  # Backward compatibility
            "timeframes_found": final_timeframes,
            "timeframe_found": extracted.get("timeframe"),  # Backward compatibility
            "pattern_found": extracted.get("pattern"),
            "date_found": extracted.get("date"),
            "explicit_ohlcv": "ohlcv" in query_types,
            "component_confidences": component_confidences
        }
        logging.debug(f"DEBUG: parse_query() - Extracted entities built: {extracted_entities}")

        # Step 9: Final validation and error handling
        logging.debug("DEBUG: parse_query() - Step 9: Building success response")
        success_response = {
            "tickers": final_tickers,
            "ticker": final_ticker,  # Backward compatibility
            "timeframes": final_timeframes,
            "timeframe": final_timeframe,  # Backward compatibility
            "query_types": query_types,
            "query_type": query_type,  # Primary intent for backward compatibility
            "pattern": final_pattern,
            "temporal_context": extracted.get("temporal_context"),
            "date": extracted.get("date"),
            "confidence": round(final_confidence, 3),
            "raw_query": original_query,
            "extracted_entities": extracted_entities,
            "error": False,
            "message": None
        }
        logging.debug(f"DEBUG: parse_query() - Success response initialized: confidence={success_response['confidence']}, error={success_response['error']}")

        # Check for critical failures
        logging.debug("DEBUG: parse_query() - Checking for critical failures")
        if query_type == "unknown":
            success_response["error"] = True
            success_response["message"] = "Could not determine query intent"
            logging.debug("DEBUG: parse_query() - Critical failure: unknown query type")

        elif final_confidence < PARSER_CONFIG["min_confidence_threshold"]:
            success_response["error"] = True
            success_response["message"] = f"Low confidence score: {final_confidence:.2f}"
            success_response["suggestions"] = [
                "Try using clearer ticker symbols (e.g., INFY, TCS)",
                "Specify timeframe explicitly (daily, hourly, etc.)",
                "Use standard pattern names (doji, hammer, etc.)"
            ]
            logging.debug(f"DEBUG: parse_query() - Critical failure: low confidence {final_confidence:.2f} < {PARSER_CONFIG['min_confidence_threshold']}")

        # Log performance
        processing_time = (time.time() - start_time) * 1000
        log_performance_metric("query_parsing", processing_time)
        logging.debug(f"DEBUG: parse_query() - Performance logged: {processing_time:.2f}ms")

        if processing_time > PARSER_CONFIG["max_processing_time_ms"]:
            logger.warning(f"Query parsing took {processing_time:.1f}ms (exceeds {PARSER_CONFIG['max_processing_time_ms']}ms limit)")
            logging.debug(f"DEBUG: parse_query() - Performance warning: {processing_time:.2f}ms > {PARSER_CONFIG['max_processing_time_ms']}ms")

        logger.info(f"Query parsed successfully: ticker={final_ticker}, type={query_type}, confidence={final_confidence:.2f}")
        logging.debug("DEBUG: parse_query() - Returning success response")
        return success_response

    except Exception as e:
        logger.error(f"Unexpected error during query parsing: {e}", exc_info=True)
        logging.debug(f"DEBUG: parse_query() - Unexpected exception caught: {type(e).__name__}: {str(e)}")
        result = build_error_response(
            f"Parsing failed due to internal error: {str(e)}",
            ["Try rephrasing your query", "Check for typos in ticker symbols"]
        )
        logging.debug("DEBUG: parse_query() - Returning error response due to exception")
        return result
