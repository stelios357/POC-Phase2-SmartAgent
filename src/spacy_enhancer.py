"""
spaCy NER enhancement for query parsing.

This module uses spaCy for Named Entity Recognition to enhance ticker extraction
and improve confidence scores. This is optional and will gracefully degrade
if spaCy is not available.
"""

from typing import Dict, Optional, List
import logging

# Optional spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)

# Configuration for spaCy enhancement
SPACY_CONFIG = {
    "model": "en_core_web_sm",
    "entity_types": ["ORG", "GPE", "PERSON"],  # Types that could be companies
    "confidence_boost": {
        "org_match": 0.1,  # Boost confidence if spaCy finds ORG entity
        "gpe_match": 0.05,  # Smaller boost for GPE (geopolitical entities)
    },
    "indian_companies": [
        "reliance", "tcs", "infosys", "infy", "hdfc", "icici", "wipro",
        "lt", "bajaj", "maruti", "mahindra", "coal india", "ntpc", "power grid",
        "ongc", "hindustan unilever", "itc", "sun pharma", "dr reddy", "cipla"
    ]
}


def is_spacy_available() -> bool:
    """Check if spaCy is available and properly configured."""
    logging.debug(f"DEBUG: is_spacy_available() - Function called, returning: {SPACY_AVAILABLE}")
    return SPACY_AVAILABLE


def load_spacy_model() -> Optional[object]:
    """
    Load the spaCy NER model.

    Returns:
        Optional[object]: Loaded spaCy model or None if unavailable
    """
    logging.debug("DEBUG: load_spacy_model() - Function called")

    if not is_spacy_available():
        logger.warning("spaCy not available. Skipping NER enhancement.")
        logging.debug("DEBUG: load_spacy_model() - spaCy not available, returning None")
        return None

    try:
        logging.debug(f"DEBUG: load_spacy_model() - Attempting to load spaCy model: {SPACY_CONFIG['model']}")
        nlp = spacy.load(SPACY_CONFIG["model"])
        logger.info(f"Loaded spaCy model: {SPACY_CONFIG['model']}")
        logging.debug("DEBUG: load_spacy_model() - Model loaded successfully")
        return nlp
    except OSError:
        logger.warning(f"Could not load spaCy model: {SPACY_CONFIG['model']}")
        logging.debug(f"DEBUG: load_spacy_model() - OSError loading model: {SPACY_CONFIG['model']}")
        return None
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        logging.debug(f"DEBUG: load_spacy_model() - Unexpected exception loading model: {e}")
        return None


def extract_entities_with_spacy(query: str, nlp_model: Optional[object] = None) -> List[Dict]:
    """
    Extract named entities from query using spaCy.

    Args:
        query (str): Input query string
        nlp_model: Pre-loaded spaCy model (optional)

    Returns:
        List[Dict]: List of extracted entities with type and confidence
    """
    logging.debug(f"DEBUG: extract_entities_with_spacy() - Function called with query: '{query}', nlp_model provided: {nlp_model is not None}")

    if not query or not isinstance(query, str):
        logging.debug("DEBUG: extract_entities_with_spacy() - Invalid query input, returning empty list")
        return []

    if not is_spacy_available():
        logging.debug("DEBUG: extract_entities_with_spacy() - spaCy not available, returning empty list")
        return []

    # Load model if not provided
    if nlp_model is None:
        logging.debug("DEBUG: extract_entities_with_spacy() - No model provided, loading model")
        nlp_model = load_spacy_model()
    else:
        logging.debug("DEBUG: extract_entities_with_spacy() - Using provided model")

    if nlp_model is None:
        logging.debug("DEBUG: extract_entities_with_spacy() - Model loading failed, returning empty list")
        return []

    try:
        logging.debug("DEBUG: extract_entities_with_spacy() - Processing query with spaCy")
        doc = nlp_model(query)
        entities = []

        logging.debug(f"DEBUG: extract_entities_with_spacy() - Found {len(doc.ents)} total entities")
        for ent in doc.ents:
            if ent.label_ in SPACY_CONFIG["entity_types"]:
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, '_.confidence', 0.8)  # spaCy confidence if available
                }
                entities.append(entity_info)
                logging.debug(f"DEBUG: extract_entities_with_spacy() - Added entity: {entity_info}")

        logging.debug(f"DEBUG: extract_entities_with_spacy() - Returning {len(entities)} relevant entities")
        return entities

    except Exception as e:
        logger.error(f"Error during spaCy NER: {e}")
        logging.debug(f"DEBUG: extract_entities_with_spacy() - Exception during NER: {e}")
        return []


def enhance_with_spacy(query: str, extracted: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Enhance extracted entities using spaCy NER.

    Args:
        query (str): Original query string
        extracted (Dict[str, Optional[str]]): Regex-extracted entities

    Returns:
        Dict[str, Optional[str]]: Enhanced extraction results
    """
    logging.debug(f"DEBUG: enhance_with_spacy() - Function called with query: '{query}', extracted: {extracted}")

    if not query or not extracted:
        logging.debug("DEBUG: enhance_with_spacy() - Invalid input, returning original extracted")
        return extracted

    # Copy original extracted results
    enhanced = extracted.copy()
    logging.debug(f"DEBUG: enhance_with_spacy() - Starting with extracted copy: {enhanced}")

    # Extract entities with spaCy
    entities = extract_entities_with_spacy(query)
    logging.debug(f"DEBUG: enhance_with_spacy() - Extracted {len(entities)} entities with spaCy")

    if not entities:
        logging.debug("DEBUG: enhance_with_spacy() - No entities found, returning original extracted")
        return enhanced

    # Look for potential ticker enhancements
    existing_ticker = extracted.get("ticker")
    logging.debug(f"DEBUG: enhance_with_spacy() - Existing ticker: {existing_ticker}")

    for entity in entities:
        entity_text = entity["text"].upper()
        entity_label = entity["label"]
        logging.debug(f"DEBUG: enhance_with_spacy() - Processing entity: {entity_text} (label: {entity_label})")

        # Focus on ORG entities as potential tickers
        if entity_label == "ORG":
            logging.debug("DEBUG: enhance_with_spacy() - Found ORG entity, checking for ticker enhancement")
            # Check if this could be a company ticker
            if not existing_ticker:
                # No ticker found by regex, spaCy found an ORG - potential enhancement
                # Check if it looks like a ticker (short, uppercase)
                if len(entity_text) <= 10 and entity_text.replace(".", "").isalnum():
                    enhanced["ticker"] = entity_text
                    logger.info(f"spaCy enhanced ticker extraction: {entity_text}")
                    logging.debug(f"DEBUG: enhance_with_spacy() - Enhanced ticker extraction: {entity_text}")
                    break

            elif existing_ticker and entity_text != existing_ticker:
                # Different ticker found - could be a conflict or additional context
                # For now, keep the regex result but log the discrepancy
                logger.debug(f"spaCy found different ORG entity: {entity_text} vs {existing_ticker}")
                logging.debug(f"DEBUG: enhance_with_spacy() - Found different ORG entity: {entity_text} vs existing {existing_ticker}")

    logging.debug(f"DEBUG: enhance_with_spacy() - Returning enhanced results: {enhanced}")
    return enhanced


def get_spacy_confidence_boost(query: str, extracted: Dict[str, Optional[str]]) -> float:
    """
    Calculate confidence boost from spaCy NER results.

    Args:
        query (str): Original query string
        extracted (Dict[str, Optional[str]]): Extracted entities

    Returns:
        float: Confidence boost between 0.0 and 1.0
    """
    logging.debug(f"DEBUG: get_spacy_confidence_boost() - Function called with query: '{query}', extracted: {extracted}")

    if not query or not extracted:
        logging.debug("DEBUG: get_spacy_confidence_boost() - Invalid input, returning 0.0")
        return 0.0

    entities = extract_entities_with_spacy(query)
    boost = 0.0
    logging.debug(f"DEBUG: get_spacy_confidence_boost() - Extracted {len(entities)} entities for confidence calculation")

    ticker = extracted.get("ticker")
    logging.debug(f"DEBUG: get_spacy_confidence_boost() - Ticker from extracted: {ticker}")

    if ticker and entities:
        logging.debug("DEBUG: get_spacy_confidence_boost() - Calculating boost based on ticker-entity matching")

        # Check if spaCy found an ORG entity that matches or relates to our ticker
        for entity in entities:
            if entity["label"] == "ORG":
                entity_upper = entity["text"].upper()
                logging.debug(f"DEBUG: get_spacy_confidence_boost() - Checking ORG entity: {entity_upper}")
                # Exact match or ticker is part of entity
                if ticker in entity_upper or entity_upper in ticker:
                    org_boost = SPACY_CONFIG["confidence_boost"]["org_match"]
                    boost += org_boost
                    logging.debug(f"DEBUG: get_spacy_confidence_boost() - Applied ORG match boost: +{org_boost}")
                    break

        # Additional boost for GPE entities (might indicate company location context)
        gpe_count = 0
        for entity in entities:
            if entity["label"] == "GPE":
                gpe_boost = SPACY_CONFIG["confidence_boost"]["gpe_match"]
                boost += gpe_boost
                gpe_count += 1
                logging.debug(f"DEBUG: get_spacy_confidence_boost() - Applied GPE boost: +{gpe_boost} for entity {entity['text']}")

        logging.debug(f"DEBUG: get_spacy_confidence_boost() - Total boost before capping: {boost}, GPE entities: {gpe_count}")

    final_boost = min(boost, 0.2)  # Cap the boost at 0.2
    logging.debug(f"DEBUG: get_spacy_confidence_boost() - Final boost after capping: {final_boost}")
    return final_boost


def map_company_name_to_ticker(company_name: str) -> Optional[str]:
    """
    Map common company names to ticker symbols.

    Args:
        company_name (str): Company name from NER

    Returns:
        Optional[str]: Ticker symbol if mapping exists
    """
    logging.debug(f"DEBUG: map_company_name_to_ticker() - Function called with company_name: '{company_name}'")

    if not company_name:
        logging.debug("DEBUG: map_company_name_to_ticker() - Empty company name, returning None")
        return None

    company_lower = company_name.lower().strip()
    logging.debug(f"DEBUG: map_company_name_to_ticker() - Normalized company name: '{company_lower}'")

    # Simple mapping for common Indian companies
    mapping = {
        "reliance industries": "RELIANCE",
        "tata consultancy services": "TCS",
        "infosys": "INFY",
        "hindustan unilever": "HINDUNILVR",
        "itc": "ITC",
        "sun pharmaceutical": "SUNPHARMA",
        "dr reddy's laboratories": "DRREDDY",
        "cipla": "CIPLA"
    }

    result = mapping.get(company_lower)
    logging.debug(f"DEBUG: map_company_name_to_ticker() - Mapping result: {company_lower} -> {result}")
    return result


# Global model cache to avoid reloading
_nlp_model = None

def get_spacy_model() -> Optional[object]:
    """Get cached spaCy model."""
    logging.debug("DEBUG: get_spacy_model() - Function called")
    global _nlp_model
    if _nlp_model is None:
        logging.debug("DEBUG: get_spacy_model() - Model not cached, loading model")
        _nlp_model = load_spacy_model()
        logging.debug(f"DEBUG: get_spacy_model() - Model loaded and cached: {_nlp_model is not None}")
    else:
        logging.debug("DEBUG: get_spacy_model() - Returning cached model")
    return _nlp_model
