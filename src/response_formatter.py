# Response Formatter for Candlestick Pattern Analysis
# Formats pattern detection results into structured JSON output

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

class ResponseFormatter:
    """Formats candlestick pattern analysis results into JSON responses"""

    def __init__(self):
        logging.debug("DEBUG: ResponseFormatter.__init__() - Initializing response formatter")
        self.supported_timeframes = {
            "1m": "1 minute",
            "5m": "5 minutes",
            "15m": "15 minutes",
            "30m": "30 minutes",
            "1h": "1 hour",
            "1d": "1 day",
            "1w": "1 week"
        }
        logging.debug("DEBUG: ResponseFormatter.__init__() - Response formatter initialized")

    def format_pattern_analysis_response(self, query_data: Dict, latest_candle: Dict,
                                       detected_patterns: List[Dict]) -> Dict:
        logging.debug(f"DEBUG: ResponseFormatter.format_pattern_analysis_response() - Called with {len(detected_patterns)} patterns")
        # [Function implementation with debug logs]
        """
        Format complete response for pattern analysis

        Args:
            query_data: Parsed query information (ticker, timeframe, etc.)
            latest_candle: Latest OHLCV candle data
            detected_patterns: List of detected patterns from pattern_detector

        Returns:
            Formatted JSON response
        """
        ticker = query_data.get('ticker', '').upper()
        timeframe = query_data.get('timeframe', '1d')
        query_type = query_data.get('query_type', 'pattern_detection')

        # Format the latest candle data
        formatted_candle = self._format_candle_data(latest_candle)

        # Format detected patterns
        formatted_patterns = self._format_patterns(detected_patterns)

        # Build response structure based on cs2.md specification
        response = {
            "query": self._build_query_string(query_data),
            "ticker": ticker,
            "timeframe": timeframe,
            "timeframe_description": self.supported_timeframes.get(timeframe, timeframe),
            "query_type": query_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "latest_candle": formatted_candle,
            "patterns_detected": formatted_patterns,
            "analysis_summary": self._build_analysis_summary(detected_patterns)
        }

        return response

    def format_error_response(self, error_type: str, message: str,
                            query_data: Optional[Dict] = None) -> Dict:
        logging.debug(
            f"DEBUG: ResponseFormatter.format_error_response() - Called with "
            f"error_type: {error_type}, message: {message}, "
            f"has_query_data: {query_data is not None}"
        )
        """
        Format error response

        Args:
            error_type: Type of error (e.g., "INVALID_TICKER", "NETWORK_ERROR")
            message: Error message
            query_data: Original query data if available

        Returns:
            Formatted error response
        """
        response = {
            "success": False,
            "error": {
                "type": error_type,
                "message": message,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }

        if query_data:
            response.update({
                "query": self._build_query_string(query_data),
                "ticker": query_data.get('ticker', '').upper(),
                "timeframe": query_data.get('timeframe', '1d')
            })

        return response

    def _format_candle_data(self, candle: Dict) -> Dict:
        logging.debug(f"DEBUG: ResponseFormatter._format_candle_data() - Called with candle data")
        # [Function implementation]
        """
        Format candle data according to specification

        Expected input: {
            "timestamp": "2024-01-15",
            "open": 150.25,
            "high": 152.50,
            "low": 149.00,
            "close": 151.75,
            "volume": 45000000,
            "change": 1.50,
            "change_percent": 1.00
        }
        """
        change_pct_val = candle.get('change_percent')
        change_val = candle.get('change')
        volume_val = candle.get('volume', 0)

        formatted = {
            "date": candle.get('timestamp', datetime.utcnow().strftime('%Y-%m-%d')),
            "open": round(float(candle.get('open', 0)), 2),
            "high": round(float(candle.get('high', 0)), 2),
            "low": round(float(candle.get('low', 0)), 2),
            "close": round(float(candle.get('close', 0)), 2),
            "volume": int(volume_val) if volume_val is not None else 0,
            "change_pct": round(float(change_pct_val), 2) if change_pct_val is not None else 0.0,
            "change": round(float(change_val), 2) if change_val is not None else 0.0,
            "previous_close": round(float(candle.get('previous_close', 0)), 2),
            "price_range": round(float(candle.get('high', 0)) - float(candle.get('low', 0)), 2),
            "body_size": round(abs(float(candle.get('close', 0)) - float(candle.get('open', 0))), 2)
        }

        # Add additional fields if available
        if 'currency' in candle:
            formatted['currency'] = candle['currency']
        if 'market_state' in candle:
            formatted['market_state'] = candle['market_state']
        if 'exchange' in candle:
            formatted['exchange'] = candle['exchange']

        return formatted

    def _format_patterns(self, patterns: List[Dict]) -> List[Dict]:
        logging.debug(f"DEBUG: ResponseFormatter._format_patterns() - Called with {len(patterns)} patterns")
        # [Function implementation]
        """
        Format detected patterns for response

        Expected pattern structure from pattern_detector:
        {
            "pattern": "Doji",
            "type": "Long-Legged",
            "detected": True,
            "confidence": 0.65,
            "signal": "Market indecision",
            "description": "...",
            "metrics": {...}
        }
        """
        formatted_patterns = []

        for pattern in patterns:
            formatted_pattern = {
                "pattern": pattern.get("pattern", "Unknown"),
                "type": pattern.get("type", "Common"),
                "detected": pattern.get("detected", False),
                "confidence": round(float(pattern.get("confidence", 0)), 2),
                "signal": pattern.get("signal", ""),
                "description": pattern.get("description", ""),
                "pattern_strength": self._calculate_pattern_strength(pattern.get("confidence", 0))
            }

            # Include metrics if available
            if "metrics" in pattern:
                formatted_pattern["technical_metrics"] = pattern["metrics"]

            formatted_patterns.append(formatted_pattern)

        return formatted_patterns

    def _build_query_string(self, query_data: Dict) -> str:
        logging.debug(f"DEBUG: ResponseFormatter._build_query_string() - Called")
        # [Function implementation]
        """Build human-readable query string"""
        ticker = query_data.get('ticker', '').upper()
        timeframe = query_data.get('timeframe', '1d')
        query_type = query_data.get('query_type', 'pattern_detection')

        if query_type == 'pattern_detection':
            pattern = query_data.get('pattern', 'any')
            if pattern.lower() == 'any':
                return f"Analyze candlestick patterns for {ticker} on {timeframe} timeframe"
            else:
                return f"Check for {pattern} pattern in {ticker} on {timeframe} timeframe"
        elif query_type == 'ohlcv':
            return f"Get OHLCV data for {ticker} on {timeframe} timeframe"
        else:
            return f"Query about {ticker} on {timeframe} timeframe"

    def _build_analysis_summary(self, patterns: List[Dict]) -> Dict:
        logging.debug(f"DEBUG: ResponseFormatter._build_analysis_summary() - Called with {len(patterns)} patterns")
        # [Function implementation]
        """Build summary of pattern analysis"""
        if not patterns:
            return {
                "total_patterns_detected": 0,
                "primary_signal": "No patterns detected",
                "confidence_level": "Low",
                "recommendation": "Monitor price action for pattern development"
            }

        # Find pattern with highest confidence
        best_pattern = max(patterns, key=lambda p: p.get('confidence', 0))

        total_detected = len([p for p in patterns if p.get('detected', False)])

        # Determine overall confidence level
        avg_confidence = sum(p.get('confidence', 0) for p in patterns) / len(patterns)
        if avg_confidence >= 0.8:
            confidence_level = "High"
        elif avg_confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        return {
            "total_patterns_detected": total_detected,
            "primary_signal": best_pattern.get('signal', 'Unknown'),
            "best_pattern": f"{best_pattern.get('pattern', 'Unknown')} ({best_pattern.get('type', 'Common')})",
            "average_confidence": round(avg_confidence, 2),
            "confidence_level": confidence_level,
            "recommendation": self._get_recommendation(best_pattern, total_detected)
        }

    def _calculate_pattern_strength(self, confidence: float) -> str:
        logging.debug(f"DEBUG: ResponseFormatter._calculate_pattern_strength() - Called with confidence: {confidence}")
        # [Function implementation]
        """Convert confidence score to strength description"""
        if confidence >= 0.8:
            return "Strong"
        elif confidence >= 0.7:
            return "Moderate"
        elif confidence >= 0.6:
            return "Weak"
        else:
            return "Very Weak"

    def _get_recommendation(self, best_pattern: Dict, total_detected: int) -> str:
        logging.debug(f"DEBUG: ResponseFormatter._get_recommendation() - Called with total_detected: {total_detected}")
        # [Function implementation]
        """Generate trading recommendation based on patterns"""
        pattern_name = best_pattern.get('pattern', '').lower()
        confidence = best_pattern.get('confidence', 0)

        if confidence < 0.6:
            return "Pattern confidence too low - wait for confirmation"

        if pattern_name == 'doji':
            return "Exercise caution - market indecision suggests waiting for trend confirmation"
        elif pattern_name == 'hammer':
            return "Potential bullish reversal - consider entry on confirmation of uptrend"
        elif pattern_name in ('shooting star', 'shooting_star'):
            return "Potential bearish reversal - consider short on confirmation of downtrend"
        elif pattern_name == 'marubozu':
            pattern_type = best_pattern.get('type', '').lower()
            if 'bullish' in pattern_type:
                return "Strong bullish momentum - consider long position"
            else:
                return "Strong bearish momentum - consider short position"
        else:
            return "Monitor price action and volume for confirmation"

    def to_json(self, response: Dict) -> str:
        logging.debug(f"DEBUG: ResponseFormatter.to_json() - Called")
        # [Function implementation]
        """Convert response dict to JSON string"""
        return json.dumps(response, indent=2, ensure_ascii=False)

    def pretty_print(self, response: Dict) -> None:
        logging.debug(f"DEBUG: ResponseFormatter.pretty_print() - Called")
        # [Function implementation]
        """Pretty print response for debugging"""
        print(json.dumps(response, indent=2, ensure_ascii=False))


# Global instance for easy access
response_formatter = ResponseFormatter()

