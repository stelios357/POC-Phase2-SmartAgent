# Candlestick Pattern Detection Module
# Implements detection for 5 single candlestick patterns with Trend Context

import math
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
import pandas as pd
from config.settings import PATTERN_CONFIG

class PatternDetector:
    """Detects candlestick patterns from OHLCV data with trend context"""

    def __init__(self):
        logging.debug("DEBUG: PatternDetector.__init__() - Initializing pattern detector")
        self.config = PATTERN_CONFIG
        logging.debug("DEBUG: PatternDetector.__init__() - Pattern detector initialized with configuration")

    def detect_all_patterns(self, candles: Union[Dict, List[Dict], pd.DataFrame]) -> List[Dict]:
        """
        Detect all patterns in the latest candle, considering historical context if available.

        Args:
            candles: Can be:
                - Dict: Single candle (no trend context possible)
                - List[Dict]: List of candles (last one is current)
                - pd.DataFrame: DataFrame of candles (last row is current)

        Returns:
            List of detected patterns with confidence scores
        """
        logging.info("Pattern detection started")

        # Normalize input to current candle and trend context
        current_candle, trend_context = self._normalize_input(candles)

        if not current_candle:
            logging.warning("Pattern detection failed: No valid candle data found")
            return []

        patterns = []
        logging.info("Starting pattern detection process")

        # Determine prior trend
        trend = self._detect_trend(trend_context) if trend_context else "unknown"
        logging.info(f"Prior trend detected: {trend}")

        # Detect Doji patterns
        logging.info("Detecting Doji patterns")
        doji_result = self.detect_doji(current_candle, trend)
        if doji_result:
            patterns.append(doji_result)
            logging.info(f"Doji pattern detected: {doji_result['type']} (confidence: {doji_result['confidence']})")

        # Detect Hammer
        logging.info("Detecting Hammer pattern")
        hammer_result = self.detect_hammer(current_candle, trend)
        if hammer_result:
            patterns.append(hammer_result)
            logging.info(f"Hammer pattern detected: {hammer_result['pattern']} (confidence: {hammer_result['confidence']})")

        # Detect Shooting Star
        logging.info("Detecting Shooting Star pattern")
        shooting_star_result = self.detect_shooting_star(current_candle, trend)
        if shooting_star_result:
            patterns.append(shooting_star_result)
            logging.info(f"Shooting Star pattern detected: {shooting_star_result['pattern']} (confidence: {shooting_star_result['confidence']})")

        # Detect Marubozu
        logging.info("Detecting Marubozu pattern")
        marubozu_result = self.detect_marubozu(current_candle, trend)
        if marubozu_result:
            patterns.append(marubozu_result)
            logging.info(f"Marubozu pattern detected: {marubozu_result['type']} (confidence: {marubozu_result['confidence']})")

        logging.info(f"Pattern detection completed, found {len(patterns)} patterns")
        return patterns

    def _normalize_input(self, candles: Union[Dict, List[Dict], pd.DataFrame]) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """Normalize input to (current_candle, historical_candles_list)"""
        if isinstance(candles, dict):
            # Single candle
            return candles, None
        
        elif isinstance(candles, list):
            if not candles:
                return None, None
            return candles[-1], candles[:-1]
            
        elif isinstance(candles, pd.DataFrame):
            if candles.empty:
                return None, None
            # Convert last row to dict
            current = candles.iloc[-1].to_dict()
            # Normalize keys to lowercase for internal use if needed, 
            # but yfinance returns Title Case (Open, High...). 
            # Let's standardize to lowercase keys for safety.
            current = {k.lower(): v for k, v in current.items()}
            
            # Convert history to list of dicts
            history = candles.iloc[:-1].to_dict('records')
            history = [{k.lower(): v for k, v in rec.items()} for rec in history]
            return current, history
            
        return None, None

    def _detect_trend(self, history: List[Dict]) -> str:
        """
        Detect trend from history.
        Simple logic: Compare moving average or last few closes.
        """
        if not history or len(history) < 3:
            return "neutral"
            
        # Use last N candles from config
        lookback = self.config.get("trend_lookback", 5)
        recent = history[-lookback:]
        
        closes = [float(c.get('close', 0)) for c in recent]
        
        if len(closes) < 2:
            return "neutral"
            
        # Simple Check: Is the start higher/lower than end?
        if closes[-1] > closes[0]:
            return "uptrend"
        elif closes[-1] < closes[0]:
            return "downtrend"
            
        return "neutral"

    def detect_doji(self, candle: Dict, trend: str = "unknown") -> Optional[Dict]:
        """Detect Doji pattern and its variants"""
        try:
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])

            # Calculate basic metrics
            body_size = abs(close_price - open_price)
            price_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price

            logging.info(f"Doji detection - O:{open_price:.4f} H:{high_price:.4f} L:{low_price:.4f} C:{close_price:.4f}")
            logging.info(f"Doji metrics - Body:{body_size:.4f} Range:{price_range:.4f} UpperShadow:{upper_shadow:.4f} LowerShadow:{lower_shadow:.4f}")

            if price_range <= 0:
                logging.info("Doji detection failed: Invalid price range")
                return None

            doji_threshold = (self.config["doji"]["body_threshold_pct"] / 100.0) * price_range
            logging.info(f"Doji threshold: {doji_threshold:.4f} (body_size: {body_size:.4f})")

            if body_size >= doji_threshold:
                logging.info(f"Doji detection failed: Body size {body_size:.4f} >= threshold {doji_threshold:.4f}")
                return None

            # Determine Doji variant
            pattern_type = "Common"
            confidence = 0.6
            cfg = self.config["doji"]

            if (lower_shadow >= cfg["dragonfly_lower_shadow_multiplier"] * body_size and 
                upper_shadow <= 0.1 * price_range):
                pattern_type = "Dragonfly"
                confidence = 0.75
            elif (upper_shadow >= cfg["gravestone_upper_shadow_multiplier"] * body_size and 
                  lower_shadow <= 0.1 * price_range):
                pattern_type = "Gravestone"
                confidence = 0.75
            elif (upper_shadow >= cfg["long_legged_shadow_multiplier"] * body_size and 
                  lower_shadow >= cfg["long_legged_shadow_multiplier"] * body_size):
                pattern_type = "Long-Legged"
                confidence = 0.8

            # Trend Context Adjustment
            if trend != "unknown":
                # Dragonfly is more significant in downtrend
                if pattern_type == "Dragonfly" and trend == "downtrend":
                    confidence += 0.1
                # Gravestone is more significant in uptrend
                elif pattern_type == "Gravestone" and trend == "uptrend":
                    confidence += 0.1

            return {
                "pattern": "doji",
                "type": pattern_type,
                "detected": True,
                "confidence": min(confidence, 1.0),
                "signal": self._get_doji_signal(pattern_type),
                "description": self._get_doji_description(pattern_type),
                "trend_context": trend,
                "metrics": {
                    "body_size": round(body_size, 4),
                    "range": round(price_range, 4)
                }
            }

        except (KeyError, ValueError, TypeError):
            return None

    def detect_hammer(self, candle: Dict, trend: str = "unknown") -> Optional[Dict]:
        """Detect Hammer pattern (Requires Downtrend)"""
        try:
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])

            # Must be bullish candle (strictly speaking, Hammer can be bearish body but bullish is better)
            # Standard definition: Small body at top, long lower shadow. Color matters less but green is stronger.
            
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            price_range = high_price - low_price

            if price_range <= 0:
                return None
            
            # Avoid division by zero
            safe_body = body_size if body_size > 0 else 0.0001

            cfg = self.config["hammer"]
            conditions = [
                lower_shadow >= cfg["lower_shadow_multiplier"] * safe_body,
                upper_shadow <= (cfg["upper_shadow_max_pct"] / 100.0) * price_range, # Revised logic: shadow vs range or body
                # Alternatively use body multiplier for upper shadow:
                # upper_shadow <= 0.5 * safe_body (strict) or allow small shadow
            ]

            if not all(conditions):
                return None
                
            # Trend Check: Hammer is a reversal pattern, needs prior downtrend
            if trend == "uptrend":
                # Called "Hanging Man" if in uptrend, identical shape
                pattern_name = "hammer"
                signal = "Bearish reversal (top of trend)"
                pattern_type = "Bearish"
            elif trend == "downtrend":
                pattern_name = "hammer"
                signal = "Bullish reversal (bottom of trend)"
                pattern_type = "Bullish"
            else:
                # Without trend, it's just a "Hammer-like candle"
                pattern_name = "hammer"
                signal = "Potential reversal"
                pattern_type = "Neutral"

            confidence = 0.7
            if trend == "downtrend":
                confidence = 0.85
            elif trend == "uptrend":
                confidence = 0.8

            return {
                "pattern": pattern_name,
                "type": pattern_type,
                "detected": True,
                "confidence": confidence,
                "signal": signal,
                "description": f"Small body with long lower shadow. Context: {trend}",
                "metrics": {
                    "body_size": round(body_size, 4),
                    "lower_shadow": round(lower_shadow, 4)
                }
            }

        except (KeyError, ValueError, TypeError):
            return None

    def detect_shooting_star(self, candle: Dict, trend: str = "unknown") -> Optional[Dict]:
        """Detect Shooting Star pattern (Requires Uptrend)"""
        try:
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])

            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            price_range = high_price - low_price

            if price_range <= 0:
                return None

            safe_body = body_size if body_size > 0 else 0.0001
            cfg = self.config["shooting_star"]
            
            conditions = [
                upper_shadow >= cfg["upper_shadow_multiplier"] * safe_body,
                lower_shadow <= (cfg["lower_shadow_max_pct"] / 100.0) * price_range
            ]

            if not all(conditions):
                return None

            # Trend Check
            if trend == "uptrend":
                pattern_name = "shooting_star"
                signal = "Bearish reversal (top of trend)"
                pattern_type = "Bearish"
            elif trend == "downtrend":
                # Inverted Hammer
                pattern_name = "shooting_star"
                signal = "Bullish reversal (bottom of trend)"
                pattern_type = "Bullish"
            else:
                pattern_name = "shooting_star"
                signal = "Potential reversal"
                pattern_type = "Neutral"

            confidence = 0.7
            if trend == "uptrend":
                confidence = 0.85

            return {
                "pattern": pattern_name,
                "type": pattern_type,
                "detected": True,
                "confidence": confidence,
                "signal": signal,
                "description": f"Small body with long upper shadow. Context: {trend}",
                "metrics": {
                    "body_size": round(body_size, 4),
                    "upper_shadow": round(upper_shadow, 4)
                }
            }

        except (KeyError, ValueError, TypeError):
            return None

    def detect_marubozu(self, candle: Dict, trend: str = "unknown") -> Optional[Dict]:
        """Detect Marubozu pattern"""
        try:
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])

            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            body_size = abs(close_price - open_price)
            price_range = high_price - low_price

            if price_range <= 0:
                return None

            cfg = self.config["marubozu"]
            conditions = [
                upper_shadow <= (cfg["shadow_max_pct"] / 100.0) * price_range,
                lower_shadow <= (cfg["shadow_max_pct"] / 100.0) * price_range,
                body_size > (cfg["min_body_pct"] / 100.0) * price_range
            ]

            if not all(conditions):
                return None

            if close_price > open_price:
                pattern_type = "Bullish"
                signal = "Continuation"
            else:
                pattern_type = "Bearish"
                signal = "Continuation"

            return {
                "pattern": "marubozu",
                "type": pattern_type,
                "detected": True,
                "confidence": 0.9,
                "signal": f"Strong {pattern_type.lower()} momentum",
                "description": f"Pure {pattern_type.lower()} candle with minimal shadows",
                "metrics": {
                    "body_size": round(body_size, 4),
                    "range": round(price_range, 4)
                }
            }

        except (KeyError, ValueError, TypeError):
            return None

    def _get_doji_signal(self, pattern_type: str) -> str:
        signals = {
            "Common": "Market indecision",
            "Dragonfly": "Potential bullish reversal",
            "Gravestone": "Potential bearish reversal",
            "Long-Legged": "High volatility indecision"
        }
        return signals.get(pattern_type, "Indecision")

    def _get_doji_description(self, pattern_type: str) -> str:
        descriptions = {
            "Common": "Open approx equals Close",
            "Dragonfly": "Long lower shadow, Open/Close at High",
            "Gravestone": "Long upper shadow, Open/Close at Low",
            "Long-Legged": "Long upper and lower shadows"
        }
        return descriptions.get(pattern_type, "Doji pattern")

pattern_detector = PatternDetector()
