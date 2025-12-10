"""
Shared pattern detection service to be reused by CLI and Flask flows.
"""

from typing import List, Optional

import pandas as pd

from src.pattern_detector import PatternDetector


class PatternService:
    """Thin wrapper around PatternDetector to keep a single instantiation."""

    def __init__(self) -> None:
        self.detector = PatternDetector()

    def detect_patterns(self, candles: pd.DataFrame, pattern: Optional[str] = None) -> List[dict]:
        detected = self.detector.detect_all_patterns(candles)
        if pattern:
            detected = [p for p in detected if p.get("pattern", "").lower() == pattern.lower()]
        return detected


pattern_service = PatternService()

