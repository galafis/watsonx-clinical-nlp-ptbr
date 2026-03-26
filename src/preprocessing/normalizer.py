"""Text normalization for clinical Portuguese documents.

Handles lowercase conversion, whitespace normalization, special character
cleanup, and optional accent handling for Brazilian Portuguese medical texts.
"""

from __future__ import annotations

import re
import unicodedata

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class ClinicalTextNormalizer:
    """Normalizes raw clinical text for downstream NLP processing.

    Applies a configurable sequence of text transformations including
    lowercasing, whitespace normalization, special character removal,
    and optional accent handling suitable for PT-BR clinical texts.
    """

    # Patterns to preserve in clinical text (measurements, dosages, codes)
    _MEASUREMENT_PATTERN = re.compile(
        r"\b\d+[.,]?\d*\s*(?:mg|ml|mcg|g|kg|mm|cm|mmHg|bpm|UI|mEq|mmol|L|dL|"
        r"mL|ug|ng|pg|U|%|x10[³3]|/mm[³3])\b",
        re.IGNORECASE,
    )

    # Numeric values with units (e.g., "3,5 mg/dL")
    _LAB_VALUE_PATTERN = re.compile(
        r"\b\d+[.,]?\d*\s*/\s*(?:mm[³3]|dL|L|mL)\b",
        re.IGNORECASE,
    )

    # Date patterns (DD/MM/YYYY)
    _DATE_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

    # Time patterns (HH:MM)
    _TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

    def __init__(self) -> None:
        config = settings.preprocessing
        self._lowercase = config.get("lowercase", True)
        self._normalize_accents = config.get("normalize_accents", False)
        self._remove_extra_whitespace = config.get("remove_extra_whitespace", True)

    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline to clinical text.

        Args:
            text: Raw clinical text in Portuguese.

        Returns:
            Normalized text ready for NER and downstream processing.
        """
        if not text or not text.strip():
            return ""

        result = text

        # Preserve measurement tokens before normalization
        preserved: dict[str, str] = {}
        result = self._preserve_patterns(result, preserved)

        # Step 1: Normalize Unicode characters (NFC form)
        result = unicodedata.normalize("NFC", result)

        # Step 2: Replace common clinical symbols
        result = self._normalize_symbols(result)

        # Step 3: Remove control characters (keep newlines and tabs)
        result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", result)

        # Step 4: Normalize whitespace
        if self._remove_extra_whitespace:
            result = re.sub(r"[ \t]+", " ", result)
            result = re.sub(r"\n{3,}", "\n\n", result)
            result = result.strip()

        # Step 5: Lowercase (preserving preserved tokens)
        if self._lowercase:
            result = result.lower()

        # Step 6: Optional accent removal
        if self._normalize_accents:
            result = self._remove_accents(result)

        # Restore preserved tokens
        result = self._restore_patterns(result, preserved)

        logger.debug("text_normalized", original_len=len(text), normalized_len=len(result))
        return result

    def _preserve_patterns(self, text: str, preserved: dict[str, str]) -> str:
        """Replace patterns that should be preserved with placeholders."""
        counter = 0
        for pattern in [
            self._MEASUREMENT_PATTERN,
            self._LAB_VALUE_PATTERN,
            self._DATE_PATTERN,
            self._TIME_PATTERN,
        ]:
            for match in pattern.finditer(text):
                placeholder = f"__PRESERVED_{counter}__"
                preserved[placeholder] = match.group()
                text = text.replace(match.group(), placeholder, 1)
                counter += 1
        return text

    @staticmethod
    def _restore_patterns(text: str, preserved: dict[str, str]) -> str:
        """Restore preserved patterns from placeholders."""
        for placeholder, original in preserved.items():
            text = text.replace(placeholder.lower(), original)
            text = text.replace(placeholder, original)
        return text

    @staticmethod
    def _normalize_symbols(text: str) -> str:
        """Normalize common clinical symbols and abbreviations."""
        replacements = {
            "\u2013": "-",  # en-dash
            "\u2014": "-",  # em-dash
            "\u2018": "'",  # left single quote
            "\u2019": "'",  # right single quote
            "\u201c": '"',  # left double quote
            "\u201d": '"',  # right double quote
            "\u2022": "-",  # bullet
            "\u00b0": " graus ",  # degree symbol
            "\u00ba": "o",  # masculine ordinal
            "\u00aa": "a",  # feminine ordinal
            "\u00b2": "2",  # superscript 2
            "\u00b3": "3",  # superscript 3
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _remove_accents(text: str) -> str:
        """Remove accents while preserving base characters.

        Note: Use with caution in clinical contexts, as accent removal
        may change medical terminology meaning in Portuguese.
        """
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))
