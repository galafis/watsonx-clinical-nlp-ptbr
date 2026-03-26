"""Clinical text tokenizer that respects medical terms and measurements.

Provides sentence and token-level segmentation optimized for Brazilian
Portuguese clinical documents, preserving dosage patterns, lab values,
and multi-word medical terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ClinicalToken:
    """A single token extracted from clinical text."""

    text: str
    start: int
    end: int
    token_type: str = "word"  # word, number, measurement, punctuation, whitespace


@dataclass
class ClinicalSentence:
    """A sentence extracted from clinical text with its tokens."""

    text: str
    start: int
    end: int
    tokens: list[ClinicalToken] = field(default_factory=list)


class ClinicalTokenizer:
    """Tokenizer for clinical Portuguese text.

    Handles special patterns common in clinical documents:
    - Dosage patterns (e.g., "500mg 2x/dia")
    - Lab values (e.g., "3,5 mg/dL")
    - Medical abbreviations (e.g., "HAS", "DM2")
    - Date and time patterns
    - Vital signs (e.g., "PA: 120x80 mmHg")
    """

    # Sentence boundary patterns
    _SENTENCE_SPLIT = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u00DC])"  # Period + space + uppercase
        r"|(?<=\n)\s*(?=\S)"  # Newline boundary
        r"|(?<=:)\s*\n"  # Colon + newline (common in clinical notes)
    )

    # Token patterns (order matters: most specific first)
    _TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        (
            "measurement",
            re.compile(
                r"\b\d+[.,]?\d*\s*(?:mg|ml|mcg|g|kg|mm|cm|mmHg|bpm|UI|mEq|mmol|"
                r"L|dL|mL|ug|ng|pg|U|%|x10[³3]|/mm[³3])\b",
                re.IGNORECASE,
            ),
        ),
        (
            "measurement",
            re.compile(
                r"\b\d+[.,]?\d*\s*/\s*(?:mm[³3]|dL|L|mL)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "measurement",
            re.compile(
                r"\b\d+x\d+\s*mmHg\b",
                re.IGNORECASE,
            ),
        ),
        ("number", re.compile(r"\b\d+[.,]\d+\b")),
        ("number", re.compile(r"\b\d+\b")),
        ("word", re.compile(r"\b[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F'-]*\b")),
        ("punctuation", re.compile(r"[^\s\w]")),
    ]

    def tokenize(self, text: str) -> list[ClinicalToken]:
        """Tokenize clinical text into a list of tokens.

        Args:
            text: Clinical text to tokenize.

        Returns:
            List of ClinicalToken objects with positions.
        """
        if not text:
            return []

        tokens: list[ClinicalToken] = []
        pos = 0

        while pos < len(text):
            # Skip whitespace
            if text[pos].isspace():
                pos += 1
                continue

            matched = False
            for token_type, pattern in self._TOKEN_PATTERNS:
                match = pattern.match(text, pos)
                if match:
                    tokens.append(
                        ClinicalToken(
                            text=match.group(),
                            start=match.start(),
                            end=match.end(),
                            token_type=token_type,
                        )
                    )
                    pos = match.end()
                    matched = True
                    break

            if not matched:
                # Single character fallback
                tokens.append(
                    ClinicalToken(
                        text=text[pos],
                        start=pos,
                        end=pos + 1,
                        token_type="unknown",
                    )
                )
                pos += 1

        logger.debug("text_tokenized", token_count=len(tokens))
        return tokens

    def sentence_split(self, text: str) -> list[ClinicalSentence]:
        """Split clinical text into sentences.

        Uses clinical-aware sentence boundary detection that handles
        abbreviations, dosage instructions, and structured note formats.

        Args:
            text: Clinical text to split.

        Returns:
            List of ClinicalSentence objects.
        """
        if not text:
            return []

        # Split on sentence boundaries
        parts = self._SENTENCE_SPLIT.split(text)
        sentences: list[ClinicalSentence] = []
        offset = 0

        for part in parts:
            stripped = part.strip()
            if not stripped:
                offset += len(part)
                continue

            start = text.find(stripped, offset)
            if start == -1:
                start = offset
            end = start + len(stripped)

            tokens = self.tokenize(stripped)

            sentences.append(
                ClinicalSentence(
                    text=stripped,
                    start=start,
                    end=end,
                    tokens=tokens,
                )
            )
            offset = end

        logger.debug("sentences_split", sentence_count=len(sentences))
        return sentences
