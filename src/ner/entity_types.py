"""Entity type definitions for clinical NER in Portuguese.

Defines the entity taxonomy used across the pipeline:
- MEDICAMENTO: Medications, drugs, active ingredients
- DOSAGEM: Dosage amounts, frequencies, routes of administration
- CONDICAO: Medical conditions, diseases, symptoms, diagnoses
- PROCEDIMENTO: Medical procedures, surgeries, exams, therapies
- VALOR_LABORATORIAL: Lab test results with numeric values and units
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EntityType(str, Enum):
    """Clinical entity types for Portuguese medical NER."""

    MEDICAMENTO = "MEDICAMENTO"
    DOSAGEM = "DOSAGEM"
    CONDICAO = "CONDICAO"
    PROCEDIMENTO = "PROCEDIMENTO"
    VALOR_LABORATORIAL = "VALOR_LABORATORIAL"

    @property
    def label_pt(self) -> str:
        """Return a human-readable Portuguese label."""
        labels = {
            "MEDICAMENTO": "Medicamento",
            "DOSAGEM": "Dosagem",
            "CONDICAO": "Condicao Clinica",
            "PROCEDIMENTO": "Procedimento",
            "VALOR_LABORATORIAL": "Valor Laboratorial",
        }
        return labels.get(self.value, self.value)

    @property
    def label_en(self) -> str:
        """Return a human-readable English label."""
        labels = {
            "MEDICAMENTO": "Medication",
            "DOSAGEM": "Dosage",
            "CONDICAO": "Medical Condition",
            "PROCEDIMENTO": "Procedure",
            "VALOR_LABORATORIAL": "Lab Value",
        }
        return labels.get(self.value, self.value)

    @property
    def color(self) -> str:
        """Return a hex color for UI display."""
        colors = {
            "MEDICAMENTO": "#4CAF50",
            "DOSAGEM": "#2196F3",
            "CONDICAO": "#F44336",
            "PROCEDIMENTO": "#FF9800",
            "VALOR_LABORATORIAL": "#9C27B0",
        }
        return colors.get(self.value, "#757575")


@dataclass
class ClinicalEntity:
    """A clinical entity extracted from text.

    Attributes:
        text: The original text span of the entity.
        entity_type: The classified entity type.
        start: Character offset start position in the source text.
        end: Character offset end position in the source text.
        confidence: Confidence score from 0.0 to 1.0.
        source: Extraction source (rule, pattern, model).
        metadata: Additional metadata (e.g., normalized form, FHIR code).
    """

    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    source: str = "rule"
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert entity to a serializable dictionary."""
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }

    @property
    def label(self) -> str:
        """Return a display label combining type and text."""
        return f"[{self.entity_type.value}] {self.text}"
