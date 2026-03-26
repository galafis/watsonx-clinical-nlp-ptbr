"""Extract relations between clinical entities.

Identifies semantic relationships between extracted entities such as
medication-dosage, medication-condition, and procedure-condition links
using proximity-based heuristics and syntactic patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from src.ner.entity_types import ClinicalEntity, EntityType

logger = structlog.get_logger(__name__)


@dataclass
class ClinicalRelation:
    """A relation between two clinical entities.

    Attributes:
        source: The source entity of the relation.
        target: The target entity of the relation.
        relation_type: The type of relation (e.g., TREATS, HAS_DOSAGE).
        confidence: Confidence score for the relation.
        evidence: The text span providing evidence for the relation.
    """

    source: ClinicalEntity
    target: ClinicalEntity
    relation_type: str
    confidence: float = 0.7
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert relation to a serializable dictionary."""
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


class RelationExtractor:
    """Extracts relations between clinical entities using proximity and patterns.

    Supported relation types:
    - HAS_DOSAGE: Links a MEDICAMENTO to its DOSAGEM
    - TREATS: Links a MEDICAMENTO to a CONDICAO it treats
    - INDICATED_BY: Links a PROCEDIMENTO to a CONDICAO
    - HAS_RESULT: Links a VALOR_LABORATORIAL to its context
    """

    # Maximum character distance to consider entities related
    MAX_PROXIMITY_CHARS = 150

    # Relation type definitions
    RELATION_TYPES = {
        ("MEDICAMENTO", "DOSAGEM"): "HAS_DOSAGE",
        ("DOSAGEM", "MEDICAMENTO"): "HAS_DOSAGE",
        ("MEDICAMENTO", "CONDICAO"): "TREATS",
        ("CONDICAO", "MEDICAMENTO"): "TREATS",
        ("PROCEDIMENTO", "CONDICAO"): "INDICATED_BY",
        ("CONDICAO", "PROCEDIMENTO"): "INDICATED_BY",
        ("VALOR_LABORATORIAL", "CONDICAO"): "HAS_RESULT",
    }

    def extract(
        self,
        entities: list[ClinicalEntity],
        text: str,
    ) -> list[ClinicalRelation]:
        """Extract relations between entities based on proximity and context.

        Args:
            entities: List of extracted clinical entities.
            text: The original clinical text.

        Returns:
            List of ClinicalRelation objects.
        """
        if len(entities) < 2:
            return []

        relations: list[ClinicalRelation] = []

        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda e: e.start)

        for i, source in enumerate(sorted_entities):
            for target in sorted_entities[i + 1 :]:
                # Check if pair has a defined relation type
                pair_key = (source.entity_type.value, target.entity_type.value)
                relation_type = self.RELATION_TYPES.get(pair_key)

                if not relation_type:
                    continue

                # Check proximity
                distance = target.start - source.end
                if distance > self.MAX_PROXIMITY_CHARS:
                    break  # Sorted by position, no closer matches possible

                if distance < 0:
                    continue  # Overlapping entities

                # Calculate confidence based on proximity
                proximity_score = max(0.5, 1.0 - (distance / self.MAX_PROXIMITY_CHARS))

                # Extract evidence text (span between entities with context)
                evidence_start = max(0, source.start - 20)
                evidence_end = min(len(text), target.end + 20)
                evidence = text[evidence_start:evidence_end].strip()

                # Boost confidence for medication-dosage pairs (very common)
                confidence = proximity_score
                if relation_type == "HAS_DOSAGE":
                    confidence = min(1.0, confidence + 0.15)

                relations.append(
                    ClinicalRelation(
                        source=source if pair_key[0] != "DOSAGEM" else target,
                        target=target if pair_key[0] != "DOSAGEM" else source,
                        relation_type=relation_type,
                        confidence=round(confidence, 3),
                        evidence=evidence,
                    )
                )

        logger.info("relations_extracted", count=len(relations))
        return relations

    def get_medication_profile(
        self,
        relations: list[ClinicalRelation],
    ) -> list[dict[str, Any]]:
        """Build a medication profile from extracted relations.

        Groups medications with their dosages and associated conditions.

        Args:
            relations: List of extracted clinical relations.

        Returns:
            List of medication profile dictionaries.
        """
        medication_map: dict[str, dict[str, Any]] = {}

        for relation in relations:
            if relation.relation_type == "HAS_DOSAGE":
                med_text = relation.source.text
                if med_text not in medication_map:
                    medication_map[med_text] = {
                        "medication": med_text,
                        "dosages": [],
                        "conditions": [],
                    }
                medication_map[med_text]["dosages"].append(relation.target.text)

            elif relation.relation_type == "TREATS":
                med_entity = (
                    relation.source
                    if relation.source.entity_type == EntityType.MEDICAMENTO
                    else relation.target
                )
                cond_entity = (
                    relation.target
                    if relation.target.entity_type == EntityType.CONDICAO
                    else relation.source
                )
                med_text = med_entity.text
                if med_text not in medication_map:
                    medication_map[med_text] = {
                        "medication": med_text,
                        "dosages": [],
                        "conditions": [],
                    }
                medication_map[med_text]["conditions"].append(cond_entity.text)

        return list(medication_map.values())
