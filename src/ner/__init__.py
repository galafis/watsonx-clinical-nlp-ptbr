"""Named Entity Recognition module for clinical Portuguese text."""

from src.ner.clinical_ner import ClinicalNER
from src.ner.entity_types import ClinicalEntity, EntityType
from src.ner.relation_extractor import RelationExtractor

__all__ = ["ClinicalEntity", "ClinicalNER", "EntityType", "RelationExtractor"]
