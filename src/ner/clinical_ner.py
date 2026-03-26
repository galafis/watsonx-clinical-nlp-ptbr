"""Rule-based and pattern-based NER for clinical entities in Portuguese.

Combines regex patterns, gazetteers, and optional spaCy backbone to
extract MEDICAMENTO, DOSAGEM, CONDICAO, PROCEDIMENTO, and
VALOR_LABORATORIAL entities from clinical text.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from src.config import settings
from src.ner.entity_types import ClinicalEntity, EntityType

logger = structlog.get_logger(__name__)


# --- Gazetteers ---
_MEDICATIONS: set[str] = {
    "losartana",
    "losartan",
    "enalapril",
    "captopril",
    "metformina",
    "insulina",
    "omeprazol",
    "pantoprazol",
    "amoxicilina",
    "azitromicina",
    "ciprofloxacino",
    "cefalexina",
    "dipirona",
    "paracetamol",
    "ibuprofeno",
    "diclofenaco",
    "prednisona",
    "prednisolona",
    "dexametasona",
    "hidrocortisona",
    "furosemida",
    "espironolactona",
    "sinvastatina",
    "atorvastatina",
    "rosuvastatina",
    "clopidogrel",
    "varfarina",
    "heparina",
    "enoxaparina",
    "metoprolol",
    "atenolol",
    "propranolol",
    "carvedilol",
    "anlodipino",
    "nifedipino",
    "diltiazem",
    "verapamil",
    "levotiroxina",
    "amiodarona",
    "digoxina",
    "clonazepam",
    "diazepam",
    "fluoxetina",
    "sertralina",
    "escitalopram",
    "risperidona",
    "haloperidol",
    "quetiapina",
    "gabapentina",
    "pregabalina",
    "tramadol",
    "morfina",
    "codeina",
    "metoclopramida",
    "ondansetrona",
    "ranitidina",
    "bromoprida",
    "domperidona",
    "salbutamol",
    "fenoterol",
    "budesonida",
    "beclometasona",
    "formoterol",
    "tiotropio",
    "metronidazol",
    "fluconazol",
    "aciclovir",
    "oseltamivir",
    "ceftriaxona",
    "meropenem",
    "vancomicina",
    "gentamicina",
    "sulfametoxazol",
    "trimetoprima",
    "clindamicina",
    "doxiciclina",
    "rifampicina",
    "isoniazida",
    "pirazinamida",
    "etambutol",
    "glibenclamida",
    "glimepirida",
    "sitagliptina",
    "empagliflozina",
    "dapagliflozina",
    "liraglutida",
    "semaglutida",
    "pioglitazona",
    "alopurinol",
    "colchicina",
    "metotrexato",
    "ciclofosfamida",
    "tacrolimo",
    "micofenolato",
    "azatioprina",
}

_CONDITIONS: set[str] = {
    "hipertensao",
    "diabetes",
    "pneumonia",
    "infarto",
    "insuficiencia",
    "arritmia",
    "fibrilacao",
    "taquicardia",
    "bradicardia",
    "anemia",
    "trombose",
    "embolia",
    "isquemia",
    "hemorragia",
    "infeccao",
    "sepse",
    "choque",
    "edema",
    "dispneia",
    "tosse",
    "febre",
    "cefaleia",
    "dor",
    "nausea",
    "vomito",
    "diarreia",
    "constipacao",
    "ictericia",
    "ascite",
    "cianose",
    "hipotensao",
    "hipoglicemia",
    "hiperglicemia",
    "hipercalemia",
    "hipocalemia",
    "acidose",
    "alcalose",
    "desidratacao",
    "obesidade",
    "desnutricao",
    "insuficiencia renal",
    "insuficiencia hepatica",
    "insuficiencia cardiaca",
    "insuficiencia respiratoria",
    "doenca pulmonar obstrutiva cronica",
    "asma",
    "bronquite",
    "tuberculose",
    "meningite",
    "encefalite",
    "acidente vascular",
    "epilepsia",
    "convulsao",
    "parkinson",
    "alzheimer",
    "demencia",
    "depressao",
    "ansiedade",
    "esquizofrenia",
    "hipotireoidismo",
    "hipertireoidismo",
    "cirrose",
    "hepatite",
    "pancreatite",
    "colecistite",
    "apendicite",
    "diverticulite",
    "gastrite",
    "ulcera",
    "neoplasia",
    "tumor",
    "cancer",
    "leucemia",
    "linfoma",
    "fratura",
    "luxacao",
    "entorse",
    "artrite",
    "artrose",
    "osteoporose",
    "lupus",
    "covid-19",
    "sars-cov-2",
    "dengue",
    "malaria",
}

_PROCEDURES: set[str] = {
    "cirurgia",
    "biopsia",
    "endoscopia",
    "colonoscopia",
    "broncoscopia",
    "laparoscopia",
    "tomografia",
    "ressonancia",
    "radiografia",
    "ultrassonografia",
    "ecografia",
    "ecocardiograma",
    "cateterismo",
    "angioplastia",
    "revascularizacao",
    "transplante",
    "dialise",
    "hemodialise",
    "intubacao",
    "traqueostomia",
    "drenagem",
    "paracentese",
    "toracocentese",
    "puncao",
    "aspiracao",
    "transfusao",
    "quimioterapia",
    "radioterapia",
    "imunoterapia",
    "fisioterapia",
    "fonoterapia",
    "nutricao enteral",
    "nutricao parenteral",
    "sutura",
    "desbridamento",
    "amputacao",
    "artroscopia",
    "artrodese",
    "mastectomia",
    "colecistectomia",
    "apendicectomia",
    "histerectomia",
    "prostatectomia",
    "nefrectomia",
    "gastrectomia",
    "colostomia",
    "ileostomia",
    "hemograma",
    "gasometria",
    "urocultura",
    "hemocultura",
    "eletrocardiograma",
    "holter",
    "mapa",
}


class ClinicalNER:
    """Clinical Named Entity Recognition engine for Portuguese text.

    Combines multiple extraction strategies:
    1. Pattern-based: Regex patterns for dosages, lab values, measurements
    2. Gazetteer-based: Dictionary lookup for medications, conditions, procedures
    3. Contextual rules: Window-based context analysis for improved precision
    """

    # Dosage patterns
    _DOSAGE_PATTERNS: list[re.Pattern[str]] = [
        # "500mg", "500 mg", "0,5 mg"
        re.compile(
            r"\b(\d+[.,]?\d*)\s*(mg|ml|mcg|g|kg|ui|meq|gotas?|gt|cp|cpr|amp|comp)\b",
            re.IGNORECASE,
        ),
        # "2x/dia", "3 vezes ao dia", "de 8/8h", "a cada 12h"
        re.compile(
            r"\b(\d+)\s*(?:x|vez(?:es)?)\s*/?\s*(?:ao\s+)?(?:dia|d)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bde\s+(\d+)\s*/\s*(\d+)\s*h(?:oras?)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\ba\s+cada\s+(\d+)\s*h(?:oras?)?\b",
            re.IGNORECASE,
        ),
        # Route of administration
        re.compile(
            r"\b(?:via\s+oral|vo|ev|im|sc|sl|intramuscular|endovenoso|subcutaneo|sublingual|topico)\b",
            re.IGNORECASE,
        ),
    ]

    # Lab value patterns
    _LAB_VALUE_PATTERNS: list[re.Pattern[str]] = [
        # "Hemoglobina: 12,5 g/dL", "Creatinina 1,2 mg/dL"
        re.compile(
            r"\b([A-Z\u00C0-\u00DC][a-z\u00E0-\u00FC]+(?:\s+[a-z\u00E0-\u00FC]+)*)"
            r"\s*[:=]\s*(\d+[.,]?\d*)\s*(mg/dL|g/dL|mmol/L|mEq/L|U/L|ng/mL|pg/mL|"
            r"mcg/dL|mm[³3]|/mm[³3]|%|mil/mm[³3]|x10[³3]/uL)\b",
            re.IGNORECASE,
        ),
        # Standalone lab values with units
        re.compile(
            r"\b(\d+[.,]?\d*)\s*(mg/dL|g/dL|mmol/L|mEq/L|U/L|ng/mL|pg/mL|"
            r"mcg/dL|/mm[³3]|mil/mm[³3]|x10[³3]/uL)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(self) -> None:
        config = settings.ner
        self._confidence_threshold = config.get("confidence_threshold", 0.65)
        self._max_entities = config.get("max_entities_per_text", 200)
        self._medications = _MEDICATIONS
        self._conditions = _CONDITIONS
        self._procedures = _PROCEDURES

    def extract(self, text: str) -> list[ClinicalEntity]:
        """Extract clinical entities from Portuguese text.

        Args:
            text: Normalized clinical text in Portuguese.

        Returns:
            List of ClinicalEntity objects sorted by position.
        """
        if not text:
            return []

        entities: list[ClinicalEntity] = []

        # Step 1: Extract dosage patterns
        entities.extend(self._extract_dosages(text))

        # Step 2: Extract lab values
        entities.extend(self._extract_lab_values(text))

        # Step 3: Extract medications (gazetteer)
        entities.extend(
            self._extract_from_gazetteer(
                text,
                self._medications,
                EntityType.MEDICAMENTO,
            )
        )

        # Step 4: Extract conditions (gazetteer)
        entities.extend(
            self._extract_from_gazetteer(
                text,
                self._conditions,
                EntityType.CONDICAO,
            )
        )

        # Step 5: Extract procedures (gazetteer)
        entities.extend(
            self._extract_from_gazetteer(
                text,
                self._procedures,
                EntityType.PROCEDIMENTO,
            )
        )

        # Remove overlapping entities (prefer longer spans)
        entities = self._resolve_overlaps(entities)

        # Apply confidence threshold filter
        entities = [e for e in entities if e.confidence >= self._confidence_threshold]

        # Limit total entities
        entities = entities[: self._max_entities]

        # Sort by position
        entities.sort(key=lambda e: (e.start, -e.end))

        logger.info("ner_extraction_complete", entity_count=len(entities))
        return entities

    def _extract_dosages(self, text: str) -> list[ClinicalEntity]:
        """Extract dosage entities using regex patterns."""
        entities: list[ClinicalEntity] = []
        for pattern in self._DOSAGE_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    ClinicalEntity(
                        text=match.group(),
                        entity_type=EntityType.DOSAGEM,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.90,
                        source="pattern",
                    )
                )
        return entities

    def _extract_lab_values(self, text: str) -> list[ClinicalEntity]:
        """Extract laboratory value entities using regex patterns."""
        entities: list[ClinicalEntity] = []
        for pattern in self._LAB_VALUE_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    ClinicalEntity(
                        text=match.group(),
                        entity_type=EntityType.VALOR_LABORATORIAL,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85,
                        source="pattern",
                    )
                )
        return entities

    def _extract_from_gazetteer(
        self,
        text: str,
        gazetteer: set[str],
        entity_type: EntityType,
    ) -> list[ClinicalEntity]:
        """Extract entities by matching against a gazetteer dictionary.

        Uses word-boundary matching to avoid partial matches.
        """
        entities: list[ClinicalEntity] = []
        text_lower = text.lower()

        for term in gazetteer:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text_lower):
                entities.append(
                    ClinicalEntity(
                        text=text[match.start() : match.end()],
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.80,
                        source="gazetteer",
                    )
                )

        return entities

    @staticmethod
    def _resolve_overlaps(entities: list[ClinicalEntity]) -> list[ClinicalEntity]:
        """Remove overlapping entities, preferring longer spans and higher confidence."""
        if not entities:
            return []

        # Sort by length descending, then confidence descending
        sorted_entities = sorted(
            entities,
            key=lambda e: (-(e.end - e.start), -e.confidence),
        )

        kept: list[ClinicalEntity] = []
        occupied: set[int] = set()

        for entity in sorted_entities:
            span = set(range(entity.start, entity.end))
            if not span & occupied:
                kept.append(entity)
                occupied |= span

        return kept

    def get_entity_stats(self, entities: list[ClinicalEntity]) -> dict[str, Any]:
        """Return statistics about extracted entities.

        Args:
            entities: List of extracted entities.

        Returns:
            Dictionary with counts per entity type and average confidence.
        """
        stats: dict[str, Any] = {
            "total": len(entities),
            "by_type": {},
            "avg_confidence": 0.0,
        }

        if not entities:
            return stats

        for etype in EntityType:
            type_entities = [e for e in entities if e.entity_type == etype]
            stats["by_type"][etype.value] = len(type_entities)

        stats["avg_confidence"] = sum(e.confidence for e in entities) / len(entities)
        return stats
