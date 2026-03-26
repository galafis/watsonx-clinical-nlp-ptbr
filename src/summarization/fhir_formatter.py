"""Convert structured clinical data to FHIR R4 JSON format.

Produces FHIR-compliant resources from extracted entities and summaries,
enabling interoperability with healthcare information systems.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.config import settings
from src.ner.entity_types import ClinicalEntity, EntityType

logger = structlog.get_logger(__name__)


class FHIRFormatter:
    """Converts clinical NLP output to FHIR R4-compliant JSON resources.

    Generates the following FHIR resource types:
    - Condition: From CONDICAO entities
    - MedicationStatement: From MEDICAMENTO + DOSAGEM entities
    - Procedure: From PROCEDIMENTO entities
    - Observation: From VALOR_LABORATORIAL entities
    - Bundle: Aggregated collection of all resources
    """

    def __init__(self) -> None:
        fhir_config = settings.fhir
        self._fhir_version = fhir_config.get("version", "R4")
        self._base_url = fhir_config.get("base_url", "http://localhost:8080/fhir")
        self._default_system = fhir_config.get("default_system", "http://loinc.org")

    def entities_to_bundle(
        self,
        entities: list[ClinicalEntity],
        patient_id: str = "synthetic-patient-001",
    ) -> dict[str, Any]:
        """Convert a list of clinical entities to a FHIR Bundle resource.

        Args:
            entities: List of extracted clinical entities.
            patient_id: Synthetic patient identifier for the FHIR resources.

        Returns:
            FHIR Bundle resource as a dictionary.
        """
        entries: list[dict[str, Any]] = []

        for entity in entities:
            resource = self._entity_to_resource(entity, patient_id)
            if resource:
                resource_type = resource.get("resourceType", "Unknown")
                resource_id = resource.get("id", str(uuid.uuid4()))
                entries.append(
                    {
                        "fullUrl": f"{self._base_url}/{resource_type}/{resource_id}",
                        "resource": resource,
                        "request": {
                            "method": "POST",
                            "url": resource_type,
                        },
                    }
                )

        bundle: dict[str, Any] = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "transaction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entry": entries,
        }

        logger.info("fhir_bundle_created", entry_count=len(entries))
        return bundle

    def _entity_to_resource(
        self,
        entity: ClinicalEntity,
        patient_id: str,
    ) -> dict[str, Any] | None:
        """Convert a single entity to the appropriate FHIR resource."""
        converters = {
            EntityType.CONDICAO: self._to_condition,
            EntityType.MEDICAMENTO: self._to_medication_statement,
            EntityType.PROCEDIMENTO: self._to_procedure,
            EntityType.VALOR_LABORATORIAL: self._to_observation,
            EntityType.DOSAGEM: None,  # Handled as part of MedicationStatement
        }

        converter = converters.get(entity.entity_type)
        if converter is None:
            return None

        return converter(entity, patient_id)

    def _to_condition(
        self,
        entity: ClinicalEntity,
        patient_id: str,
    ) -> dict[str, Any]:
        """Convert a CONDICAO entity to a FHIR Condition resource."""
        return {
            "resourceType": "Condition",
            "id": str(uuid.uuid4()),
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active",
                    }
                ],
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed",
                        "display": "Confirmed",
                    }
                ],
            },
            "code": {
                "text": entity.text,
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "recordedDate": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "note": [
                {
                    "text": f"Extraido automaticamente por NLP (confianca: {entity.confidence:.2f})",
                }
            ],
        }

    def _to_medication_statement(
        self,
        entity: ClinicalEntity,
        patient_id: str,
    ) -> dict[str, Any]:
        """Convert a MEDICAMENTO entity to a FHIR MedicationStatement resource."""
        return {
            "resourceType": "MedicationStatement",
            "id": str(uuid.uuid4()),
            "status": "active",
            "medicationCodeableConcept": {
                "text": entity.text,
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "effectiveDateTime": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "note": [
                {
                    "text": f"Extraido automaticamente por NLP (confianca: {entity.confidence:.2f})",
                }
            ],
        }

    def _to_procedure(
        self,
        entity: ClinicalEntity,
        patient_id: str,
    ) -> dict[str, Any]:
        """Convert a PROCEDIMENTO entity to a FHIR Procedure resource."""
        return {
            "resourceType": "Procedure",
            "id": str(uuid.uuid4()),
            "status": "completed",
            "code": {
                "text": entity.text,
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "performedDateTime": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "note": [
                {
                    "text": f"Extraido automaticamente por NLP (confianca: {entity.confidence:.2f})",
                }
            ],
        }

    def _to_observation(
        self,
        entity: ClinicalEntity,
        patient_id: str,
    ) -> dict[str, Any]:
        """Convert a VALOR_LABORATORIAL entity to a FHIR Observation resource."""
        return {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "status": "final",
            "code": {
                "text": entity.text,
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "effectiveDateTime": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "valueString": entity.text,
            "note": [
                {
                    "text": f"Extraido automaticamente por NLP (confianca: {entity.confidence:.2f})",
                }
            ],
        }

    def summary_to_composition(
        self,
        summary: dict[str, Any],
        patient_id: str = "synthetic-patient-001",
    ) -> dict[str, Any]:
        """Convert a structured clinical summary to a FHIR Composition resource.

        Args:
            summary: Structured summary from ClinicalSummarizer.
            patient_id: Synthetic patient identifier.

        Returns:
            FHIR Composition resource as a dictionary.
        """
        sections: list[dict[str, Any]] = []

        if summary.get("condicoes"):
            sections.append(
                {
                    "title": "Condicoes Clinicas",
                    "code": {"text": "Conditions"},
                    "text": {
                        "status": "generated",
                        "div": "<div>" + ", ".join(summary["condicoes"]) + "</div>",
                    },
                }
            )

        if summary.get("medicamentos"):
            med_texts = []
            for med in summary["medicamentos"]:
                med_text = med.get("nome", "")
                if med.get("dosagem"):
                    med_text += f" - {med['dosagem']}"
                med_texts.append(med_text)
            sections.append(
                {
                    "title": "Medicamentos",
                    "code": {"text": "Medications"},
                    "text": {
                        "status": "generated",
                        "div": "<div>" + "; ".join(med_texts) + "</div>",
                    },
                }
            )

        if summary.get("procedimentos"):
            sections.append(
                {
                    "title": "Procedimentos",
                    "code": {"text": "Procedures"},
                    "text": {
                        "status": "generated",
                        "div": "<div>" + ", ".join(summary["procedimentos"]) + "</div>",
                    },
                }
            )

        if summary.get("valores_laboratoriais"):
            sections.append(
                {
                    "title": "Exames Laboratoriais",
                    "code": {"text": "Lab Results"},
                    "text": {
                        "status": "generated",
                        "div": "<div>" + "; ".join(summary["valores_laboratoriais"]) + "</div>",
                    },
                }
            )

        composition: dict[str, Any] = {
            "resourceType": "Composition",
            "id": str(uuid.uuid4()),
            "status": "final",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11488-4",
                        "display": "Consultation Note",
                    }
                ],
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "date": datetime.now(timezone.utc).isoformat(),
            "title": "Resumo Clinico Gerado por NLP",
            "section": sections,
        }

        logger.info("fhir_composition_created", sections=len(sections))
        return composition
