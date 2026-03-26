"""Tests for the summarization module: ClinicalSummarizer and FHIRFormatter."""

from __future__ import annotations

import pytest

from src.ner.entity_types import ClinicalEntity, EntityType
from src.ner.relation_extractor import ClinicalRelation
from src.summarization.clinical_summarizer import ClinicalSummarizer
from src.summarization.fhir_formatter import FHIRFormatter

# ---------------------------------------------------------------------------
# ClinicalSummarizer
# ---------------------------------------------------------------------------


class TestClinicalSummarizer:
    """Tests for ClinicalSummarizer (rule-based summarization, no Watsonx calls)."""

    @pytest.fixture()
    def summarizer(self) -> ClinicalSummarizer:
        return ClinicalSummarizer()

    @pytest.fixture()
    def sample_entities(self) -> list[ClinicalEntity]:
        return [
            ClinicalEntity(
                text="hipertensao",
                entity_type=EntityType.CONDICAO,
                start=0,
                end=11,
                confidence=0.80,
                source="gazetteer",
            ),
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=20,
                end=29,
                confidence=0.80,
                source="gazetteer",
            ),
            ClinicalEntity(
                text="50mg",
                entity_type=EntityType.DOSAGEM,
                start=30,
                end=34,
                confidence=0.90,
                source="pattern",
            ),
            ClinicalEntity(
                text="diabetes",
                entity_type=EntityType.CONDICAO,
                start=40,
                end=48,
                confidence=0.80,
                source="gazetteer",
            ),
            ClinicalEntity(
                text="metformina",
                entity_type=EntityType.MEDICAMENTO,
                start=55,
                end=65,
                confidence=0.80,
                source="gazetteer",
            ),
            ClinicalEntity(
                text="850mg",
                entity_type=EntityType.DOSAGEM,
                start=66,
                end=71,
                confidence=0.90,
                source="pattern",
            ),
            ClinicalEntity(
                text="hemograma",
                entity_type=EntityType.PROCEDIMENTO,
                start=80,
                end=89,
                confidence=0.80,
                source="gazetteer",
            ),
        ]

    @pytest.fixture()
    def sample_relations(self, sample_entities: list[ClinicalEntity]) -> list[ClinicalRelation]:
        return [
            ClinicalRelation(
                source=sample_entities[1],  # losartana
                target=sample_entities[2],  # 50mg
                relation_type="HAS_DOSAGE",
                confidence=0.95,
            ),
            ClinicalRelation(
                source=sample_entities[1],  # losartana
                target=sample_entities[0],  # hipertensao
                relation_type="TREATS",
                confidence=0.85,
            ),
            ClinicalRelation(
                source=sample_entities[4],  # metformina
                target=sample_entities[5],  # 850mg
                relation_type="HAS_DOSAGE",
                confidence=0.95,
            ),
            ClinicalRelation(
                source=sample_entities[4],  # metformina
                target=sample_entities[3],  # diabetes
                relation_type="TREATS",
                confidence=0.85,
            ),
        ]

    def test_summarize_empty_entities(self, summarizer: ClinicalSummarizer) -> None:
        result = summarizer.summarize_from_entities([])
        assert isinstance(result, dict)
        assert result["condicoes"] == []
        assert result["medicamentos"] == []

    def test_summarize_from_entities_structure(
        self,
        summarizer: ClinicalSummarizer,
        sample_entities: list[ClinicalEntity],
        sample_relations: list[ClinicalRelation],
    ) -> None:
        result = summarizer.summarize_from_entities(sample_entities, sample_relations)
        assert "condicoes" in result
        assert "medicamentos" in result
        assert "procedimentos" in result
        assert "valores_laboratoriais" in result

    def test_summarize_conditions(
        self,
        summarizer: ClinicalSummarizer,
        sample_entities: list[ClinicalEntity],
    ) -> None:
        result = summarizer.summarize_from_entities(sample_entities)
        conditions = [c.lower() for c in result["condicoes"]]
        assert "hipertensao" in conditions
        assert "diabetes" in conditions

    def test_summarize_medications_with_dosages(
        self,
        summarizer: ClinicalSummarizer,
        sample_entities: list[ClinicalEntity],
        sample_relations: list[ClinicalRelation],
    ) -> None:
        result = summarizer.summarize_from_entities(sample_entities, sample_relations)
        meds = result["medicamentos"]
        assert len(meds) >= 2

        med_names = [m["nome"].lower() for m in meds]
        assert "losartana" in med_names
        assert "metformina" in med_names

        # Check dosage association
        losartana_entry = next(m for m in meds if m["nome"].lower() == "losartana")
        assert "dosagem" in losartana_entry
        assert "50mg" in losartana_entry["dosagem"]

    def test_summarize_medications_with_conditions(
        self,
        summarizer: ClinicalSummarizer,
        sample_entities: list[ClinicalEntity],
        sample_relations: list[ClinicalRelation],
    ) -> None:
        result = summarizer.summarize_from_entities(sample_entities, sample_relations)
        meds = result["medicamentos"]
        losartana_entry = next(m for m in meds if m["nome"].lower() == "losartana")
        assert "indicacao" in losartana_entry
        assert "hipertensao" in losartana_entry["indicacao"].lower()

    def test_summarize_procedures(
        self,
        summarizer: ClinicalSummarizer,
        sample_entities: list[ClinicalEntity],
    ) -> None:
        result = summarizer.summarize_from_entities(sample_entities)
        procs = [p.lower() for p in result["procedimentos"]]
        assert "hemograma" in procs

    def test_summarize_deduplicates_entities(self, summarizer: ClinicalSummarizer) -> None:
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=0,
                end=9,
            ),
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=50,
                end=59,
            ),
        ]
        result = summarizer.summarize_from_entities(entities)
        # Should deduplicate
        med_names = [m["nome"].lower() for m in result["medicamentos"]]
        assert med_names.count("losartana") == 1

    def test_summarize_with_lab_values(self, summarizer: ClinicalSummarizer) -> None:
        entities = [
            ClinicalEntity(
                text="Hemoglobina: 12,5 g/dL",
                entity_type=EntityType.VALOR_LABORATORIAL,
                start=0,
                end=22,
                confidence=0.85,
                source="pattern",
            ),
        ]
        result = summarizer.summarize_from_entities(entities)
        assert len(result["valores_laboratoriais"]) >= 1


# ---------------------------------------------------------------------------
# FHIRFormatter
# ---------------------------------------------------------------------------


class TestFHIRFormatter:
    """Tests for FHIRFormatter FHIR R4 output generation."""

    @pytest.fixture()
    def formatter(self) -> FHIRFormatter:
        return FHIRFormatter()

    @pytest.fixture()
    def sample_entities(self) -> list[ClinicalEntity]:
        return [
            ClinicalEntity(
                text="hipertensao",
                entity_type=EntityType.CONDICAO,
                start=0,
                end=11,
                confidence=0.80,
            ),
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=20,
                end=29,
                confidence=0.80,
            ),
            ClinicalEntity(
                text="hemograma",
                entity_type=EntityType.PROCEDIMENTO,
                start=40,
                end=49,
                confidence=0.80,
            ),
            ClinicalEntity(
                text="Hemoglobina: 12,5 g/dL",
                entity_type=EntityType.VALOR_LABORATORIAL,
                start=60,
                end=82,
                confidence=0.85,
            ),
        ]

    def test_entities_to_bundle_structure(
        self, formatter: FHIRFormatter, sample_entities: list[ClinicalEntity]
    ) -> None:
        bundle = formatter.entities_to_bundle(sample_entities)
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "transaction"
        assert "id" in bundle
        assert "timestamp" in bundle
        assert "entry" in bundle

    def test_entities_to_bundle_entry_count(
        self, formatter: FHIRFormatter, sample_entities: list[ClinicalEntity]
    ) -> None:
        bundle = formatter.entities_to_bundle(sample_entities)
        # DOSAGEM entities are skipped, so all 4 should produce entries
        assert len(bundle["entry"]) == 4

    def test_condition_resource(self, formatter: FHIRFormatter) -> None:
        entities = [
            ClinicalEntity(
                text="hipertensao",
                entity_type=EntityType.CONDICAO,
                start=0,
                end=11,
                confidence=0.80,
            ),
        ]
        bundle = formatter.entities_to_bundle(entities)
        resource = bundle["entry"][0]["resource"]
        assert resource["resourceType"] == "Condition"
        assert resource["code"]["text"] == "hipertensao"
        assert resource["clinicalStatus"]["coding"][0]["code"] == "active"
        assert "Patient/" in resource["subject"]["reference"]

    def test_medication_statement_resource(self, formatter: FHIRFormatter) -> None:
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=0,
                end=9,
                confidence=0.80,
            ),
        ]
        bundle = formatter.entities_to_bundle(entities)
        resource = bundle["entry"][0]["resource"]
        assert resource["resourceType"] == "MedicationStatement"
        assert resource["medicationCodeableConcept"]["text"] == "losartana"
        assert resource["status"] == "active"

    def test_procedure_resource(self, formatter: FHIRFormatter) -> None:
        entities = [
            ClinicalEntity(
                text="hemograma",
                entity_type=EntityType.PROCEDIMENTO,
                start=0,
                end=9,
                confidence=0.80,
            ),
        ]
        bundle = formatter.entities_to_bundle(entities)
        resource = bundle["entry"][0]["resource"]
        assert resource["resourceType"] == "Procedure"
        assert resource["code"]["text"] == "hemograma"
        assert resource["status"] == "completed"

    def test_observation_resource(self, formatter: FHIRFormatter) -> None:
        entities = [
            ClinicalEntity(
                text="Hemoglobina: 12,5 g/dL",
                entity_type=EntityType.VALOR_LABORATORIAL,
                start=0,
                end=22,
                confidence=0.85,
            ),
        ]
        bundle = formatter.entities_to_bundle(entities)
        resource = bundle["entry"][0]["resource"]
        assert resource["resourceType"] == "Observation"
        assert resource["status"] == "final"
        assert "12,5" in resource["valueString"]

    def test_dosage_entity_skipped(self, formatter: FHIRFormatter) -> None:
        entities = [
            ClinicalEntity(
                text="50mg",
                entity_type=EntityType.DOSAGEM,
                start=0,
                end=4,
                confidence=0.90,
            ),
        ]
        bundle = formatter.entities_to_bundle(entities)
        assert len(bundle["entry"]) == 0

    def test_bundle_empty_entities(self, formatter: FHIRFormatter) -> None:
        bundle = formatter.entities_to_bundle([])
        assert bundle["resourceType"] == "Bundle"
        assert len(bundle["entry"]) == 0

    def test_custom_patient_id(self, formatter: FHIRFormatter) -> None:
        entities = [
            ClinicalEntity(
                text="hipertensao",
                entity_type=EntityType.CONDICAO,
                start=0,
                end=11,
            ),
        ]
        bundle = formatter.entities_to_bundle(entities, patient_id="patient-123")
        resource = bundle["entry"][0]["resource"]
        assert resource["subject"]["reference"] == "Patient/patient-123"

    def test_bundle_entries_have_full_url(
        self, formatter: FHIRFormatter, sample_entities: list[ClinicalEntity]
    ) -> None:
        bundle = formatter.entities_to_bundle(sample_entities)
        for entry in bundle["entry"]:
            assert "fullUrl" in entry
            assert entry["fullUrl"].startswith("http")

    def test_bundle_entries_have_request(
        self, formatter: FHIRFormatter, sample_entities: list[ClinicalEntity]
    ) -> None:
        bundle = formatter.entities_to_bundle(sample_entities)
        for entry in bundle["entry"]:
            assert entry["request"]["method"] == "POST"

    def test_summary_to_composition(self, formatter: FHIRFormatter) -> None:
        summary = {
            "condicoes": ["hipertensao", "diabetes"],
            "medicamentos": [
                {"nome": "losartana", "dosagem": "50mg"},
                {"nome": "metformina", "dosagem": "850mg"},
            ],
            "procedimentos": ["hemograma"],
            "valores_laboratoriais": ["Hemoglobina: 12,5 g/dL"],
        }
        composition = formatter.summary_to_composition(summary)
        assert composition["resourceType"] == "Composition"
        assert composition["status"] == "final"
        assert composition["type"]["coding"][0]["system"] == "http://loinc.org"
        assert len(composition["section"]) == 4  # All four sections

    def test_composition_sections_content(self, formatter: FHIRFormatter) -> None:
        summary = {
            "condicoes": ["hipertensao"],
            "medicamentos": [{"nome": "losartana", "dosagem": "50mg"}],
            "procedimentos": [],
            "valores_laboratoriais": [],
        }
        composition = formatter.summary_to_composition(summary)
        # Only sections with content should appear
        assert len(composition["section"]) == 2

    def test_composition_custom_patient(self, formatter: FHIRFormatter) -> None:
        summary = {
            "condicoes": ["febre"],
            "medicamentos": [],
            "procedimentos": [],
            "valores_laboratoriais": [],
        }
        composition = formatter.summary_to_composition(summary, patient_id="pac-999")
        assert composition["subject"]["reference"] == "Patient/pac-999"

    def test_resource_notes_contain_confidence(self, formatter: FHIRFormatter) -> None:
        entities = [
            ClinicalEntity(
                text="pneumonia",
                entity_type=EntityType.CONDICAO,
                start=0,
                end=9,
                confidence=0.82,
            ),
        ]
        bundle = formatter.entities_to_bundle(entities)
        resource = bundle["entry"][0]["resource"]
        note_text = resource["note"][0]["text"]
        assert "0.82" in note_text
