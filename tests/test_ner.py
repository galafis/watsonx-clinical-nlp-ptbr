"""Tests for the NER module: ClinicalNER entity extraction."""

from __future__ import annotations

import pytest

from src.ner.clinical_ner import ClinicalNER
from src.ner.entity_types import ClinicalEntity, EntityType


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_type_values(self) -> None:
        assert EntityType.MEDICAMENTO.value == "MEDICAMENTO"
        assert EntityType.DOSAGEM.value == "DOSAGEM"
        assert EntityType.CONDICAO.value == "CONDICAO"
        assert EntityType.PROCEDIMENTO.value == "PROCEDIMENTO"
        assert EntityType.VALOR_LABORATORIAL.value == "VALOR_LABORATORIAL"

    def test_entity_type_label_pt(self) -> None:
        assert EntityType.MEDICAMENTO.label_pt == "Medicamento"
        assert EntityType.CONDICAO.label_pt == "Condicao Clinica"

    def test_entity_type_label_en(self) -> None:
        assert EntityType.MEDICAMENTO.label_en == "Medication"
        assert EntityType.CONDICAO.label_en == "Medical Condition"

    def test_entity_type_color(self) -> None:
        assert EntityType.MEDICAMENTO.color == "#4CAF50"
        assert EntityType.DOSAGEM.color == "#2196F3"
        assert EntityType.CONDICAO.color == "#F44336"


class TestClinicalEntity:
    """Tests for ClinicalEntity dataclass."""

    def test_entity_creation(self) -> None:
        entity = ClinicalEntity(
            text="losartana",
            entity_type=EntityType.MEDICAMENTO,
            start=0,
            end=9,
            confidence=0.85,
            source="gazetteer",
        )
        assert entity.text == "losartana"
        assert entity.entity_type == EntityType.MEDICAMENTO
        assert entity.confidence == 0.85

    def test_entity_to_dict(self) -> None:
        entity = ClinicalEntity(
            text="hipertensao",
            entity_type=EntityType.CONDICAO,
            start=10,
            end=21,
            confidence=0.80,
            source="gazetteer",
        )
        d = entity.to_dict()
        assert d["text"] == "hipertensao"
        assert d["entity_type"] == "CONDICAO"
        assert d["start"] == 10
        assert d["end"] == 21
        assert d["confidence"] == 0.80

    def test_entity_label(self) -> None:
        entity = ClinicalEntity(
            text="losartana",
            entity_type=EntityType.MEDICAMENTO,
            start=0,
            end=9,
        )
        assert entity.label == "[MEDICAMENTO] losartana"


class TestClinicalNER:
    """Tests for ClinicalNER entity extraction."""

    @pytest.fixture()
    def ner(self) -> ClinicalNER:
        return ClinicalNER()

    def test_extract_empty_text(self, ner: ClinicalNER) -> None:
        assert ner.extract("") == []

    def test_extract_medication_losartana(self, ner: ClinicalNER) -> None:
        text = "paciente em uso de losartana diariamente"
        entities = ner.extract(text)
        med_entities = [e for e in entities if e.entity_type == EntityType.MEDICAMENTO]
        assert len(med_entities) >= 1
        assert any("losartana" in e.text.lower() for e in med_entities)

    def test_extract_medication_metformina(self, ner: ClinicalNER) -> None:
        text = "prescrito metformina para controle glicemico"
        entities = ner.extract(text)
        med_entities = [e for e in entities if e.entity_type == EntityType.MEDICAMENTO]
        assert any("metformina" in e.text.lower() for e in med_entities)

    def test_extract_multiple_medications(self, ner: ClinicalNER) -> None:
        text = "em uso de losartana, metformina e omeprazol"
        entities = ner.extract(text)
        med_entities = [e for e in entities if e.entity_type == EntityType.MEDICAMENTO]
        med_texts = [e.text.lower() for e in med_entities]
        assert "losartana" in med_texts
        assert "metformina" in med_texts
        assert "omeprazol" in med_texts

    def test_extract_dosage_mg(self, ner: ClinicalNER) -> None:
        text = "losartana 50mg via oral"
        entities = ner.extract(text)
        dosage_entities = [e for e in entities if e.entity_type == EntityType.DOSAGEM]
        assert len(dosage_entities) >= 1
        assert any("50mg" in e.text for e in dosage_entities)

    def test_extract_dosage_frequency(self, ner: ClinicalNER) -> None:
        text = "metformina 850mg 2x/dia"
        entities = ner.extract(text)
        dosage_entities = [e for e in entities if e.entity_type == EntityType.DOSAGEM]
        assert len(dosage_entities) >= 1

    def test_extract_dosage_interval(self, ner: ClinicalNER) -> None:
        text = "dipirona de 6/6h se dor"
        entities = ner.extract(text)
        dosage_entities = [e for e in entities if e.entity_type == EntityType.DOSAGEM]
        assert len(dosage_entities) >= 1

    def test_extract_dosage_route(self, ner: ClinicalNER) -> None:
        text = "administrar via oral com agua"
        entities = ner.extract(text)
        dosage_entities = [e for e in entities if e.entity_type == EntityType.DOSAGEM]
        assert any("via oral" in e.text.lower() for e in dosage_entities)

    def test_extract_condition_hipertensao(self, ner: ClinicalNER) -> None:
        text = "paciente com hipertensao arterial"
        entities = ner.extract(text)
        cond_entities = [e for e in entities if e.entity_type == EntityType.CONDICAO]
        assert any("hipertensao" in e.text.lower() for e in cond_entities)

    def test_extract_condition_diabetes(self, ner: ClinicalNER) -> None:
        text = "portador de diabetes com controle irregular"
        entities = ner.extract(text)
        cond_entities = [e for e in entities if e.entity_type == EntityType.CONDICAO]
        assert any("diabetes" in e.text.lower() for e in cond_entities)

    def test_extract_condition_pneumonia(self, ner: ClinicalNER) -> None:
        text = "diagnosticado com pneumonia comunitaria"
        entities = ner.extract(text)
        cond_entities = [e for e in entities if e.entity_type == EntityType.CONDICAO]
        assert any("pneumonia" in e.text.lower() for e in cond_entities)

    def test_extract_procedure_tomografia(self, ner: ClinicalNER) -> None:
        text = "solicitada tomografia de torax"
        entities = ner.extract(text)
        proc_entities = [e for e in entities if e.entity_type == EntityType.PROCEDIMENTO]
        assert any("tomografia" in e.text.lower() for e in proc_entities)

    def test_extract_procedure_hemograma(self, ner: ClinicalNER) -> None:
        text = "hemograma completo solicitado"
        entities = ner.extract(text)
        proc_entities = [e for e in entities if e.entity_type == EntityType.PROCEDIMENTO]
        assert any("hemograma" in e.text.lower() for e in proc_entities)

    def test_extract_lab_value_with_units(self, ner: ClinicalNER) -> None:
        text = "Hemoglobina: 12,5 g/dL"
        entities = ner.extract(text)
        lab_entities = [e for e in entities if e.entity_type == EntityType.VALOR_LABORATORIAL]
        assert len(lab_entities) >= 1

    def test_extract_lab_value_creatinina(self, ner: ClinicalNER) -> None:
        text = "Creatinina: 1,2 mg/dL"
        entities = ner.extract(text)
        lab_entities = [e for e in entities if e.entity_type == EntityType.VALOR_LABORATORIAL]
        assert len(lab_entities) >= 1

    def test_extract_full_clinical_note(self, ner: ClinicalNER) -> None:
        text = (
            "paciente com hipertensao em uso de losartana 50mg 1x/dia. "
            "diabetes controlada com metformina 850mg. "
            "hemograma solicitado. Hemoglobina: 12,5 g/dL."
        )
        entities = ner.extract(text)
        types = {e.entity_type for e in entities}
        assert EntityType.MEDICAMENTO in types
        assert EntityType.DOSAGEM in types
        assert EntityType.CONDICAO in types

    def test_extract_entities_sorted_by_position(self, ner: ClinicalNER) -> None:
        text = "losartana 50mg para hipertensao e metformina 850mg para diabetes"
        entities = ner.extract(text)
        positions = [e.start for e in entities]
        assert positions == sorted(positions)

    def test_extract_entities_have_confidence(self, ner: ClinicalNER) -> None:
        text = "losartana 50mg para hipertensao"
        entities = ner.extract(text)
        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0

    def test_extract_entities_have_source(self, ner: ClinicalNER) -> None:
        text = "losartana 50mg para hipertensao"
        entities = ner.extract(text)
        for entity in entities:
            assert entity.source in ("pattern", "gazetteer", "rule", "model")

    def test_get_entity_stats(self, ner: ClinicalNER) -> None:
        text = "losartana 50mg para hipertensao com hemograma normal"
        entities = ner.extract(text)
        stats = ner.get_entity_stats(entities)
        assert "total" in stats
        assert "by_type" in stats
        assert "avg_confidence" in stats
        assert stats["total"] == len(entities)

    def test_get_entity_stats_empty(self, ner: ClinicalNER) -> None:
        stats = ner.get_entity_stats([])
        assert stats["total"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_no_overlapping_entities(self, ner: ClinicalNER) -> None:
        text = "losartana 50mg para hipertensao e metformina 850mg para diabetes"
        entities = ner.extract(text)
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                span1 = set(range(e1.start, e1.end))
                span2 = set(range(e2.start, e2.end))
                assert not span1 & span2, f"Overlap: {e1.text!r} and {e2.text!r}"
