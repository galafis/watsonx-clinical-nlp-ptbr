"""Tests for the relation extraction module."""

from __future__ import annotations

import pytest

from src.ner.entity_types import ClinicalEntity, EntityType
from src.ner.relation_extractor import ClinicalRelation, RelationExtractor


class TestClinicalRelation:
    """Tests for ClinicalRelation dataclass."""

    def test_relation_creation(self) -> None:
        source = ClinicalEntity(
            text="losartana", entity_type=EntityType.MEDICAMENTO, start=0, end=9
        )
        target = ClinicalEntity(
            text="50mg", entity_type=EntityType.DOSAGEM, start=10, end=14
        )
        relation = ClinicalRelation(
            source=source,
            target=target,
            relation_type="HAS_DOSAGE",
            confidence=0.9,
            evidence="losartana 50mg",
        )
        assert relation.source.text == "losartana"
        assert relation.target.text == "50mg"
        assert relation.relation_type == "HAS_DOSAGE"

    def test_relation_to_dict(self) -> None:
        source = ClinicalEntity(
            text="losartana", entity_type=EntityType.MEDICAMENTO, start=0, end=9
        )
        target = ClinicalEntity(
            text="50mg", entity_type=EntityType.DOSAGEM, start=10, end=14
        )
        relation = ClinicalRelation(
            source=source,
            target=target,
            relation_type="HAS_DOSAGE",
            confidence=0.9,
        )
        d = relation.to_dict()
        assert d["relation_type"] == "HAS_DOSAGE"
        assert d["source"]["text"] == "losartana"
        assert d["target"]["text"] == "50mg"
        assert d["confidence"] == 0.9


class TestRelationExtractor:
    """Tests for RelationExtractor."""

    @pytest.fixture()
    def extractor(self) -> RelationExtractor:
        return RelationExtractor()

    def test_extract_empty_entities(self, extractor: RelationExtractor) -> None:
        assert extractor.extract([], "some text") == []

    def test_extract_single_entity(self, extractor: RelationExtractor) -> None:
        entities = [
            ClinicalEntity(
                text="losartana", entity_type=EntityType.MEDICAMENTO, start=0, end=9
            ),
        ]
        assert extractor.extract(entities, "losartana") == []

    def test_extract_medication_dosage_relation(self, extractor: RelationExtractor) -> None:
        text = "losartana 50mg via oral"
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=0,
                end=9,
                confidence=0.80,
                source="gazetteer",
            ),
            ClinicalEntity(
                text="50mg",
                entity_type=EntityType.DOSAGEM,
                start=10,
                end=14,
                confidence=0.90,
                source="pattern",
            ),
        ]
        relations = extractor.extract(entities, text)
        has_dosage = [r for r in relations if r.relation_type == "HAS_DOSAGE"]
        assert len(has_dosage) >= 1
        assert has_dosage[0].source.entity_type == EntityType.MEDICAMENTO
        assert has_dosage[0].target.entity_type == EntityType.DOSAGEM

    def test_extract_medication_condition_relation(
        self, extractor: RelationExtractor
    ) -> None:
        text = "losartana para hipertensao"
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=0,
                end=9,
            ),
            ClinicalEntity(
                text="hipertensao",
                entity_type=EntityType.CONDICAO,
                start=15,
                end=26,
            ),
        ]
        relations = extractor.extract(entities, text)
        treats = [r for r in relations if r.relation_type == "TREATS"]
        assert len(treats) >= 1

    def test_extract_procedure_condition_relation(
        self, extractor: RelationExtractor
    ) -> None:
        text = "tomografia indicada para pneumonia"
        entities = [
            ClinicalEntity(
                text="tomografia",
                entity_type=EntityType.PROCEDIMENTO,
                start=0,
                end=10,
            ),
            ClinicalEntity(
                text="pneumonia",
                entity_type=EntityType.CONDICAO,
                start=24,
                end=33,
            ),
        ]
        relations = extractor.extract(entities, text)
        indicated = [r for r in relations if r.relation_type == "INDICATED_BY"]
        assert len(indicated) >= 1

    def test_no_relation_for_distant_entities(self, extractor: RelationExtractor) -> None:
        text = "losartana " + ("x" * 200) + " hipertensao"
        start_cond = 10 + 200 + 1
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=0,
                end=9,
            ),
            ClinicalEntity(
                text="hipertensao",
                entity_type=EntityType.CONDICAO,
                start=start_cond,
                end=start_cond + 11,
            ),
        ]
        relations = extractor.extract(entities, text)
        assert len(relations) == 0

    def test_confidence_based_on_proximity(self, extractor: RelationExtractor) -> None:
        text = "losartana 50mg"
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=0,
                end=9,
            ),
            ClinicalEntity(
                text="50mg",
                entity_type=EntityType.DOSAGEM,
                start=10,
                end=14,
            ),
        ]
        relations = extractor.extract(entities, text)
        assert len(relations) >= 1
        # Very close entities should have high confidence
        assert relations[0].confidence >= 0.8

    def test_relation_has_evidence(self, extractor: RelationExtractor) -> None:
        text = "prescrito losartana 50mg para hipertensao"
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=10,
                end=19,
            ),
            ClinicalEntity(
                text="50mg",
                entity_type=EntityType.DOSAGEM,
                start=20,
                end=24,
            ),
        ]
        relations = extractor.extract(entities, text)
        assert len(relations) >= 1
        assert len(relations[0].evidence) > 0

    def test_multiple_relations_from_clinical_note(
        self, extractor: RelationExtractor
    ) -> None:
        text = "losartana 50mg para hipertensao, metformina 850mg para diabetes"
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=0,
                end=9,
            ),
            ClinicalEntity(
                text="50mg",
                entity_type=EntityType.DOSAGEM,
                start=10,
                end=14,
            ),
            ClinicalEntity(
                text="hipertensao",
                entity_type=EntityType.CONDICAO,
                start=20,
                end=31,
            ),
            ClinicalEntity(
                text="metformina",
                entity_type=EntityType.MEDICAMENTO,
                start=33,
                end=43,
            ),
            ClinicalEntity(
                text="850mg",
                entity_type=EntityType.DOSAGEM,
                start=44,
                end=49,
            ),
            ClinicalEntity(
                text="diabetes",
                entity_type=EntityType.CONDICAO,
                start=55,
                end=63,
            ),
        ]
        relations = extractor.extract(entities, text)
        relation_types = {r.relation_type for r in relations}
        assert "HAS_DOSAGE" in relation_types
        assert "TREATS" in relation_types

    def test_get_medication_profile(self, extractor: RelationExtractor) -> None:
        med = ClinicalEntity(
            text="losartana", entity_type=EntityType.MEDICAMENTO, start=0, end=9
        )
        dose = ClinicalEntity(
            text="50mg", entity_type=EntityType.DOSAGEM, start=10, end=14
        )
        cond = ClinicalEntity(
            text="hipertensao", entity_type=EntityType.CONDICAO, start=20, end=31
        )

        relations = [
            ClinicalRelation(
                source=med, target=dose, relation_type="HAS_DOSAGE", confidence=0.95
            ),
            ClinicalRelation(
                source=med, target=cond, relation_type="TREATS", confidence=0.85
            ),
        ]

        profiles = extractor.get_medication_profile(relations)
        assert len(profiles) >= 1
        profile = profiles[0]
        assert profile["medication"] == "losartana"
        assert "50mg" in profile["dosages"]
        assert "hipertensao" in profile["conditions"]

    def test_get_medication_profile_empty(self, extractor: RelationExtractor) -> None:
        profiles = extractor.get_medication_profile([])
        assert profiles == []

    def test_no_relation_between_same_type(self, extractor: RelationExtractor) -> None:
        text = "losartana e metformina"
        entities = [
            ClinicalEntity(
                text="losartana",
                entity_type=EntityType.MEDICAMENTO,
                start=0,
                end=9,
            ),
            ClinicalEntity(
                text="metformina",
                entity_type=EntityType.MEDICAMENTO,
                start=12,
                end=22,
            ),
        ]
        relations = extractor.extract(entities, text)
        # No relation type defined for MEDICAMENTO-MEDICAMENTO
        assert len(relations) == 0
