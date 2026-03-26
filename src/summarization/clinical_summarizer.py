"""Clinical summarization using IBM Watsonx Granite models.

Generates structured clinical summaries from extracted entities and
relations, organizing information by medical categories.
"""

from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.ner.entity_types import ClinicalEntity, EntityType
from src.ner.relation_extractor import ClinicalRelation

logger = structlog.get_logger(__name__)


class ClinicalSummarizer:
    """Generates structured clinical summaries using Watsonx Granite.

    Takes extracted entities and relations to produce organized clinical
    summaries suitable for medical record documentation.
    """

    def __init__(self) -> None:
        self._model_id = settings.watsonx.generation_model
        self._params = settings.generation_params
        self._system_prompt = settings.summarization.get("system_prompt", "")
        self._max_length = settings.summarization.get("max_summary_length", 1500)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the Watsonx AI client."""
        if self._client is None:
            try:
                from ibm_watsonx_ai.foundation_models import Model
                from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

                params = {
                    GenTextParamsMetaNames.MAX_NEW_TOKENS: self._params.get("max_new_tokens", 2048),
                    GenTextParamsMetaNames.TEMPERATURE: self._params.get("temperature", 0.1),
                    GenTextParamsMetaNames.TOP_P: self._params.get("top_p", 0.9),
                    GenTextParamsMetaNames.TOP_K: self._params.get("top_k", 40),
                    GenTextParamsMetaNames.REPETITION_PENALTY: self._params.get(
                        "repetition_penalty", 1.1
                    ),
                }

                self._client = Model(
                    model_id=self._model_id,
                    credentials={
                        "apikey": settings.watsonx.api_key,
                        "url": settings.watsonx.url,
                    },
                    project_id=settings.watsonx.project_id,
                    params=params,
                )
                logger.info("watsonx_client_initialized", model=self._model_id)
            except Exception as e:
                logger.error("watsonx_client_init_failed", error=str(e))
                raise
        return self._client

    def summarize_from_entities(
        self,
        entities: list[ClinicalEntity],
        relations: list[ClinicalRelation] | None = None,
        original_text: str = "",
    ) -> dict[str, Any]:
        """Generate a structured clinical summary from extracted entities.

        Args:
            entities: List of extracted clinical entities.
            relations: Optional list of entity relations.
            original_text: Original clinical text for context.

        Returns:
            Dictionary with structured summary sections.
        """
        # Build structured entity groups
        grouped = self._group_entities(entities)
        medication_profiles = self._build_medication_section(entities, relations or [])

        # Build the summary without calling Watsonx (rule-based fallback)
        summary = self._build_structured_summary(grouped, medication_profiles)

        logger.info(
            "summary_generated",
            entity_count=len(entities),
            sections=list(summary.keys()),
        )
        return summary

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def summarize_with_granite(
        self,
        entities: list[ClinicalEntity],
        relations: list[ClinicalRelation] | None = None,
        original_text: str = "",
    ) -> str:
        """Generate a natural language clinical summary using Granite.

        Args:
            entities: List of extracted clinical entities.
            relations: Optional list of entity relations.
            original_text: Original clinical text for context.

        Returns:
            Natural language clinical summary string.
        """
        # Build context from entities
        entity_context = self._format_entities_for_prompt(entities, relations or [])

        prompt = (
            f"{self._system_prompt}\n\n"
            f"Entidades clinicas extraidas:\n{entity_context}\n\n"
        )

        if original_text:
            # Include a truncated version of the original text
            truncated = original_text[:2000] if len(original_text) > 2000 else original_text
            prompt += f"Texto clinico original:\n{truncated}\n\n"

        prompt += (
            "Gere um resumo clinico estruturado organizado em:\n"
            "1. Condicoes/Diagnosticos\n"
            "2. Medicamentos e Dosagens\n"
            "3. Procedimentos Realizados\n"
            "4. Resultados Laboratoriais\n"
            "5. Observacoes Clinicas\n"
        )

        client = self._get_client()
        response = client.generate_text(prompt=prompt)

        logger.info("granite_summary_generated", response_length=len(response))
        return response

    def _group_entities(
        self, entities: list[ClinicalEntity],
    ) -> dict[str, list[str]]:
        """Group entities by type into categorized lists."""
        groups: dict[str, list[str]] = {
            EntityType.CONDICAO.value: [],
            EntityType.MEDICAMENTO.value: [],
            EntityType.DOSAGEM.value: [],
            EntityType.PROCEDIMENTO.value: [],
            EntityType.VALOR_LABORATORIAL.value: [],
        }

        seen: dict[str, set[str]] = {k: set() for k in groups}

        for entity in entities:
            key = entity.entity_type.value
            normalized = entity.text.strip().lower()
            if normalized not in seen[key]:
                groups[key].append(entity.text)
                seen[key].add(normalized)

        return groups

    def _build_medication_section(
        self,
        entities: list[ClinicalEntity],
        relations: list[ClinicalRelation],
    ) -> list[dict[str, Any]]:
        """Build medication profiles linking drugs to dosages and conditions."""
        profiles: dict[str, dict[str, Any]] = {}

        # From relations
        for rel in relations:
            if rel.relation_type == "HAS_DOSAGE":
                med = rel.source.text.lower()
                if med not in profiles:
                    profiles[med] = {"medication": rel.source.text, "dosages": [], "conditions": []}
                profiles[med]["dosages"].append(rel.target.text)
            elif rel.relation_type == "TREATS":
                med_entity = (
                    rel.source
                    if rel.source.entity_type == EntityType.MEDICAMENTO
                    else rel.target
                )
                cond_entity = (
                    rel.target
                    if rel.target.entity_type == EntityType.CONDICAO
                    else rel.source
                )
                med = med_entity.text.lower()
                if med not in profiles:
                    profiles[med] = {"medication": med_entity.text, "dosages": [], "conditions": []}
                profiles[med]["conditions"].append(cond_entity.text)

        # Add standalone medications not yet in profiles
        for entity in entities:
            if entity.entity_type == EntityType.MEDICAMENTO:
                med = entity.text.lower()
                if med not in profiles:
                    profiles[med] = {"medication": entity.text, "dosages": [], "conditions": []}

        return list(profiles.values())

    def _build_structured_summary(
        self,
        grouped: dict[str, list[str]],
        medication_profiles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a structured summary dictionary from grouped entities."""
        summary: dict[str, Any] = {
            "condicoes": grouped.get(EntityType.CONDICAO.value, []),
            "medicamentos": [],
            "procedimentos": grouped.get(EntityType.PROCEDIMENTO.value, []),
            "valores_laboratoriais": grouped.get(EntityType.VALOR_LABORATORIAL.value, []),
        }

        # Build medication entries with dosage info
        for profile in medication_profiles:
            entry: dict[str, Any] = {"nome": profile["medication"]}
            if profile["dosages"]:
                entry["dosagem"] = ", ".join(profile["dosages"])
            if profile["conditions"]:
                entry["indicacao"] = ", ".join(profile["conditions"])
            summary["medicamentos"].append(entry)

        return summary

    def _format_entities_for_prompt(
        self,
        entities: list[ClinicalEntity],
        relations: list[ClinicalRelation],
    ) -> str:
        """Format entities and relations as text for the Granite prompt."""
        lines: list[str] = []

        grouped = self._group_entities(entities)

        for entity_type, items in grouped.items():
            if items:
                lines.append(f"\n{entity_type}:")
                for item in items:
                    lines.append(f"  - {item}")

        if relations:
            lines.append("\nRelacoes:")
            for rel in relations:
                lines.append(
                    f"  - {rel.source.text} --[{rel.relation_type}]--> {rel.target.text}"
                )

        return "\n".join(lines)
