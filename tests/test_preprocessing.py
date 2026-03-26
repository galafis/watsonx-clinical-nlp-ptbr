"""Tests for the preprocessing module: normalizer, abbreviation expander, tokenizer."""

from __future__ import annotations

import pytest

from src.preprocessing.abbreviation_expander import AbbreviationExpander
from src.preprocessing.normalizer import ClinicalTextNormalizer
from src.preprocessing.tokenizer import ClinicalTokenizer

# ---------------------------------------------------------------------------
# ClinicalTextNormalizer
# ---------------------------------------------------------------------------


class TestClinicalTextNormalizer:
    """Tests for ClinicalTextNormalizer."""

    @pytest.fixture()
    def normalizer(self) -> ClinicalTextNormalizer:
        return ClinicalTextNormalizer()

    def test_normalize_empty_string(self, normalizer: ClinicalTextNormalizer) -> None:
        assert normalizer.normalize("") == ""

    def test_normalize_whitespace_only(self, normalizer: ClinicalTextNormalizer) -> None:
        assert normalizer.normalize("   ") == ""

    def test_normalize_removes_extra_whitespace(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Paciente   com    dor   abdominal"
        result = normalizer.normalize(text)
        assert "   " not in result
        assert "dor" in result

    def test_normalize_lowercases_text(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "PACIENTE COM HAS"
        result = normalizer.normalize(text)
        assert "paciente" in result

    def test_normalize_preserves_measurements(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Prescrito Losartana 50mg para hipertensao"
        result = normalizer.normalize(text)
        assert "50mg" in result

    def test_normalize_preserves_lab_values(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Hemoglobina 12,5 g/dL dentro da normalidade"
        result = normalizer.normalize(text)
        assert "12,5 g/dl" in result

    def test_normalize_preserves_dates(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Consulta em 15/03/2024 paciente estavel"
        result = normalizer.normalize(text)
        assert "15/03/2024" in result

    def test_normalize_preserves_times(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Administrado as 14:30 via oral"
        result = normalizer.normalize(text)
        assert "14:30" in result

    def test_normalize_replaces_unicode_symbols(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Temperatura: 38\u00b0C \u2013 febre moderada"
        result = normalizer.normalize(text)
        assert "\u00b0" not in result
        assert "graus" in result

    def test_normalize_removes_control_characters(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Paciente\x00 com\x0b dor"
        result = normalizer.normalize(text)
        assert "\x00" not in result
        assert "\x0b" not in result

    def test_normalize_collapses_newlines(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Linha 1\n\n\n\n\nLinha 2"
        result = normalizer.normalize(text)
        assert "\n\n\n" not in result

    def test_normalize_clinical_note(self, normalizer: ClinicalTextNormalizer) -> None:
        text = "Paciente com HAS em uso de Losartana 50mg 1x/dia. PA: 130x85 mmHg. FC: 72 bpm."
        result = normalizer.normalize(text)
        assert "50mg" in result
        assert result.strip() == result


# ---------------------------------------------------------------------------
# AbbreviationExpander
# ---------------------------------------------------------------------------


class TestAbbreviationExpander:
    """Tests for AbbreviationExpander."""

    @pytest.fixture()
    def expander(self) -> AbbreviationExpander:
        return AbbreviationExpander()

    def test_expand_empty_string(self, expander: AbbreviationExpander) -> None:
        assert expander.expand("") == ""

    def test_expand_has_inline(self, expander: AbbreviationExpander) -> None:
        result = expander.expand("Paciente com HAS", inline=True)
        assert "Hipertensao Arterial Sistemica" in result
        assert "HAS" not in result

    def test_expand_has_parenthetical(self, expander: AbbreviationExpander) -> None:
        result = expander.expand("Paciente com HAS", inline=False)
        assert "HAS" in result
        assert "Hipertensao Arterial Sistemica" in result
        assert "(" in result

    def test_expand_dm2(self, expander: AbbreviationExpander) -> None:
        result = expander.expand("DM2 controlada com dieta")
        assert "Diabetes Mellitus Tipo 2" in result

    def test_expand_iam(self, expander: AbbreviationExpander) -> None:
        result = expander.expand("Paciente com IAM previo")
        assert "Infarto Agudo do Miocardio" in result

    def test_expand_multiple_abbreviations(self, expander: AbbreviationExpander) -> None:
        text = "Paciente com HAS e DM em uso de IECA"
        result = expander.expand(text)
        assert "Hipertensao Arterial Sistemica" in result
        assert "Diabetes Mellitus" in result
        assert "Inibidor da Enzima Conversora de Angiotensina" in result

    def test_expand_medication_route(self, expander: AbbreviationExpander) -> None:
        result = expander.expand("Dipirona 500mg VO de 6/6h")
        assert "Via Oral" in result

    def test_expand_preserves_non_abbreviation_text(self, expander: AbbreviationExpander) -> None:
        text = "Paciente nega alergias medicamentosas"
        result = expander.expand(text)
        assert result == text

    def test_lookup_existing(self, expander: AbbreviationExpander) -> None:
        assert expander.lookup("HAS") == "Hipertensao Arterial Sistemica"

    def test_lookup_nonexistent(self, expander: AbbreviationExpander) -> None:
        assert expander.lookup("XYZ123") is None

    def test_abbreviation_count(self, expander: AbbreviationExpander) -> None:
        assert expander.abbreviation_count >= 100

    def test_add_abbreviation(self, expander: AbbreviationExpander) -> None:
        expander.add_abbreviation("TESTE", "Termo de Teste")
        assert expander.lookup("TESTE") == "Termo de Teste"
        result = expander.expand("Paciente com TESTE")
        assert "Termo de Teste" in result

    def test_expand_laboratory_abbreviations(self, expander: AbbreviationExpander) -> None:
        text = "HMG com HB 12 e PLT 200"
        result = expander.expand(text)
        assert "Hemograma" in result
        assert "Hemoglobina" in result
        assert "Plaquetas" in result

    def test_expand_clinical_context(self, expander: AbbreviationExpander) -> None:
        text = "BEG, LOTE, MV+ bilateral, BRNF"
        result = expander.expand(text)
        assert "Bom Estado Geral" in result
        assert "Lucido Orientado no Tempo e Espaco" in result


# ---------------------------------------------------------------------------
# ClinicalTokenizer
# ---------------------------------------------------------------------------


class TestClinicalTokenizer:
    """Tests for ClinicalTokenizer."""

    @pytest.fixture()
    def tokenizer(self) -> ClinicalTokenizer:
        return ClinicalTokenizer()

    def test_tokenize_empty_string(self, tokenizer: ClinicalTokenizer) -> None:
        assert tokenizer.tokenize("") == []

    def test_tokenize_simple_sentence(self, tokenizer: ClinicalTokenizer) -> None:
        tokens = tokenizer.tokenize("Paciente estavel")
        texts = [t.text for t in tokens]
        assert "Paciente" in texts
        assert "estavel" in texts

    def test_tokenize_measurement(self, tokenizer: ClinicalTokenizer) -> None:
        tokens = tokenizer.tokenize("Losartana 50mg via oral")
        types = {t.text: t.token_type for t in tokens}
        assert types.get("50mg") == "measurement"

    def test_tokenize_lab_value(self, tokenizer: ClinicalTokenizer) -> None:
        tokens = tokenizer.tokenize("Creatinina 1,2 mg/dL")
        has_measurement = any(t.token_type == "measurement" for t in tokens)
        assert has_measurement

    def test_tokenize_blood_pressure(self, tokenizer: ClinicalTokenizer) -> None:
        tokens = tokenizer.tokenize("PA 120x80 mmHg")
        has_measurement = any(t.token_type == "measurement" for t in tokens)
        assert has_measurement

    def test_tokenize_preserves_positions(self, tokenizer: ClinicalTokenizer) -> None:
        text = "Febre 38,5 graus"
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            assert text[token.start : token.end] == token.text

    def test_tokenize_numbers(self, tokenizer: ClinicalTokenizer) -> None:
        tokens = tokenizer.tokenize("Idade 45 anos")
        number_tokens = [t for t in tokens if t.token_type == "number"]
        assert len(number_tokens) >= 1
        assert any(t.text == "45" for t in number_tokens)

    def test_tokenize_punctuation(self, tokenizer: ClinicalTokenizer) -> None:
        tokens = tokenizer.tokenize("Paciente estavel.")
        punct_tokens = [t for t in tokens if t.token_type == "punctuation"]
        assert len(punct_tokens) >= 1

    def test_sentence_split_empty(self, tokenizer: ClinicalTokenizer) -> None:
        assert tokenizer.sentence_split("") == []

    def test_sentence_split_single(self, tokenizer: ClinicalTokenizer) -> None:
        sentences = tokenizer.sentence_split("Paciente estavel")
        assert len(sentences) == 1
        assert sentences[0].text == "Paciente estavel"

    def test_sentence_split_multiple(self, tokenizer: ClinicalTokenizer) -> None:
        text = "Paciente estavel. Evolucao favoravel. Alta programada."
        sentences = tokenizer.sentence_split(text)
        assert len(sentences) >= 2

    def test_sentence_split_newline_boundary(self, tokenizer: ClinicalTokenizer) -> None:
        text = "Queixa principal: dor abdominal\nHistoria: inicio ha 2 dias"
        sentences = tokenizer.sentence_split(text)
        assert len(sentences) >= 2

    def test_sentence_tokens_populated(self, tokenizer: ClinicalTokenizer) -> None:
        text = "Paciente com febre. Prescrito Dipirona."
        sentences = tokenizer.sentence_split(text)
        for sentence in sentences:
            assert len(sentence.tokens) > 0

    def test_tokenize_clinical_note(self, tokenizer: ClinicalTokenizer) -> None:
        text = "Paciente com HAS em uso de Losartana 50mg 1x/dia. PA: 130x85 mmHg. FC: 72 bpm."
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 10
        measurement_tokens = [t for t in tokens if t.token_type == "measurement"]
        assert len(measurement_tokens) >= 1
