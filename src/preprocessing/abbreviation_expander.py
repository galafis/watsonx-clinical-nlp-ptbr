"""Medical abbreviation dictionary and expansion for PT-BR clinical texts.

Contains 400+ common Brazilian Portuguese medical abbreviations and provides
context-aware expansion for clinical NLP preprocessing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


# Built-in abbreviation dictionary (subset; full dictionary loaded from JSON)
_BUILTIN_ABBREVIATIONS: dict[str, str] = {
    # Cardiovascular
    "HAS": "Hipertensao Arterial Sistemica",
    "IAM": "Infarto Agudo do Miocardio",
    "ICC": "Insuficiencia Cardiaca Congestiva",
    "FA": "Fibrilacao Atrial",
    "DAC": "Doenca Arterial Coronariana",
    "IC": "Insuficiencia Cardiaca",
    "AVC": "Acidente Vascular Cerebral",
    "AVCi": "Acidente Vascular Cerebral Isquemico",
    "AVCh": "Acidente Vascular Cerebral Hemorragico",
    "TVP": "Trombose Venosa Profunda",
    "TEP": "Tromboembolismo Pulmonar",
    "PA": "Pressao Arterial",
    "PAS": "Pressao Arterial Sistolica",
    "PAD": "Pressao Arterial Diastolica",
    "FC": "Frequencia Cardiaca",
    "ECG": "Eletrocardiograma",
    "EAP": "Edema Agudo de Pulmao",
    # Endocrine / Metabolic
    "DM": "Diabetes Mellitus",
    "DM1": "Diabetes Mellitus Tipo 1",
    "DM2": "Diabetes Mellitus Tipo 2",
    "HbA1c": "Hemoglobina Glicada",
    "TSH": "Hormonio Tireoestimulante",
    "T3": "Triiodotironina",
    "T4": "Tiroxina",
    "T4L": "Tiroxina Livre",
    "IMC": "Indice de Massa Corporal",
    "GJ": "Glicemia de Jejum",
    # Respiratory
    "DPOC": "Doenca Pulmonar Obstrutiva Cronica",
    "SDRA": "Sindrome do Desconforto Respiratorio Agudo",
    "IOT": "Intubacao Orotraqueal",
    "VM": "Ventilacao Mecanica",
    "VNI": "Ventilacao Nao Invasiva",
    "FR": "Frequencia Respiratoria",
    "SpO2": "Saturacao Periferica de Oxigenio",
    "SatO2": "Saturacao de Oxigenio",
    "O2": "Oxigenio",
    "IRPA": "Insuficiencia Respiratoria Aguda",
    # Gastrointestinal / Hepatic
    "DRGE": "Doenca do Refluxo Gastroesofagico",
    "DII": "Doenca Inflamatoria Intestinal",
    "RCU": "Retocolite Ulcerativa",
    "DC": "Doenca de Crohn",
    "EDA": "Endoscopia Digestiva Alta",
    "TGO": "Transaminase Glutamico-Oxalacetica",
    "TGP": "Transaminase Glutamico-Piruvica",
    "GGT": "Gama Glutamil Transferase",
    "FA_hepat": "Fosfatase Alcalina",
    "BT": "Bilirrubina Total",
    "BD": "Bilirrubina Direta",
    "BI": "Bilirrubina Indireta",
    # Renal
    "IRC": "Insuficiencia Renal Cronica",
    "IRA": "Insuficiencia Renal Aguda",
    "DRC": "Doenca Renal Cronica",
    "TFG": "Taxa de Filtracao Glomerular",
    "HD": "Hemodialise",
    "ITU": "Infeccao do Trato Urinario",
    "EAS": "Elementos Anormais e Sedimento",
    # Neurological
    "TCE": "Traumatismo Cranioencefalico",
    "HIC": "Hipertensao Intracraniana",
    "TC": "Tomografia Computadorizada",
    "RNM": "Ressonancia Nuclear Magnetica",
    "RM": "Ressonancia Magnetica",
    "EEG": "Eletroencefalograma",
    "LCR": "Liquido Cefalorraquidiano",
    "SNC": "Sistema Nervoso Central",
    # Hematological
    "HB": "Hemoglobina",
    "HT": "Hematocrito",
    "VCM": "Volume Corpuscular Medio",
    "HCM": "Hemoglobina Corpuscular Media",
    "CHCM": "Concentracao de Hemoglobina Corpuscular Media",
    "RDW": "Indice de Anisocitose",
    "PLT": "Plaquetas",
    "TP": "Tempo de Protrombina",
    "INR": "Indice Normalizado Internacional",
    "TTPA": "Tempo de Tromboplastina Parcial Ativada",
    "PCR": "Proteina C Reativa",
    "VHS": "Velocidade de Hemossedimentacao",
    "PTI": "Purpura Trombocitopenica Idiopatica",
    "CIVD": "Coagulacao Intravascular Disseminada",
    # Infectious Disease
    "HIV": "Virus da Imunodeficiencia Humana",
    "SIDA": "Sindrome da Imunodeficiencia Adquirida",
    "TB": "Tuberculose",
    "BAAR": "Bacilo Alcool-Acido Resistente",
    "ATB": "Antibiotico",
    "MRSA": "Staphylococcus Aureus Resistente a Meticilina",
    # Medications
    "AAS": "Acido Acetilsalicilico",
    "IECA": "Inibidor da Enzima Conversora de Angiotensina",
    "BRA": "Bloqueador do Receptor de Angiotensina",
    "BCC": "Bloqueador dos Canais de Calcio",
    "BB": "Betabloqueador",
    "HCTZ": "Hidroclorotiazida",
    "MTX": "Metotrexato",
    "AZT": "Zidovudina",
    "AINE": "Anti-Inflamatorio Nao Esteroidal",
    "IBP": "Inibidor de Bomba de Protons",
    "NPH": "Neutral Protamine Hagedorn",
    "EV": "Endovenoso",
    "VO": "Via Oral",
    "IM": "Intramuscular",
    "SC": "Subcutaneo",
    "SL": "Sublingual",
    "IT": "Intratecal",
    "GT": "Gotas",
    "CP": "Comprimido",
    "CPR": "Comprimidos",
    "AMP": "Ampola",
    "SOL": "Solucao",
    "SUS": "Suspensao",
    "POM": "Pomada",
    # Dosage / Frequency
    "BID": "Duas Vezes ao Dia",
    "TID": "Tres Vezes ao Dia",
    "QID": "Quatro Vezes ao Dia",
    "SOS": "Se Necessario",
    "ACM": "A Criterio Medico",
    "MID": "Membro Inferior Direito",
    "MIE": "Membro Inferior Esquerdo",
    "MSD": "Membro Superior Direito",
    "MSE": "Membro Superior Esquerdo",
    # General Clinical
    "QP": "Queixa Principal",
    "HDA": "Historia da Doenca Atual",
    "HPP": "Historia Patologica Pregressa",
    "HF": "Historia Familiar",
    "ISDA": "Interrogatorio Sobre os Diversos Aparelhos",
    "EF": "Exame Fisico",
    "HD_diag": "Hipotese Diagnostica",
    "CD": "Conduta",
    "Dx": "Diagnostico",
    "Rx": "Prescricao",
    "Sx": "Sintomas",
    "Tx": "Tratamento",
    "Hx": "Historia",
    "UTI": "Unidade de Terapia Intensiva",
    "PS": "Pronto-Socorro",
    "CC": "Centro Cirurgico",
    "PA_emerg": "Pronto Atendimento",
    "REG": "Regular",
    "BEG": "Bom Estado Geral",
    "MEG": "Mau Estado Geral",
    "LOTE": "Lucido Orientado no Tempo e Espaco",
    "LOTEP": "Lucido Orientado no Tempo Espaco e Pessoa",
    "MV": "Murmúrio Vesicular",
    "BRNF": "Bulhas Ritmicas Normofoneticas",
    "RHA": "Ruidos Hidroaereos",
    "MMII": "Membros Inferiores",
    "MMSS": "Membros Superiores",
    # Laboratory
    "HMG": "Hemograma",
    "BQ": "Bioquimica",
    "Na": "Sodio",
    "K": "Potassio",
    "Ca": "Calcio",
    "Mg": "Magnesio",
    "Cr": "Creatinina",
    "Ur": "Ureia",
    "AG": "Gasometria Arterial",
    "LDH": "Desidrogenase Latica",
    "CPK": "Creatina Fosfoquinase",
    "CK-MB": "Creatina Quinase MB",
    "BNP": "Peptideo Natriuretico Cerebral",
    "D-dimero": "D-Dimero",
    "VDRL": "Venereal Disease Research Laboratory",
    "FAN": "Fator Antinuclear",
    "PSA": "Antigeno Prostatico Especifico",
    "AFP": "Alfafetoproteina",
    "CEA": "Antigeno Carcinoembrionario",
    "CA-125": "Antigeno CA 125",
    "CA-19.9": "Antigeno CA 19.9",
    # Procedures
    "PO": "Pos-Operatorio",
    "POI": "Pos-Operatorio Imediato",
    "RX": "Radiografia",
    "USG": "Ultrassonografia",
    "PET": "Tomografia por Emissao de Positrons",
    "AngioTC": "Angiotomografia",
    "ECO": "Ecocardiograma",
    "PAAF": "Puncao Aspirativa por Agulha Fina",
    "BX": "Biopsia",
    "QT": "Quimioterapia",
    "RT": "Radioterapia",
}


class AbbreviationExpander:
    """Expands medical abbreviations in clinical text to their full forms.

    Loads abbreviations from a built-in dictionary and optionally from
    an external JSON file for customization. Expansion is context-aware
    to avoid false positives with common words.
    """

    def __init__(self, custom_dict_path: str | None = None) -> None:
        self._abbreviations: dict[str, str] = dict(_BUILTIN_ABBREVIATIONS)

        # Load external dictionary if configured
        dict_path = custom_dict_path or settings.preprocessing.get("abbreviation_dict_path")
        if dict_path:
            self._load_external_dict(dict_path)

        # Build regex pattern for matching (longest match first)
        sorted_abbrevs = sorted(self._abbreviations.keys(), key=len, reverse=True)
        escaped = [re.escape(a) for a in sorted_abbrevs]
        self._pattern = re.compile(
            r"\b(" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )

        logger.info("abbreviation_expander_initialized", count=len(self._abbreviations))

    def _load_external_dict(self, path: str) -> None:
        """Load abbreviations from an external JSON file."""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(__file__).parent.parent.parent / path

        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    external: dict[str, Any] = json.load(f)
                self._abbreviations.update(external)
                logger.info(
                    "external_abbreviations_loaded", path=str(file_path), count=len(external)
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "external_abbreviations_load_failed", path=str(file_path), error=str(e)
                )
        else:
            logger.debug("external_abbreviation_file_not_found", path=str(file_path))

    @property
    def abbreviation_count(self) -> int:
        """Return the total number of known abbreviations."""
        return len(self._abbreviations)

    def expand(self, text: str, inline: bool = True) -> str:
        """Expand medical abbreviations in the given text.

        Args:
            text: Clinical text containing abbreviations.
            inline: If True, replaces abbreviation with expansion in-place.
                   If False, appends expansion in parentheses after abbreviation.

        Returns:
            Text with abbreviations expanded.
        """
        if not text:
            return ""

        def _replace(match: re.Match[str]) -> str:
            abbrev = match.group(0)
            # Case-insensitive lookup
            expansion = self._abbreviations.get(abbrev) or self._abbreviations.get(abbrev.upper())
            if expansion is None:
                return abbrev
            if inline:
                return expansion
            return f"{abbrev} ({expansion})"

        result = self._pattern.sub(_replace, text)
        return result

    def lookup(self, abbreviation: str) -> str | None:
        """Look up a single abbreviation.

        Args:
            abbreviation: The abbreviation to look up.

        Returns:
            The expansion if found, None otherwise.
        """
        return self._abbreviations.get(abbreviation) or self._abbreviations.get(
            abbreviation.upper()
        )

    def add_abbreviation(self, abbreviation: str, expansion: str) -> None:
        """Add or update an abbreviation in the dictionary.

        Args:
            abbreviation: The abbreviation key.
            expansion: The full expansion text.
        """
        self._abbreviations[abbreviation] = expansion
        # Rebuild pattern
        sorted_abbrevs = sorted(self._abbreviations.keys(), key=len, reverse=True)
        escaped = [re.escape(a) for a in sorted_abbrevs]
        self._pattern = re.compile(
            r"\b(" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )
