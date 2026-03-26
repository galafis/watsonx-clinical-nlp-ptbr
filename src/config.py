"""Application configuration loaded from environment variables and settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML configuration from config/settings.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


_yaml = _load_yaml_config()


class WatsonxSettings(BaseSettings):
    """IBM Watsonx connection and model settings."""

    api_key: str = Field(default="", alias="WATSONX_API_KEY")
    project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        alias="WATSONX_URL",
    )
    generation_model: str = Field(default="ibm/granite-13b-chat-v2")


class AppSettings(BaseSettings):
    """Application server settings."""

    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8080, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")


class Settings:
    """Aggregated application settings combining env vars and YAML config."""

    def __init__(self) -> None:
        self.watsonx = WatsonxSettings()
        self.app = AppSettings()
        self.yaml = _yaml

    @property
    def ner(self) -> dict[str, Any]:
        """NER configuration from YAML."""
        return self.yaml.get("ner", {})

    @property
    def preprocessing(self) -> dict[str, Any]:
        """Preprocessing configuration from YAML."""
        return self.yaml.get("preprocessing", {})

    @property
    def summarization(self) -> dict[str, Any]:
        """Summarization configuration from YAML."""
        return self.yaml.get("summarization", {})

    @property
    def governance(self) -> dict[str, Any]:
        """Governance and compliance configuration from YAML."""
        return self.yaml.get("governance", {})

    @property
    def fhir(self) -> dict[str, Any]:
        """FHIR output configuration from YAML."""
        return self.yaml.get("fhir", {})

    @property
    def generation_params(self) -> dict[str, Any]:
        """Granite generation parameters from YAML."""
        return self.yaml.get("watsonx", {}).get("generation", {}).get("parameters", {})


settings = Settings()
