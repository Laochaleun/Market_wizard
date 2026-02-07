"""Configuration settings for Market Wizard."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get absolute path to .env file in backend directory
BACKEND_DIR = Path(__file__).parent.parent
ENV_FILE = BACKEND_DIR / ".env"
DEFAULT_SSR_CALIBRATION_ARTIFACT = BACKEND_DIR / "app" / "data" / "ssr_calibrator_default.json"
DEFAULT_SSR_CALIBRATION_POLICY_ARTIFACT = BACKEND_DIR / "app" / "data" / "ssr_calibration_policy_default.json"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    google_api_key: str = Field(default="", validation_alias="GOOGLE_API_KEY")
    openai_api_key: str = ""

    # LLM Settings
    llm_model: str = "gemini-2.0-flash-001"
    llm_temperature: float = 0.5
    research_llm_model: str = "gemini-2.5-flash-lite"
    research_interpretation_model: str = "gemini-3-flash-preview"
    report_analysis_model: str = "gemini-3-pro-preview"
    report_analysis_thinking_budget: int = 256
    report_analysis_max_output_tokens: int = 16384
    research_playwright_fallback_limit: int = 2
    research_playwright_timeout_ms: int = 15000
    research_json_ld_only: bool = False

    # Embedding Settings
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_warmup: bool = True
    ssr_temperature: float = 1.0
    ssr_epsilon: float = 0.0
    ssr_calibration_enabled: bool = True
    ssr_calibration_artifact_path: str = str(DEFAULT_SSR_CALIBRATION_ARTIFACT)
    ssr_calibration_policy_path: str = str(DEFAULT_SSR_CALIBRATION_POLICY_ARTIFACT)

    # Database
    database_url: str = "sqlite+aiosqlite:///./market_wizard.db"

    # GUS API
    gus_api_key: str = ""
    gus_api_base_url: str = "https://bdl.stat.gov.pl/api/v1"
    gus_use_live: bool = True
    gus_cache_ttl_hours: int = 24
    gus_unit_id_poland: str = "000000000000"
    gus_income_net_ratio: float = 0.72


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
