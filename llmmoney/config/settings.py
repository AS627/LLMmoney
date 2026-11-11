"""Central configuration loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    simfin_api_key: str = Field(default="", alias="SIMFIN_API_KEY")
    xai_api_key: str = Field(default="", alias="XAI_API_KEY")
    xai_model: str = Field(default="grok-4", alias="XAI_MODEL")
    xai_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")
    data_dir: Path = Field(default=Path("data"), alias="LLMMONEY_DATA_DIR")

    # Signal definition
    horizon_days: int = 5
    benchmark: str = "SPY"
    flat_band: float = 0.01  # |excess return| below this => "flat"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "llmmoney.duckdb"

    @property
    def simfin_dir(self) -> Path:
        return self.data_dir / "simfin"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s


def load_universe() -> list[str]:
    with open(CONFIG_DIR / "universe.yaml") as f:
        return list(yaml.safe_load(f)["tickers"])


def load_voices() -> list[dict]:
    """Curated X accounts: [{handle, weight, note}]."""
    with open(CONFIG_DIR / "voices.yaml") as f:
        return list(yaml.safe_load(f)["voices"])
