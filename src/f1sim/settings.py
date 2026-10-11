"""Typed settings (P0-13). The only module that reads environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable with ``F1SIM_*`` environment variables or ``.env``."""

    model_config = SettingsConfigDict(env_prefix="F1SIM_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    log_level: str = "INFO"
    seed: int = 20261012
    n_sims: int = 10_000
    sim_time_budget_s: float = 60.0
    predictions_dir: Path = Path("predictions")
    cors_origins: list[str] = []
    """Origins allowed to call the API from a browser, e.g. ``["http://localhost:5173"]``."""
