"""Typed runtime settings (P0-13).

This is the only module that reads environment variables. Every setting can be overridden
with an ``F1SIM_``-prefixed variable or a ``.env`` file, for example ``F1SIM_DATA_DIR=/data``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the CLI, the API and scheduled runs.

    Attributes:
        data_dir: Root of the local data store (``<data_dir>/lake`` holds DuckLake).
        cache_dir: HTTP and FastF1 caches.
        log_level: Minimum level written to the JSON logs.
        seed: Default random seed for simulations.
        n_sims: Monte Carlo runs per race.
        sim_time_budget_s: Time budget for one race simulation, in seconds.
        predictions_dir: Where published prediction records are read and written.
        cors_origins: Browser origins allowed to call the API directly.
        pace_hyper_path: Fitted pace-model hyperparameters written by ``f1sim pace fit``;
            the built-in defaults are used while the file does not exist.
        tyre_model_path: Fitted tyre and strategy model written by ``f1sim tyres fit``.
        rules_dir: Folder holding ``<season>.yaml`` season-rules files (P4-06).
        experiments_path: Natural-experiment manifest for the regulation-reset study (P7-01).
    """

    model_config = SettingsConfigDict(env_prefix="F1SIM_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    log_level: str = "INFO"
    seed: int = 20261012
    n_sims: int = 10_000
    sim_time_budget_s: float = 60.0
    predictions_dir: Path = Path("predictions")
    cors_origins: list[str] = []
    pace_hyper_path: Path = Path("configs/pace_hyper.json")
    tyre_model_path: Path = Path("configs/tyre_model.json")
    rules_dir: Path = Path("configs/rules")
    experiments_path: Path = Path("configs/natural-experiments.yaml")

    @property
    def state_dir(self) -> Path:
        """Folder for fitted model states (P3-10).

        Returns:
            ``<data_dir>/state``.
        """
        return self.data_dir / "state"
