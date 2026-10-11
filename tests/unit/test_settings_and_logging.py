"""Settings loading and structured-logging setup."""

from f1sim.observability import configure_logging
from f1sim.settings import Settings


def test_settings_defaults_and_env_override(monkeypatch) -> None:
    monkeypatch.setenv("F1SIM_N_SIMS", "500")
    s = Settings()
    assert s.n_sims == 500
    assert s.seed == 20261012


def test_configure_logging_returns_run_id() -> None:
    run_id = configure_logging("INFO")
    assert len(run_id) == 12
