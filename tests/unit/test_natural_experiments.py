"""The natural-experiment manifest for the regulation-reset study (P7-01)."""

from pathlib import Path

import pytest

from f1sim.adapters.config_files import load_natural_experiments
from f1sim.core.domain.natural_experiment import LAP_LEVEL, RESULTS_LEVEL
from f1sim.core.errors import DataError

MANIFEST = Path(__file__).resolve().parents[2] / "configs" / "natural-experiments.yaml"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "experiments.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_shipped_manifest_declares_every_regulation_reset() -> None:
    experiments = load_natural_experiments(MANIFEST)
    by_id = {e.experiment_id: e for e in experiments}
    assert set(by_id) == {"reg-2014", "reg-2022", "reg-2026"}
    assert by_id["reg-2022"].level == LAP_LEVEL
    assert by_id["reg-2022"].snapshot == 10
    assert by_id["reg-2022"].seasons == (2021, 2022)
    assert by_id["reg-2014"].level == RESULTS_LEVEL
    assert not by_id["reg-2014"].pinned
    assert by_id["reg-2026"].reset_season == 2026
    assert not by_id["reg-2026"].pinned


def test_a_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load_natural_experiments(tmp_path / "absent.yaml")


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load_natural_experiments(_write(tmp_path, "experiments: [oops"))


def test_an_empty_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load_natural_experiments(_write(tmp_path, "experiments: []"))


def test_a_bad_level_is_rejected(tmp_path: Path) -> None:
    body = """
experiments:
  - id: reg-x
    name: X
    level: sector
    seasons: [2021, 2022]
    reset_season: 2022
"""
    with pytest.raises(DataError):
        load_natural_experiments(_write(tmp_path, body))


def test_a_reset_season_outside_the_seasons_is_rejected(tmp_path: Path) -> None:
    body = """
experiments:
  - id: reg-x
    name: X
    level: lap
    seasons: [2021, 2022]
    reset_season: 2025
"""
    with pytest.raises(DataError):
        load_natural_experiments(_write(tmp_path, body))


def test_an_entry_without_a_snapshot_is_declared_but_not_pinned(tmp_path: Path) -> None:
    body = """
experiments:
  - id: reg-x
    name: X
    level: results
    seasons: [2013, 2014]
    reset_season: 2014
    notes: not synced
"""
    experiment = load_natural_experiments(_write(tmp_path, body))[0]
    assert experiment.snapshot is None
    assert not experiment.pinned
    assert experiment.reset_round == 1
    assert experiment.notes == "not synced"
