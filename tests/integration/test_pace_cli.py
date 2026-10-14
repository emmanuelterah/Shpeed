"""``f1sim pace`` end to end on a DuckLake store holding synthetic seasons."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from f1sim import cli
from f1sim.adapters.store.ducklake_store import DuckLakeStore
from tests.pace_world import build_world

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A working folder with a DuckLake store holding two synthetic seasons."""
    root = tmp_path_factory.mktemp("pace")
    store = DuckLakeStore(root / "data" / "lake")
    try:
        store.commit(build_world(seasons=(2023, 2024), rounds=4).tables, run_id="t")
    finally:
        store.close()
    return root


@pytest.fixture
def env(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at the work folder."""
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("F1SIM_DATA_DIR", str(workdir / "data"))
    return workdir


def test_fit_writes_hyperparameters_used_by_check(env: Path) -> None:
    runner = CliRunner()
    out = env / "configs" / "pace_hyper.json"
    fitted = runner.invoke(
        cli.app,
        ["pace", "fit", "--seasons", "2023-2024", "--method", "laplace"],
    )
    assert fitted.exit_code in (0, 1), fitted.output
    assert "R-hat" in fitted.output
    record = json.loads(out.read_text())
    assert record["seasons"] == [2023, 2024]
    assert fitted.exit_code == (0 if record["converged"] else 1)
    checked = runner.invoke(
        cli.app, ["pace", "check", "--seasons", "2023-2024", "--holdout", "2024"]
    )
    assert ("PASS" in checked.output) == (checked.exit_code == 0), checked.output
    assert "90% holds" in checked.output


def test_reference_and_weekend_update(env: Path) -> None:
    runner = CliRunner()
    reference = runner.invoke(cli.app, ["pace", "reference", "--seasons", "2024"])
    assert reference.exit_code == 0, reference.output
    assert "Held-out lap error" in reference.output
    first = runner.invoke(cli.app, ["pace", "update", "--season", "2024", "--round", "4"])
    assert first.exit_code == 0, first.output
    assert "team:" in first.output
    again = runner.invoke(cli.app, ["pace", "update", "--season", "2024", "--round", "4"])
    assert "resumed from 2024 round 4" in again.output
    assert list((env / "data" / "state" / "pace").rglob("2024-04-s*.json"))
