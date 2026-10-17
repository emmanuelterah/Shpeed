# pylint: disable=duplicate-code  # the CLI store/env fixtures mirror test_pace_cli's
"""``f1sim tyres`` end to end on a DuckLake store holding synthetic seasons."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from f1sim import cli
from f1sim.adapters.store.ducklake_store import DuckLakeStore
from tests.tyres_world import build_tyres_world

pytestmark = pytest.mark.integration

RULES_DIR = Path(__file__).resolve().parents[2] / "configs" / "rules"


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A working folder with a DuckLake store holding two synthetic seasons."""
    root = tmp_path_factory.mktemp("tyres")
    store = DuckLakeStore(root / "data" / "lake")
    try:
        store.commit(build_tyres_world(seasons=(2023, 2024)).tables, run_id="t")
    finally:
        store.close()
    return root


@pytest.fixture
def env(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at the work folder and the repository's rules files."""
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("F1SIM_DATA_DIR", str(workdir / "data"))
    monkeypatch.setenv("F1SIM_RULES_DIR", str(RULES_DIR))
    return workdir


def test_fit_writes_the_model(env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["tyres", "fit"])
    assert result.exit_code == 0, result.output
    record = json.loads((env / "configs" / "tyre_model.json").read_text())
    assert set(record["degradation"]["global"]) == {"SOFT", "MEDIUM", "HARD"}
    assert "Wrote" in result.output


def test_check_reports_the_held_out_error(env: Path) -> None:
    assert env.is_dir()
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["tyres", "check", "--seasons", "2023", "--holdout", "2024", "--max-rmse", "100"]
    )
    assert result.exit_code == 0, result.output
    assert "Held-out stint RMSE" in result.output
    assert "PASS" in result.output


def test_windows_renders_a_report(env: Path) -> None:
    runner = CliRunner()
    out = env / "pit_windows.md"
    result = runner.invoke(
        cli.app,
        ["tyres", "windows", "--train", "2023", "--holdout", "2024", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.read_text().startswith("# Pit-window calibration")
