"""``f1sim backtest`` end to end: DuckLake store, every baseline, report, record and gate."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from f1sim import cli
from f1sim.adapters.benchmarks import load_runs
from f1sim.adapters.models.registry import model_names
from f1sim.adapters.store.ducklake_store import DuckLakeStore
from tests.support import synthetic_results

pytestmark = pytest.mark.integration

RULES_DIR = Path(__file__).resolve().parents[2] / "configs" / "rules"
"""The repository's season rules; the simulator reads them and the test runs elsewhere."""


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working folder with a DuckLake store holding two synthetic seasons."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("F1SIM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("F1SIM_RULES_DIR", str(RULES_DIR))
    monkeypatch.setenv("F1SIM_N_SIMS", "300")
    store = DuckLakeStore(tmp_path / "data" / "lake")
    try:
        store.commit({"results": synthetic_results(seasons=(2023, 2024), rounds=6)}, run_id="t")
    finally:
        store.close()
    return tmp_path


def test_backtest_reports_records_and_gates(workdir: Path) -> None:
    runner = CliRunner()
    args = ["backtest", "--model", "all", "--seasons", "2024", "--workers", "2"]

    printed = runner.invoke(cli.app, args)
    assert printed.exit_code == 0, printed.output
    assert "## Leaderboard" in printed.output

    report = workdir / "backtest.md"
    recorded = runner.invoke(cli.app, [*args, "--report", str(report), "--record"])
    assert recorded.exit_code == 0, recorded.output
    assert "Calibration: podium" in report.read_text()
    runs = load_runs(workdir / "benchmarks", "plackett_luce")
    assert runs[0]["seasons"] == [2024]
    assert cli.cited_snapshots([workdir / "benchmarks"]) == {runs[0]["snapshot_id"]}

    gated = runner.invoke(cli.app, [*args, "--check"])
    assert gated.exit_code == 0, gated.output
    assert gated.output.count("(ok)") == len(model_names())


def test_gate_fails_on_a_regression(workdir: Path) -> None:
    runner = CliRunner()
    args = ["backtest", "--model", "grid_noise", "--seasons", "2024"]
    assert runner.invoke(cli.app, [*args, "--record"]).exit_code == 0
    path = workdir / "benchmarks" / "grid_noise.json"
    path.write_text(path.read_text().replace('"rps": 0.', '"rps": 0.0'))
    gated = runner.invoke(cli.app, [*args, "--check"])
    assert gated.exit_code == 1
    assert "REGRESSION" in gated.output


@pytest.mark.parametrize(
    ("option", "value"), [("--model", "nope"), ("--seasons", "20x4"), ("--seasons", ",")]
)
def test_bad_options_are_rejected(workdir: Path, option: str, value: str) -> None:
    del workdir
    result = CliRunner().invoke(cli.app, ["backtest", option, value])
    assert result.exit_code == 2


def test_parse_seasons_accepts_lists_and_ranges() -> None:
    assert cli.parse_seasons("2024,2021-2022") == (2021, 2022, 2024)
