"""``f1sim predict`` and ``f1sim report score`` end to end through a DuckLake store (P8-01)."""

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from f1sim import cli
from f1sim.adapters.records import validate_index, validate_record
from tests.contract.test_publisher_contract import init_repo
from tests.support import seed_store, synthetic_results

pytestmark = pytest.mark.integration

RULES = Path(__file__).resolve().parents[2] / "configs" / "rules"
PREDICT = ["predict", "--season", "2024", "--model", "plackett_luce"]
SCORE = ["report", "score", "--season", "2024"]


def _calendar(lights_out: datetime) -> pd.DataFrame:
    """A calendar whose round 3 starts at ``lights_out``."""
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "round": r,
                "event_name": f"Round {r} Grand Prix",
                "circuit_id": f"c{r}",
                "locality": "Testville",
                "country": "Testland",
                "event_date": lights_out + timedelta(days=14 * (r - 3)),
                "event_format": "conventional",
                "available_at": datetime(2024, 1, 1),
            }
            for r in (1, 2, 3)
        ]
    )


def _workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, results: pd.DataFrame, **tables: Any
) -> Path:
    """A git working folder with a store and the test settings."""
    init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("F1SIM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("F1SIM_RULES_DIR", str(RULES))
    monkeypatch.setenv("F1SIM_N_SIMS", "200")
    seed_store(tmp_path / "data", results, **tables)
    return tmp_path


def _unraced_round_3() -> pd.DataFrame:
    """Three rounds, the third past qualifying but not yet raced."""
    df = synthetic_results(seasons=(2024,), rounds=3)
    return df.loc[~((df["round"] == 3) & (df["session"] == "R"))].reset_index(drop=True)


def _run(*args: str) -> Result:
    """Invoke the CLI."""
    return CliRunner().invoke(cli.app, list(args))


def _read(path: Path) -> dict[str, Any]:
    """Parse one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _commits(repo: Path) -> list[str]:
    """Commit subjects, newest first."""
    done = subprocess.run(
        ["git", "log", "--format=%s"], cwd=repo, capture_output=True, text=True, check=False
    )
    return done.stdout.split("\n")[:-1]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working folder with one fully raced synthetic season and no calendar."""
    return _workdir(tmp_path, monkeypatch, synthetic_results(seasons=(2024,), rounds=3))


def test_predict_then_score_round_trip(workdir: Path) -> None:
    forecast = _run(*PREDICT, "--round", "3", "--no-commit")
    assert forecast.exit_code == 0, forecast.output
    assert "published" in forecast.output

    record = _read(workdir / "predictions" / "2024" / "03.json")
    validate_record(record)
    assert record["score"] is None
    assert len(record["drivers"]) == 10
    index = _read(workdir / "predictions" / "index.json")
    validate_index(index)
    assert index["latest"] == "2024/03.json"
    assert index["races"][0]["score"] is None

    again = _run(*PREDICT, "--round", "3", "--no-commit")
    assert again.exit_code == 0, again.output
    assert "already up to date" in again.output

    scored = _run(*SCORE, "--round", "3", "--no-commit")
    assert scored.exit_code == 0, scored.output
    assert "RPS" in scored.output

    after = _read(workdir / "predictions" / "2024" / "03.json")
    validate_record(after)
    assert after["drivers"] == record["drivers"]
    assert sorted(p for p in after["score"]["positions"] if p is not None)[0] == 1
    index = _read(workdir / "predictions" / "index.json")
    assert index["races"][0]["score"]["rps"] == after["score"]["rps"]
    assert index["calibration"]["races_scored"] == 1

    rescore = _run(*SCORE, "--round", "3", "--no-commit")
    assert "already scored" in rescore.output


def test_each_publish_is_one_commit(workdir: Path) -> None:
    assert _run(*PREDICT, "--round", "3").exit_code == 0
    assert _run(*PREDICT, "--round", "3").exit_code == 0
    assert _run(*SCORE, "--round", "3").exit_code == 0
    assert _commits(workdir) == [
        "chore(predictions): score 2024 round 3 (P8-07)",
        "chore(predictions): publish 2024 round 3 forecast (P8-06)",
    ]


def test_latest_forecasts_the_newest_qualified_weekend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lights_out = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)
    workdir = _workdir(tmp_path, monkeypatch, _unraced_round_3(), events=_calendar(lights_out))
    result = _run(*PREDICT, "--latest", "--no-commit")
    assert result.exit_code == 0, result.output
    record = _read(workdir / "predictions" / "2024" / "03.json")
    assert record["event_name"] == "Round 3 Grand Prix"
    nothing = _run(*SCORE, "--latest", "--no-commit")
    assert nothing.exit_code == 0, nothing.output
    assert "nothing to score" in nothing.output


def test_latest_is_a_no_op_outside_a_race_weekend(workdir: Path) -> None:
    result = _run(*PREDICT, "--latest", "--no-commit")
    assert result.exit_code == 0, result.output
    assert "no weekend" in result.output
    assert not (workdir / "predictions").exists()


def test_a_forecast_after_lights_out_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = datetime(2024, 3, 29, 15, 0)
    workdir = _workdir(tmp_path, monkeypatch, _unraced_round_3(), events=_calendar(started))
    result = _run(*PREDICT, "--round", "3", "--no-commit")
    assert result.exit_code == 1
    assert "too late" in result.output
    assert not (workdir / "predictions" / "2024" / "03.json").exists()


def test_score_without_a_prediction_fails(workdir: Path) -> None:
    del workdir
    result = _run(*SCORE, "--round", "1", "--no-commit")
    assert result.exit_code == 1
    assert "no published prediction" in result.output


def test_round_and_latest_are_exclusive(workdir: Path) -> None:
    del workdir
    assert _run(*PREDICT, "--round", "1", "--latest").exit_code == 2
    assert _run(*PREDICT).exit_code == 2
