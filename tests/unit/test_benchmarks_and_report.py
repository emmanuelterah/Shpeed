"""Benchmark registry, regression gate and the Markdown backtest report."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from f1sim.adapters.benchmarks import (
    GATE_TOLERANCE,
    UNKNOWN_COMMIT,
    check_regression,
    current_commit,
    load_runs,
    record_run,
)
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.errors import DataError
from f1sim.core.scoring.metrics import CalibrationBin
from f1sim.core.scoring.race_score import RaceScore
from f1sim.core.usecases.backtest import METRICS, BacktestResult, SkippedRace
from f1sim.reports.backtest_report import LABELS, render_backtest


def _result(
    model: str = "elo", rps: float = 0.15, seasons: tuple[int, ...] = (2024,)
) -> BacktestResult:
    race = RaceScore(2024, 1, 20, 0.04, 0.1, 0.2, 2.5, rps, 0.6)
    bins = {"win": [CalibrationBin(0.0, 0.1, 0.05, 0.04, 380)]}
    skipped = [SkippedRace(EventKey(2024, 2), "no qualifying or sprint results")]
    return BacktestResult(model, seasons, SnapshotId(3), [race], skipped, bins)


def test_report_labels_cover_every_metric() -> None:
    assert set(LABELS) == set(METRICS)


def test_report_has_leaderboard_calibration_and_races() -> None:
    text = render_backtest([_result("elo", 0.16), _result("plackett_luce", 0.15)])
    board = text.split("## Leaderboard")[1].split("## ")[0]
    assert board.index("plackett_luce") < board.index("elo")
    assert "### Calibration: win" in text
    assert "| 2024 | 1 | 20 |" in text
    assert "2024 round 2: no qualifying" in text
    assert "No backtests were run." in render_backtest([])


def test_record_and_reload_runs(tmp_path: Path) -> None:
    folder = tmp_path / "benchmarks"
    when = datetime(2026, 10, 11, tzinfo=UTC)
    path = record_run(folder, _result(), commit="abc", seed=1, now=when)
    record_run(folder, _result(rps=0.14), commit="def", seed=1)
    runs = load_runs(folder, "elo")
    assert [r["git_commit"] for r in runs] == ["abc", "def"]
    assert runs[0]["snapshot_id"] == 3
    assert runs[0]["recorded_at"].startswith("2026-10-11")
    assert json.loads(path.read_text())["model"] == "elo"
    assert load_runs(folder, "nobody") == []


def test_gate_compares_against_the_best_run_on_the_same_seasons(tmp_path: Path) -> None:
    assert check_regression(tmp_path, _result()).passed
    record_run(tmp_path, _result(rps=0.15), commit="a", seed=0)
    record_run(tmp_path, _result(rps=0.10, seasons=(2023,)), commit="a", seed=0)
    ok = check_regression(tmp_path, _result(rps=0.15 + GATE_TOLERANCE / 2))
    assert ok.passed
    assert ok.best == pytest.approx(0.15)
    bad = check_regression(tmp_path, _result(rps=0.16))
    assert not bad.passed
    assert "REGRESSION" in bad.message


def test_gate_needs_scored_races(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        check_regression(tmp_path, BacktestResult("elo", (2024,), SnapshotId(1), []))


def test_corrupt_benchmark_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "elo.json").write_text("{not json")
    with pytest.raises(DataError):
        load_runs(tmp_path, "elo")
    (tmp_path / "elo.json").write_text('{"runs": 3}')
    with pytest.raises(DataError):
        load_runs(tmp_path, "elo")


def test_current_commit_outside_git_is_unknown(tmp_path: Path) -> None:
    assert current_commit(tmp_path) == UNKNOWN_COMMIT
    assert current_commit(tmp_path / "missing") == UNKNOWN_COMMIT


def test_current_commit_marks_uncommitted_changes(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "a.txt").write_text("one")
    git("add", "a.txt")
    git("commit", "-q", "-m", "first")
    clean = current_commit(tmp_path)
    assert len(clean) == 40
    (tmp_path / "a.txt").write_text("two")
    assert current_commit(tmp_path) == f"{clean}-dirty"
