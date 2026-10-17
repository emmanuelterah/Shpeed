"""The pit-window calibration counts hits and renders a Markdown report (P4-10)."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from f1sim.adapters.models.strategy import PitWindow
from f1sim.core.domain.entities import SnapshotId
from f1sim.reports.pit_windows import (
    PitWindowCalibration,
    calibrate_pit_windows,
    render_pit_windows,
    window_hits,
)
from tests.tyres_world import build_tyres_world

RULES_DIR = Path(__file__).resolve().parents[2] / "configs" / "rules"


class StubReader:
    """A DataReader over an in-memory table mapping."""

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        """Store the tables the reader will serve."""
        self._tables = tables

    def latest_snapshot(self) -> SnapshotId:
        """Return a fixed snapshot ID."""
        return SnapshotId(1)

    def read(  # pylint: disable=unused-argument
        self, table: str, *, snapshot: SnapshotId, as_of: datetime
    ) -> pd.DataFrame:
        """Return the stored table, ignoring the cutoff."""
        return self._tables[table]


def test_window_hits_counts_inside_and_missing_stops() -> None:
    predicted = {"a": [PitWindow(0, 10, 20)], "b": [PitWindow(0, 10, 20), PitWindow(1, 40, 50)]}
    actual = {"a": [15], "b": [12, 99]}
    assert window_hits(predicted, actual) == (2, 3)


def test_window_hits_skips_drivers_without_actual_stops() -> None:
    assert window_hits({"a": [PitWindow(0, 10, 20)]}, {}) == (0, 0)


def test_calibration_measures_held_out_races() -> None:
    tables = build_tyres_world().tables
    reader = StubReader(dict(tables))
    result = calibrate_pit_windows(
        reader,
        SnapshotId(1),
        train_seasons=(2023,),
        holdout_seasons=(2024,),
        rules_dir=RULES_DIR,
    )
    assert result.stops > 0
    assert 0.0 <= result.share <= 1.0
    assert result.share > 0.5
    assert result.hits == sum(row.hit for row in result.rows)
    assert sum(stops for _hits, stops in result.by_circuit().values()) == result.stops


def test_report_renders_the_share_and_a_row_per_circuit() -> None:
    tables = build_tyres_world().tables
    result = calibrate_pit_windows(
        StubReader(dict(tables)),
        SnapshotId(1),
        train_seasons=(2023,),
        holdout_seasons=(2024,),
        rules_dir=RULES_DIR,
    )
    report = render_pit_windows(result)
    assert report.startswith("# Pit-window calibration")
    assert "| Circuit | Stops | Inside | Share |" in report
    assert "| monza |" in report


def test_empty_report_says_nothing_was_checked() -> None:
    report = render_pit_windows(PitWindowCalibration())
    assert "none" in report
    assert "0 stops" in report
