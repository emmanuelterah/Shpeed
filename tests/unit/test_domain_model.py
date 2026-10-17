"""Domain entities build themselves from canonical rows (P1-01)."""

from dataclasses import FrozenInstanceError
from datetime import datetime
from typing import Any, cast

import pandas as pd
import pytest

from f1sim.adapters.store.schemas import validate
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType
from f1sim.core.domain.model import (
    Driver,
    Event,
    Lap,
    PitStop,
    RaceControl,
    Result,
    Season,
    Session,
    Standing,
    Stint,
    Team,
    TrackStatus,
)
from tests.support import lap_rows, result_rows

T0 = datetime(2025, 3, 16, 4, 0)
KEY = SessionKey(EventKey(2025, 1), SessionType.RACE)

EVENT = {
    "season": 2025,
    "round": 1,
    "event_name": "Australian Grand Prix",
    "circuit_id": "albert_park",
    "locality": "Melbourne",
    "country": "Australia",
    "event_date": T0,
    "event_format": "conventional",
    "available_at": datetime(2024, 1, 1),
}
PIT_STOP = {
    "season": 2025,
    "round": 1,
    "session": "R",
    "driver_id": "lando_norris",
    "stop": 1,
    "lap": 3,
    "duration_s": 2.4,
    "available_at": T0,
}
TRACK_STATUS = {
    "season": 2025,
    "round": 1,
    "session": "R",
    "time_utc": T0,
    "status": "1",
    "message": "AllClear",
    "available_at": T0,
}
STINT = {
    "season": 2025,
    "round": 1,
    "session": "R",
    "driver_id": "max_verstappen",
    "stint": 1,
    "compound": "MEDIUM",
    "start_lap": 1,
    "end_lap": 3,
    "available_at": T0,
}
RACE_CONTROL = {
    "season": 2025,
    "round": 1,
    "session": "R",
    "time_utc": T0,
    "category": "Flag",
    "message": "GREEN LIGHT - PIT EXIT OPEN",
    "flag": "GREEN",
    "scope": "Track",
    "sector": None,
    "driver_number": None,
    "lap": None,
    "available_at": T0,
}
STANDING = {
    "season": 2025,
    "round": 1,
    "scope": "driver",
    "subject_id": "max_verstappen",
    "position": 1,
    "points": 25.0,
    "wins": 1,
    "available_at": T0,
}


def _row(table: str, values: Any) -> dict[str, Any]:
    """Validate a raw row so the entity reads the types the store really writes."""
    return cast(dict[str, Any], validate(table, pd.DataFrame([values])).iloc[0].to_dict())


def test_event_from_row() -> None:
    event = Event.from_row(_row("events", EVENT))
    assert event.key == EventKey(2025, 1)
    assert event.circuit_id == "albert_park"
    assert event.date == T0
    assert event.available_at == datetime(2024, 1, 1)


def test_driver_and_team_from_a_results_row() -> None:
    row = _row("results", result_rows().iloc[0].to_dict())
    assert Driver.from_row(row) == Driver("max_verstappen", "MAX", 1)
    assert Team.from_row(row) == Team("red_bull", "Red_Bull")


def test_lap_from_row() -> None:
    lap = Lap.from_row(_row("laps", lap_rows().iloc[0].to_dict()))
    assert lap.key == KEY
    assert lap.lap_number == 1
    assert lap.stint == 1
    assert lap.compound == "MEDIUM"
    assert lap.tyre_life == 1.0
    assert lap.pit_in is False
    assert lap.clean_lap is False  # lap 1 is never clean


def test_result_keeps_a_retirement_and_missing_times() -> None:
    values = result_rows().iloc[1].to_dict()
    values.update(position=None, position_text="R")
    result = Result.from_row(_row("results", values))
    assert result.key == KEY
    assert result.position is None
    assert result.position_text == "R"
    assert result.q1_s is None


def test_stint_pit_stop_track_status_and_race_control() -> None:
    stint = Stint.from_row(_row("stints", STINT))
    assert (stint.start_lap, stint.end_lap) == (1, 3)
    assert PitStop.from_row(_row("pit_stops", PIT_STOP)).duration_s == 2.4
    assert TrackStatus.from_row(_row("track_status", TRACK_STATUS)).message == "AllClear"
    control = RaceControl.from_row(_row("race_control", RACE_CONTROL))
    assert control.category == "Flag"
    assert control.flag == "GREEN"
    assert control.sector is None


def test_standing_from_row() -> None:
    standing = Standing.from_row(_row("standings", STANDING))
    assert standing.event == EventKey(2025, 1)
    assert standing.scope == "driver"
    assert standing.points == 25.0


def test_season_and_session_are_plain_values() -> None:
    assert Season(2025).year == 2025
    assert Session(KEY, T0, T0).key == KEY


def test_entities_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        Season(2025).year = 2024  # type: ignore[misc]
