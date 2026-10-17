"""Jolpica-F1 adapter, run against recorded responses served with real pagination."""

from datetime import datetime

import pytest

from f1sim.adapters.sources.http import HttpClient, RateLimiter
from f1sim.adapters.sources.jolpica_adapter import JolpicaAdapter
from f1sim.adapters.store.schemas import validate
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType
from tests.fixtures.jolpica_transport import jolpica_transport

pytestmark = pytest.mark.integration
R1 = EventKey(2024, 1)


def _adapter(now: datetime, page_size: int = 100, log: list[str] | None = None) -> JolpicaAdapter:
    limiter = RateLimiter(per_second=1e6, burst=1000, per_hour=10**6, sleep=lambda s: None)
    http = HttpClient(cache_dir=None, limiter=limiter, transport=jolpica_transport(log))
    return JolpicaAdapter(http, clock=lambda: now, page_size=page_size)


def test_lists_only_finished_sessions_including_sprints() -> None:
    adapter = _adapter(datetime(2024, 5, 5, 0, 0))  # Miami race day, before the race
    ordered = adapter.sessions(2024)
    sessions = set(ordered)
    assert SessionKey(R1, SessionType.RACE) in sessions
    assert SessionKey(EventKey(2024, 6), SessionType.SPRINT) in sessions
    assert SessionKey(EventKey(2024, 6), SessionType.QUALIFYING) in sessions
    assert SessionKey(EventKey(2024, 6), SessionType.RACE) not in sessions
    assert all(s.event.round != 24 for s in sessions)
    sprint = ordered.index(SessionKey(EventKey(2024, 6), SessionType.SPRINT))
    quali = ordered.index(SessionKey(EventKey(2024, 6), SessionType.QUALIFYING))
    assert sprint < quali  # the sprint runs before qualifying on a sprint weekend


def test_race_fetch_returns_valid_results_pit_stops_and_event() -> None:
    log: list[str] = []
    adapter = _adapter(datetime(2025, 1, 1), page_size=3, log=log)
    tables = adapter.fetch(SessionKey(R1, SessionType.RACE))
    results = validate("results", tables["results"])
    stops = validate("pit_stops", tables["pit_stops"])
    events = validate("events", tables["events"])

    winner = results.set_index("driver_id").loc["max_verstappen"].to_dict()
    assert winner["position"] == 1
    assert winner["time_s"] == pytest.approx(5504.742)
    retired = results.set_index("driver_id").loc["norris"].to_dict()
    assert retired["position_text"] == "R"
    assert retired["position"] is None or str(retired["position"]) == "<NA>"
    assert retired["status"] == "Accident"

    assert len(stops) == 5  # both pages merged
    assert "2024/1/pitstops@3" in log
    assert stops.set_index(["driver_id", "stop"]).loc[("norris", 1), "duration_s"] == pytest.approx(
        62.345
    )

    assert events.loc[0, "event_name"] == "Bahrain Grand Prix"
    # Race results become available at the end of the race, not before
    assert results["available_at"].iloc[0] == datetime(2024, 3, 2, 18, 0)


def test_race_fetch_returns_driver_and_constructor_standings() -> None:
    adapter = _adapter(datetime(2025, 1, 1))
    tables = adapter.fetch(SessionKey(R1, SessionType.RACE))
    standings = validate("standings", tables["standings"])

    assert set(standings["scope"]) == {"driver", "constructor"}
    leader = standings.loc[(standings["scope"] == "driver") & (standings["position"] == 1)].iloc[0]
    assert leader["subject_id"] == "max_verstappen"
    assert leader["points"] == pytest.approx(26.0)
    assert leader["wins"] == 1
    assert len(standings.loc[standings["scope"] == "constructor"]) == 10


def test_qualifying_times_are_parsed() -> None:
    adapter = _adapter(datetime(2025, 1, 1))
    quali = validate("results", adapter.fetch(SessionKey(R1, SessionType.QUALIFYING))["results"])
    pole = quali.set_index("driver_id").loc["max_verstappen"].to_dict()
    assert pole["q3_s"] == pytest.approx(89.179)
    assert quali["session"].unique().tolist() == ["Q"]


def test_sprint_results() -> None:
    adapter = _adapter(datetime(2025, 1, 1))
    sprint = validate(
        "results", adapter.fetch(SessionKey(EventKey(2024, 6), SessionType.SPRINT))["results"]
    )
    assert sprint["points"].tolist() == [8.0, 7.0]
