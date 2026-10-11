"""Jolpica-F1 source adapter (P1-03): schedule, results, qualifying, sprints and pit stops.

Jolpica-F1 is the community successor to the Ergast API and serves the same JSON format.
It owns the ``events``, ``results`` and ``pit_stops`` tables, and its driver and
constructor IDs are Shpeed's canonical keys (P1-05).
"""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from f1sim.adapters.sources.http import HttpClient
from f1sim.adapters.sources.parsing import (
    SESSION_DURATION,
    parse_float,
    parse_int,
    parse_time_s,
    parse_utc,
)
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType
from f1sim.core.errors import DataError

BASE_URL = "https://api.jolpi.ca/ergast/f1"

_RESULT_KEYS: dict[SessionType, tuple[str, str]] = {
    SessionType.RACE: ("results", "Results"),
    SessionType.QUALIFYING: ("qualifying", "QualifyingResults"),
    SessionType.SPRINT: ("sprint", "SprintResults"),
}


class JolpicaAdapter:
    name = "jolpica"

    def __init__(
        self,
        http: HttpClient,
        *,
        clock: Callable[[], datetime],
        base_url: str = BASE_URL,
        page_size: int = 100,
    ) -> None:
        self._http = http
        self._clock = clock
        self._base = base_url.rstrip("/")
        self._page = page_size
        self._schedules: dict[int, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ SourceAdapter
    def sessions(self, season: int) -> Sequence[SessionKey]:
        """Race, qualifying and sprint sessions of ``season`` that have finished."""
        now = self._clock()
        found: list[SessionKey] = []
        for race in self._schedule(season):
            event = EventKey(season, int(race["round"]))
            for session in (SessionType.QUALIFYING, SessionType.SPRINT, SessionType.RACE):
                start = _session_start(race, session)
                if start is not None and start + SESSION_DURATION[session.value] <= now:
                    found.append(SessionKey(event, session))
        return found

    def fetch(self, session: SessionKey) -> Mapping[str, pd.DataFrame]:
        race = self._race(session.event)
        start = _session_start(race, session.session)
        if start is None:
            raise DataError(f"{session} is not on the {session.event.season} schedule")
        available_at = start + SESSION_DURATION[session.session.value]
        tables = {"results": self._results(session, available_at)}
        if session.session is SessionType.RACE:
            tables["pit_stops"] = self._pit_stops(session, available_at)
            tables["events"] = _event_row(race, session.event.season)
        return tables

    # ------------------------------------------------------------------ helpers
    def _schedule(self, season: int) -> list[dict[str, Any]]:
        if season not in self._schedules:
            live = season >= self._clock().year
            self._schedules[season] = self._paged(f"{season}", "Races", cache=not live)
        return self._schedules[season]

    def _race(self, event: EventKey) -> dict[str, Any]:
        for race in self._schedule(event.season):
            if int(race["round"]) == event.round:
                return race
        raise DataError(f"No round {event.round} in the {event.season} schedule")

    def _paged(self, path: str, key: str, *, cache: bool = True) -> list[dict[str, Any]]:
        """Fetch every page of ``RaceTable.Races`` for a path, merging nested lists."""
        items: list[dict[str, Any]] = []
        offset = 0
        while True:
            data = self._http.get_json(
                f"{self._base}/{path}.json", {"limit": self._page, "offset": offset}, cache=cache
            )
            mr = data["MRData"]
            races: list[dict[str, Any]] = mr["RaceTable"]["Races"]
            if key == "Races":
                items.extend(races)
            else:
                for race in races:
                    items.extend(race.get(key, []))
            offset += self._page
            if offset >= int(mr["total"]):
                return items

    def _results(self, session: SessionKey, available_at: datetime) -> pd.DataFrame:
        path, key = _RESULT_KEYS[session.session]
        e = session.event
        rows = [
            _result_row(r, session, available_at)
            for r in self._paged(f"{e.season}/{e.round}/{path}", key)
        ]
        return pd.DataFrame(rows)

    def _pit_stops(self, session: SessionKey, available_at: datetime) -> pd.DataFrame:
        e = session.event
        rows = [
            {
                "season": e.season,
                "round": e.round,
                "session": session.session.value,
                "driver_id": p["driverId"],
                "stop": parse_int(p.get("stop")),
                "lap": parse_int(p.get("lap")),
                "duration_s": parse_time_s(p.get("duration")),
                "available_at": available_at,
            }
            for p in self._paged(f"{e.season}/{e.round}/pitstops", "PitStops")
        ]
        return pd.DataFrame(rows)


def _session_start(race: dict[str, Any], session: SessionType) -> datetime | None:
    race_start = parse_utc(race.get("date"), race.get("time"))
    if race_start is None:
        return None
    if session is SessionType.RACE:
        return race_start
    if session is SessionType.SPRINT:
        sprint = race.get("Sprint")
        return parse_utc(sprint.get("date"), sprint.get("time")) if sprint else None
    quali = race.get("Qualifying")
    if quali:
        return parse_utc(quali.get("date"), quali.get("time"))
    return race_start - timedelta(days=1)  # older seasons publish only the race date


def _event_row(race: dict[str, Any], season: int) -> pd.DataFrame:
    location = race["Circuit"].get("Location", {})
    fmt = "sprint" if race.get("Sprint") else "conventional"
    return pd.DataFrame(
        [
            {
                "season": season,
                "round": int(race["round"]),
                "event_name": race.get("raceName"),
                "circuit_id": race["Circuit"].get("circuitId"),
                "locality": location.get("locality"),
                "country": location.get("country"),
                "event_date": parse_utc(race.get("date"), race.get("time")),
                "event_format": fmt,
                "available_at": datetime(season, 1, 1),  # schedules are published in advance
            }
        ]
    )


def _result_row(r: dict[str, Any], session: SessionKey, available_at: datetime) -> dict[str, Any]:
    driver, team = r["Driver"], r["Constructor"]
    text = str(r.get("positionText", r.get("position", "")))
    time_info: dict[str, Any] = r.get("Time") or {}
    millis = time_info.get("millis")
    return {
        "season": session.event.season,
        "round": session.event.round,
        "session": session.session.value,
        "driver_id": driver["driverId"],
        "driver_code": driver.get("code"),
        "driver_number": parse_int(r.get("number")),
        "team_id": team["constructorId"],
        "team_name": team.get("name"),
        "grid": parse_int(r.get("grid")),
        "position": parse_int(r.get("position")) if text.isdigit() else None,
        "position_text": text,
        "status": r.get("status"),
        "points": parse_float(r.get("points")),
        "laps": parse_int(r.get("laps")),
        "time_s": int(millis) / 1000 if millis else None,
        "q1_s": parse_time_s(r.get("Q1")),
        "q2_s": parse_time_s(r.get("Q2")),
        "q3_s": parse_time_s(r.get("Q3")),
        "available_at": available_at,
    }
