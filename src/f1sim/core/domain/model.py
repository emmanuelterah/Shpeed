"""Frozen value objects for the canonical F1 entities (P1-01).

Each entity mirrors one canonical table row (see ``f1sim.adapters.store.tables``) and can
build itself from that row with ``from_row``. Adapters, reports and models therefore share
one typed, immutable view of the data instead of passing bare dicts or DataFrames around.

``Season`` and ``Session`` describe the calendar, which has no canonical table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from f1sim.core.domain.entities import EventKey, SessionKey, SessionType


def _missing(value: Any) -> bool:
    """Whether a row value is null, NaN or NaT."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def _optional_text(row: Mapping[str, Any], key: str) -> str | None:
    value = row[key]
    return None if _missing(value) else str(value)


def _optional_int(row: Mapping[str, Any], key: str) -> int | None:
    value = row[key]
    return None if _missing(value) else int(value)


def _optional_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row[key]
    return None if _missing(value) else float(value)


def _optional_bool(row: Mapping[str, Any], key: str) -> bool | None:
    value = row[key]
    return None if _missing(value) else bool(value)


def _optional_when(row: Mapping[str, Any], key: str) -> datetime | None:
    value = row[key]
    return None if _missing(value) else pd.Timestamp(value).to_pydatetime()


def _required_text(row: Mapping[str, Any], key: str) -> str:
    return str(row[key])


def _required_int(row: Mapping[str, Any], key: str) -> int:
    return int(row[key])


def _required_when(row: Mapping[str, Any], key: str) -> datetime:
    return pd.Timestamp(row[key]).to_pydatetime()


def _session_key(row: Mapping[str, Any]) -> SessionKey:
    event = EventKey(_required_int(row, "season"), _required_int(row, "round"))
    return SessionKey(event, SessionType(_required_text(row, "session")))


@dataclass(frozen=True, slots=True)
class Season:
    """One championship year.

    Attributes:
        year: The championship year, for example 2024.
    """

    year: int


@dataclass(frozen=True, slots=True)
class Event:
    """One race weekend (an ``events`` row).

    Attributes:
        season: Championship year.
        round: Round number within the season, starting at 1.
        name: Event name, for example ``"Bahrain Grand Prix"``.
        circuit_id: Jolpica circuit ID.
        locality: Host town or city.
        country: Host country.
        date: Scheduled race start, naive UTC.
        event_format: ``conventional`` or ``sprint``.
        available_at: When the row became knowable (normally publication of the calendar).
    """

    season: int
    round: int
    name: str | None
    circuit_id: str | None
    locality: str | None
    country: str | None
    date: datetime
    event_format: str | None
    available_at: datetime

    @property
    def key(self) -> EventKey:
        """The weekend's key."""
        return EventKey(self.season, self.round)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Event:
        """Build an event from an ``events`` row.

        Args:
            row: Canonical ``events`` row keyed by column name.

        Returns:
            The event described by the row.
        """
        return cls(
            season=_required_int(row, "season"),
            round=_required_int(row, "round"),
            name=_optional_text(row, "event_name"),
            circuit_id=_optional_text(row, "circuit_id"),
            locality=_optional_text(row, "locality"),
            country=_optional_text(row, "country"),
            date=_required_when(row, "event_date"),
            event_format=_optional_text(row, "event_format"),
            available_at=_required_when(row, "available_at"),
        )


@dataclass(frozen=True, slots=True)
class Session:
    """One session of a race weekend.

    Attributes:
        key: The weekend and which session.
        start_utc: Planned start time, naive UTC.
        available_at: When the session's data became knowable (normally the end).
    """

    key: SessionKey
    start_utc: datetime
    available_at: datetime


@dataclass(frozen=True, slots=True)
class Driver:
    """A driver, keyed by the canonical Jolpica driver ID.

    Attributes:
        driver_id: Canonical driver ID, for example ``"max_verstappen"``.
        code: Three-letter code, for example ``"VER"``.
        number: Permanent car number.
    """

    driver_id: str
    code: str | None = None
    number: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Driver:
        """Build a driver from a ``results`` row.

        Args:
            row: Canonical ``results`` row keyed by column name.

        Returns:
            The driver described by the row.
        """
        return cls(
            driver_id=_required_text(row, "driver_id"),
            code=_optional_text(row, "driver_code"),
            number=_optional_int(row, "driver_number"),
        )


@dataclass(frozen=True, slots=True)
class Team:
    """A constructor, keyed by the canonical Jolpica constructor ID.

    Attributes:
        team_id: Canonical constructor ID, for example ``"red_bull"``.
        name: Display name, for example ``"Red Bull"``.
    """

    team_id: str
    name: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Team:
        """Build a team from a ``results`` row.

        Args:
            row: Canonical ``results`` row keyed by column name.

        Returns:
            The team described by the row.
        """
        return cls(team_id=_required_text(row, "team_id"), name=_optional_text(row, "team_name"))


# A lap mirrors the ``laps`` table row, so it legitimately has many fields.
@dataclass(frozen=True, slots=True)
class Lap:  # pylint: disable=too-many-instance-attributes
    """One timed lap (a ``laps`` row).

    Attributes:
        key: The session the lap belongs to.
        driver_id: Canonical driver ID.
        driver_code: Three-letter driver code.
        team_id: Canonical team ID.
        lap_number: Lap number within the session, starting at 1.
        lap_time_s: Lap time in seconds, or None for an untimed lap.
        sector1_s: First sector time in seconds.
        sector2_s: Second sector time in seconds.
        sector3_s: Third sector time in seconds.
        stint: Stint number within the session.
        compound: Tyre compound, for example ``"MEDIUM"``.
        tyre_life: Laps completed on the current tyre set.
        fresh_tyre: Whether the tyre set was new at the start of the stint.
        pit_in: Whether the car entered the pit lane during the lap.
        pit_out: Whether the car left the pit lane during the lap.
        track_status: FastF1 track status code for the lap.
        position: Track position at the end of the lap.
        is_accurate: Whether FastF1 considers the lap time accurate.
        deleted: Whether the lap time was deleted.
        lap_start_utc: UTC time the lap started.
        clean_lap: Whether the lap reflects real pace (P1-08).
        available_at: When the row became knowable.
    """

    key: SessionKey
    driver_id: str
    driver_code: str | None
    team_id: str | None
    lap_number: int
    lap_time_s: float | None
    sector1_s: float | None
    sector2_s: float | None
    sector3_s: float | None
    stint: int | None
    compound: str | None
    tyre_life: float | None
    fresh_tyre: bool | None
    pit_in: bool | None
    pit_out: bool | None
    track_status: str | None
    position: int | None
    is_accurate: bool | None
    deleted: bool | None
    lap_start_utc: datetime | None
    clean_lap: bool | None
    available_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Lap:
        """Build a lap from a ``laps`` row.

        Args:
            row: Canonical ``laps`` row keyed by column name.

        Returns:
            The lap described by the row.
        """
        return cls(
            key=_session_key(row),
            driver_id=_required_text(row, "driver_id"),
            driver_code=_optional_text(row, "driver_code"),
            team_id=_optional_text(row, "team_id"),
            lap_number=_required_int(row, "lap_number"),
            lap_time_s=_optional_float(row, "lap_time_s"),
            sector1_s=_optional_float(row, "sector1_s"),
            sector2_s=_optional_float(row, "sector2_s"),
            sector3_s=_optional_float(row, "sector3_s"),
            stint=_optional_int(row, "stint"),
            compound=_optional_text(row, "compound"),
            tyre_life=_optional_float(row, "tyre_life"),
            fresh_tyre=_optional_bool(row, "fresh_tyre"),
            pit_in=_optional_bool(row, "pit_in"),
            pit_out=_optional_bool(row, "pit_out"),
            track_status=_optional_text(row, "track_status"),
            position=_optional_int(row, "position"),
            is_accurate=_optional_bool(row, "is_accurate"),
            deleted=_optional_bool(row, "deleted"),
            lap_start_utc=_optional_when(row, "lap_start_utc"),
            clean_lap=_optional_bool(row, "clean_lap"),
            available_at=_required_when(row, "available_at"),
        )


@dataclass(frozen=True, slots=True)
class Stint:
    """One driver's stint on one set of tyres (a ``stints`` row).

    Attributes:
        key: The session the stint belongs to.
        driver_id: Canonical driver ID.
        stint: Stint number within the session.
        compound: Tyre compound, for example ``"MEDIUM"``.
        start_lap: First lap of the stint.
        end_lap: Last lap of the stint.
        available_at: When the row became knowable.
    """

    key: SessionKey
    driver_id: str
    stint: int
    compound: str | None
    start_lap: int
    end_lap: int
    available_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Stint:
        """Build a stint from a ``stints`` row.

        Args:
            row: Canonical ``stints`` row keyed by column name.

        Returns:
            The stint described by the row.
        """
        return cls(
            key=_session_key(row),
            driver_id=_required_text(row, "driver_id"),
            stint=_required_int(row, "stint"),
            compound=_optional_text(row, "compound"),
            start_lap=_required_int(row, "start_lap"),
            end_lap=_required_int(row, "end_lap"),
            available_at=_required_when(row, "available_at"),
        )


@dataclass(frozen=True, slots=True)
class PitStop:
    """One pit stop (a ``pit_stops`` row).

    Attributes:
        key: The session the stop belongs to.
        driver_id: Canonical driver ID.
        stop: Stop number for the driver within the session.
        lap: Lap the stop happened on.
        duration_s: Stationary time in seconds.
        available_at: When the row became knowable.
    """

    key: SessionKey
    driver_id: str
    stop: int
    lap: int | None
    duration_s: float | None
    available_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> PitStop:
        """Build a pit stop from a ``pit_stops`` row.

        Args:
            row: Canonical ``pit_stops`` row keyed by column name.

        Returns:
            The pit stop described by the row.
        """
        return cls(
            key=_session_key(row),
            driver_id=_required_text(row, "driver_id"),
            stop=_required_int(row, "stop"),
            lap=_optional_int(row, "lap"),
            duration_s=_optional_float(row, "duration_s"),
            available_at=_required_when(row, "available_at"),
        )


@dataclass(frozen=True, slots=True)
class TrackStatus:
    """One track-status change (a ``track_status`` row).

    Attributes:
        key: The session the change belongs to.
        time_utc: UTC time of the change.
        status: FastF1 status code (1 green, 2 yellow, 4 safety car, 5 red flag and so on).
        message: Human-readable status message.
        available_at: When the row became knowable.
    """

    key: SessionKey
    time_utc: datetime
    status: str
    message: str | None
    available_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> TrackStatus:
        """Build a track status from a ``track_status`` row.

        Args:
            row: Canonical ``track_status`` row keyed by column name.

        Returns:
            The track status described by the row.
        """
        return cls(
            key=_session_key(row),
            time_utc=_required_when(row, "time_utc"),
            status=_required_text(row, "status"),
            message=_optional_text(row, "message"),
            available_at=_required_when(row, "available_at"),
        )


@dataclass(frozen=True, slots=True)
class RaceControl:
    """One race-control message (a ``race_control`` row).

    Attributes:
        key: The session the message belongs to.
        time_utc: UTC time the message was issued.
        category: Message category, for example ``"Flag"`` or ``"SafetyCar"``.
        message: Full message text.
        flag: Flag colour when the category is a flag.
        scope: Affected scope, for example ``"Track"`` or ``"Driver"``.
        sector: Affected sector.
        driver_number: Affected car number.
        lap: Lap the message refers to.
        available_at: When the row became knowable.
    """

    key: SessionKey
    time_utc: datetime
    category: str
    message: str
    flag: str | None
    scope: str | None
    sector: int | None
    driver_number: int | None
    lap: int | None
    available_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> RaceControl:
        """Build a race-control message from a ``race_control`` row.

        Args:
            row: Canonical ``race_control`` row keyed by column name.

        Returns:
            The message described by the row.
        """
        return cls(
            key=_session_key(row),
            time_utc=_required_when(row, "time_utc"),
            category=_required_text(row, "category"),
            message=_required_text(row, "message"),
            flag=_optional_text(row, "flag"),
            scope=_optional_text(row, "scope"),
            sector=_optional_int(row, "sector"),
            driver_number=_optional_int(row, "driver_number"),
            lap=_optional_int(row, "lap"),
            available_at=_required_when(row, "available_at"),
        )


# A classification mirrors the ``results`` table row, so it legitimately has many fields.
@dataclass(frozen=True, slots=True)
class Result:  # pylint: disable=too-many-instance-attributes
    """One driver's classification in a session (a ``results`` row).

    Attributes:
        key: The session the classification belongs to.
        driver_id: Canonical driver ID.
        driver_code: Three-letter driver code.
        driver_number: Car number.
        team_id: Canonical team ID.
        team_name: Team display name.
        grid: Starting position (0 for a pit-lane start).
        position: Finishing position, or None for a retirement.
        position_text: Position as published, for example ``"R"`` for a retirement.
        status: Finishing status, for example ``"Finished"`` or ``"Accident"``.
        points: Championship points scored.
        laps: Laps completed.
        time_s: Total race time or gap in seconds.
        q1_s: Q1 best time in seconds.
        q2_s: Q2 best time in seconds.
        q3_s: Q3 best time in seconds.
        available_at: When the row became knowable.
    """

    key: SessionKey
    driver_id: str
    driver_code: str | None
    driver_number: int | None
    team_id: str | None
    team_name: str | None
    grid: int | None
    position: int | None
    position_text: str | None
    status: str | None
    points: float | None
    laps: int | None
    time_s: float | None
    q1_s: float | None
    q2_s: float | None
    q3_s: float | None
    available_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Result:
        """Build a classification from a ``results`` row.

        Args:
            row: Canonical ``results`` row keyed by column name.

        Returns:
            The classification described by the row.
        """
        return cls(
            key=_session_key(row),
            driver_id=_required_text(row, "driver_id"),
            driver_code=_optional_text(row, "driver_code"),
            driver_number=_optional_int(row, "driver_number"),
            team_id=_optional_text(row, "team_id"),
            team_name=_optional_text(row, "team_name"),
            grid=_optional_int(row, "grid"),
            position=_optional_int(row, "position"),
            position_text=_optional_text(row, "position_text"),
            status=_optional_text(row, "status"),
            points=_optional_float(row, "points"),
            laps=_optional_int(row, "laps"),
            time_s=_optional_float(row, "time_s"),
            q1_s=_optional_float(row, "q1_s"),
            q2_s=_optional_float(row, "q2_s"),
            q3_s=_optional_float(row, "q3_s"),
            available_at=_required_when(row, "available_at"),
        )


@dataclass(frozen=True, slots=True)
class Standing:
    """One championship standing after a round (a ``standings`` row).

    Attributes:
        event: The weekend the standings are taken after.
        scope: ``"driver"`` or ``"constructor"``.
        subject_id: Canonical driver or team ID.
        position: Championship position.
        points: Points scored so far.
        wins: Race wins so far.
        available_at: When the row became knowable.
    """

    event: EventKey
    scope: str
    subject_id: str
    position: int | None
    points: float | None
    wins: int | None
    available_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Standing:
        """Build a standing from a ``standings`` row.

        Args:
            row: Canonical ``standings`` row keyed by column name.

        Returns:
            The standing described by the row.
        """
        return cls(
            event=EventKey(_required_int(row, "season"), _required_int(row, "round")),
            scope=_required_text(row, "scope"),
            subject_id=_required_text(row, "subject_id"),
            position=_optional_int(row, "position"),
            points=_optional_float(row, "points"),
            wins=_optional_int(row, "wins"),
            available_at=_required_when(row, "available_at"),
        )
