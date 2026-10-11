"""Identifiers for seasons, events, sessions and data snapshots (P1-01).

The key types are frozen dataclasses, so they are safe to share, compare and use in sets.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

SnapshotId = NewType("SnapshotId", int)
"""A DuckLake snapshot version (ADR-0002)."""


class SessionType(StrEnum):
    """The sessions of a race weekend, valued by their short codes."""

    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    SPRINT_QUALIFYING = "SQ"
    SPRINT = "S"
    QUALIFYING = "Q"
    RACE = "R"


@dataclass(frozen=True, slots=True)
class EventKey:
    """One race weekend.

    Attributes:
        season: Championship year, for example 2024.
        round: Round number within the season, starting at 1.
    """

    season: int
    round: int


@dataclass(frozen=True, slots=True)
class SessionKey:
    """One session within a race weekend.

    Attributes:
        event: The race weekend.
        session: Which session of that weekend.
    """

    event: EventKey
    session: SessionType
