"""Identifiers and core entities (P1-01). Frozen so they are safe to share and hash."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

SnapshotId = NewType("SnapshotId", int)
"""A DuckLake snapshot version (ADR-0002)."""


class SessionType(StrEnum):
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    SPRINT_QUALIFYING = "SQ"
    SPRINT = "S"
    QUALIFYING = "Q"
    RACE = "R"


@dataclass(frozen=True, slots=True)
class EventKey:
    """One race weekend."""

    season: int
    round: int


@dataclass(frozen=True, slots=True)
class SessionKey:
    """One session within a race weekend."""

    event: EventKey
    session: SessionType
