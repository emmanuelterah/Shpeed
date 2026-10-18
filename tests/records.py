"""Builders and an in-memory double for prediction records, shared by the P8 tests."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.record import PredictionRecord, RecordScore
from f1sim.core.ports.publisher import PublishResult

CUTOFF = datetime(2024, 3, 23, 16, 0)
"""End of qualifying for the test race."""

LIGHTS_OUT = datetime(2024, 3, 24, 15, 0)
"""Scheduled start of the test race."""


def make_record(  # pylint: disable=too-many-arguments
    *,
    drivers: Sequence[str] = ("a", "b", "c"),
    season: int = 2024,
    round_: int = 3,
    git_commit: str = "abcdef0",
    snapshot: int = 7,
    favourite: float = 0.5,
) -> PredictionRecord:
    """Build a small valid record whose first driver is the favourite.

    Args:
        drivers: Driver IDs, in grid order.
        season: Championship year.
        round_: Round number.
        git_commit: Provenance commit.
        snapshot: Provenance snapshot ID.
        favourite: Probability that the first driver wins; changing it changes the forecast.

    Returns:
        The record, with no score.
    """
    n = len(drivers)
    p_dnf = np.full(n, 0.1)
    p_position = np.full((n, n), 0.9 / n)
    if n > 1:
        p_position[0] = (0.9 - favourite) / (n - 1)
        p_position[0, 0] = favourite
        p_position[1:, 0] = (0.9 - favourite) / (n - 1)
        rest = (0.9 - p_position[1:, 0]) / (n - 1)
        p_position[1:, 1:] = rest[:, np.newaxis]
    else:
        p_position[0, 0] = 0.9
    matrix = PositionMatrix(tuple(drivers), p_position, p_dnf)
    entries = tuple(
        RaceEntry(driver_id=d, team_id=f"team_{d}", grid=i) for i, d in enumerate(drivers, 1)
    )
    return PredictionRecord.from_matrix(
        matrix,
        entries,
        season=season,
        round_=round_,
        data_cutoff=CUTOFF,
        snapshot_id=SnapshotId(snapshot),
        model_version="plackett_luce-0.0.0",
        git_commit=git_commit,
        n_sims=200,
        seed=11,
        event_name="Test Grand Prix",
    )


def make_score(positions: Sequence[int | None], *, rps: float = 0.2) -> RecordScore:
    """A score with fixed metrics and the given finishing positions.

    Args:
        positions: Each driver's finishing position, None for a DNF.
        rps: Ranked probability score to report.

    Returns:
        The score.
    """
    return RecordScore(
        brier_win=0.1,
        brier_podium=0.15,
        brier_top10=0.12,
        log_loss=1.5,
        rps=rps,
        scored_at=datetime(2024, 3, 24, 18, 0),
        positions=tuple(positions),
    )


class MemoryRecords:
    """In-memory publisher and record reader, the test double for the publishing ports."""

    def __init__(self) -> None:
        """Start with nothing published."""
        self.stored: dict[EventKey, dict[str, Any]] = {}
        self.writes = 0

    def load(self, event: EventKey) -> PredictionRecord | None:
        """Read back a published record.

        Args:
            event: The race weekend.

        Returns:
            The record, or None when nothing is published.
        """
        found = self.stored.get(event)
        return None if found is None else PredictionRecord.from_mapping(found)

    def publish(self, record: Mapping[str, Any]) -> PublishResult:
        """Store a record, reporting a no-op when it is unchanged.

        Args:
            record: A JSON-ready record.

        Returns:
            Where it lives and whether it changed.
        """
        event = EventKey(int(record["season"]), int(record["round"]))
        location = f"memory://{event.season}/{event.round:02d}"
        if self.stored.get(event) == dict(record):
            return PublishResult(location, changed=False)
        self.stored[event] = dict(record)
        self.writes += 1
        return PublishResult(location, changed=True)
