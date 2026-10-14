"""The fitted-state store port (P3-10).

Fitted models save their posterior after each race weekend, so the next weekend's update
starts from it instead of refitting from the first race.
"""

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from f1sim.core.domain.entities import EventKey, SnapshotId


@dataclass(frozen=True, slots=True)
class StateKey:
    """Identifies one saved state.

    Attributes:
        model: Model name, such as ``"pace"``.
        version: Model version; a state from another version is never reused.
        event: The last race weekend the state includes.
        snapshot: Data snapshot the state was fitted from.
    """

    model: str
    version: str
    event: EventKey
    snapshot: SnapshotId


class FittedStateStore(Protocol):
    """Saves and loads fitted model states atomically."""

    @abstractmethod
    def save(self, key: StateKey, state: Mapping[str, Any]) -> None:
        """Save a state; a crash mid-write leaves the previous file intact.

        Args:
            key: What the state is.
            state: JSON-compatible state.
        """

    @abstractmethod
    def load_latest(
        self, model: str, version: str, event: EventKey, max_snapshot: SnapshotId
    ) -> tuple[SnapshotId, dict[str, Any]] | None:
        """Load the newest state for one weekend from a snapshot no newer than given.

        Args:
            model: Model name.
            version: Model version.
            event: The weekend the state must end at.
            max_snapshot: Newest snapshot allowed.

        Returns:
            ``(snapshot, state)``, or None when nothing matches.
        """
