"""Predictor port (P2-03). Every model and baseline implements it, so all are substitutable."""

from datetime import datetime
from typing import Protocol

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.ports.store import DataReader


class Predictor(Protocol):
    name: str

    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Fit using only data at or before ``cutoff``."""
        ...

    def predict(self, event: EventKey) -> PositionMatrix: ...
