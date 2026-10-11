"""The Predictor port (P2-03).

Every model and baseline implements it, so any of them can stand in for another.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Protocol

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.ports.store import DataReader


class Predictor(Protocol):
    """A race forecasting model.

    Attributes:
        name: Registry name, such as ``"plackett_luce"``.
    """

    name: str

    @abstractmethod
    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Fit the model using only data available at or before ``cutoff``.

        Args:
            reader: Read-only access to the data store.
            snapshot: Snapshot to read from.
            cutoff: Latest moment whose data the model may see.
        """

    @abstractmethod
    def predict(self, event: EventKey) -> PositionMatrix:
        """Forecast one race.

        Args:
            event: The race weekend to forecast.

        Returns:
            Finishing-position probabilities for every driver.
        """
