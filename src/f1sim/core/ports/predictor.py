"""The Predictor port (P2-03).

Every model and baseline implements it, so any of them can stand in for another.
"""

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.ports.store import DataReader


class Predictor(Protocol):
    """A race forecasting model.

    A predictor is fitted once per race and then predicts that race, so it may keep
    whatever state it learned in ``fit``.

    Attributes:
        name: Registry name, such as ``"plackett_luce"``.
    """

    name: str

    @abstractmethod
    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Fit the model using only data available at or before ``cutoff``.

        Args:
            reader: Read-only access to the data store. In backtests it raises
                ``LeakageError`` on any read past ``cutoff``.
            snapshot: Snapshot to read from.
            cutoff: Latest moment whose data the model may see.
        """

    @abstractmethod
    def predict(self, event: EventKey, entries: Sequence[RaceEntry]) -> PositionMatrix:
        """Forecast one race.

        Args:
            event: The race weekend to forecast.
            entries: The drivers starting the race, with their expected grid positions.

        Returns:
            Finishing-position probabilities, with one row per entry in ``entries`` order.
        """


PredictorFactory = Callable[[int], Predictor]
"""Builds a fresh, unfitted predictor from a random seed."""


@dataclass(frozen=True)
class ModelSpec:
    """Everything a backtest needs to run one model.

    Attributes:
        name: Registry name, such as ``"elo"``.
        factory: Builds a fresh, unfitted predictor from a seed.
        tables: Tables the model reads during ``fit``.
    """

    name: str
    factory: PredictorFactory
    tables: tuple[str, ...] = ("results",)
