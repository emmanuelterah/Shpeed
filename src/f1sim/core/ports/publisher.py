"""The publishing ports (P8-02, P8-03): where records go and how they are read back.

The core owns the publishing policy (validate, compare with what is already published, lock
at lights out); the adapters own the file system, the JSON Schema validator and git. Reading
and writing are separate ports, so a component that only needs to look up a record cannot
write one (interface segregation).
"""

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from f1sim.core.domain.entities import EventKey
from f1sim.core.domain.record import PredictionRecord


@dataclass(frozen=True)
class PublishResult:
    """What a publisher did with a record.

    Attributes:
        location: Where the record lives, such as a file path.
        changed: Whether the write changed anything; False means the record was already up
            to date, so re-running a round is a no-op (P8-03).
    """

    location: str
    changed: bool


class RecordValidator(Protocol):
    """Checks a record against the shared JSON Schema before it is written (P8-02)."""

    @abstractmethod
    def validate(self, record: Mapping[str, Any]) -> None:
        """Validate one record.

        Args:
            record: A JSON-ready prediction record.

        Raises:
            ValidationError: If the record does not validate.
        """


class RecordReader(Protocol):
    """Looks up the record already published for a race."""

    @abstractmethod
    def load(self, event: EventKey) -> PredictionRecord | None:
        """Read one race's published record.

        Args:
            event: The race weekend.

        Returns:
            The record, or None when nothing is published for that race.

        Raises:
            DataError: If a record exists but cannot be read.
        """


class Publisher(Protocol):
    """Writes prediction records somewhere readers can find them."""

    @abstractmethod
    def publish(self, record: Mapping[str, Any]) -> PublishResult:
        """Write one validated record atomically, with the index that lists it.

        Writing content identical to what is stored changes nothing. Whether a record may
        be written at all is decided by :func:`f1sim.core.usecases.publish.publish_record`.

        Args:
            record: A record valid against ``schemas/prediction-record.v1.schema.json``.

        Returns:
            Where the record was written and whether anything changed.

        Raises:
            DataError: If the record could not be written or committed.
        """
