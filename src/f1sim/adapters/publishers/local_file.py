"""Writes records to a local predictions folder (P8-03).

Each record is written through a temp file and renamed into place, and the index is rebuilt
alongside it. Writing a record identical to the stored one changes nothing, so re-running a
round is a no-op. Whether a record may be written at all (the lights-out lock) is decided in
the core by :func:`f1sim.core.usecases.publish.publish_record`, not here.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from f1sim.adapters.publishers.index import (
    read_json,
    rebuild_index,
    record_path,
    to_json,
    write_if_changed,
)
from f1sim.core.domain.entities import EventKey
from f1sim.core.domain.record import PredictionRecord
from f1sim.core.errors import DataError, InvariantError
from f1sim.core.ports.publisher import PublishResult


class LocalFilePublisher:
    """Publishes records as JSON files under a root folder, and reads them back."""

    def __init__(self, root: Path) -> None:
        """Create a publisher.

        Args:
            root: The predictions folder; created on first write.
        """
        self._root = root

    @property
    def root(self) -> Path:
        """The folder records are written under.

        Returns:
            The root path.
        """
        return self._root

    def load(self, event: EventKey) -> PredictionRecord | None:
        """Read one race's published record.

        Args:
            event: The race weekend.

        Returns:
            The record, or None when nothing is published for that race.

        Raises:
            DataError: If the file exists but is not a valid record.
        """
        path = record_path(self._root, event.season, event.round)
        if not path.is_file():
            return None
        try:
            return PredictionRecord.from_mapping(read_json(path))
        except InvariantError as exc:
            raise DataError(f"{path} is not a valid prediction record: {exc}") from exc

    def publish(self, record: Mapping[str, Any]) -> PublishResult:
        """Write one record atomically and refresh the index.

        Args:
            record: A record valid against the shared schema.

        Returns:
            Where the record lives and whether the record or the index changed.
        """
        path = record_path(self._root, int(record["season"]), int(record["round"]))
        wrote = write_if_changed(path, to_json(dict(record)))
        _, index_changed = rebuild_index(self._root)
        return PublishResult(str(path), changed=wrote or index_changed)
