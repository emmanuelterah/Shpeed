"""Data store ports (ADR-0002).

Reading, writing and the sync ledger are separate interfaces (interface segregation), so a
prediction run can be handed only a ``DataReader``.
"""

from abc import abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

import pandas as pd

from f1sim.core.domain.entities import SessionKey, SnapshotId


class DataWriter(Protocol):
    """Commits validated tables atomically as a new snapshot."""

    @abstractmethod
    def commit(self, tables: Mapping[str, pd.DataFrame], *, run_id: str) -> SnapshotId:
        """Write every table in one transaction.

        Rows matching each table's replace key are replaced, so re-syncing a session is
        idempotent.

        Args:
            tables: Canonical tables keyed by table name.
            run_id: ID of the run making the commit, recorded with it.

        Returns:
            The snapshot the commit created.

        Raises:
            ValidationError: If any table fails validation; nothing is written.
            StoreBusyError: If another process is writing to the store.
        """


class DataReader(Protocol):
    """Reads one pinned snapshot and never returns rows from after the cutoff."""

    @abstractmethod
    def latest_snapshot(self) -> SnapshotId:
        """Find the newest committed snapshot.

        Returns:
            Its ID, or 0 for a store with nothing committed.
        """

    @abstractmethod
    def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
        """Read one table as it was in a snapshot, up to a cutoff.

        Args:
            table: Canonical table name, such as ``"laps"``.
            snapshot: Snapshot to read.
            as_of: Cutoff; only rows with ``available_at`` at or before it are returned.

        Returns:
            The matching rows, sorted by the table's unique key.

        Raises:
            DataError: If ``table`` is unknown or internal.
        """


class SyncLedger(Protocol):
    """Records which sessions each source has already stored, for incremental syncs."""

    @abstractmethod
    def ingested_sessions(self, source: str) -> frozenset[SessionKey]:
        """List the sessions a source has already stored.

        Args:
            source: Source name, such as ``"fastf1"``.

        Returns:
            Every session that source has committed to the store.
        """
