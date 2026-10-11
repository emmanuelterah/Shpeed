"""Data store ports (ADR-0002). Reading, writing and the sync ledger are separate
interfaces (interface segregation): a prediction run is handed only a ``DataReader``.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

import pandas as pd

from f1sim.core.domain.entities import SessionKey, SnapshotId


class DataWriter(Protocol):
    """Commits validated tables atomically as a new snapshot."""

    def commit(self, tables: Mapping[str, pd.DataFrame], *, run_id: str) -> SnapshotId:
        """Write all tables in one transaction.

        Rows matching each table's replace key are replaced, so re-syncing a session is
        idempotent. Raises ``ValidationError`` and writes nothing if any check fails.
        """
        ...


class DataReader(Protocol):
    """Reads one pinned snapshot, never past the cutoff."""

    def latest_snapshot(self) -> SnapshotId: ...

    def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
        """Rows of ``table`` in ``snapshot`` whose ``available_at`` is at or before ``as_of``."""
        ...


class SyncLedger(Protocol):
    """Which sessions each source has already stored, for incremental syncs."""

    def ingested_sessions(self, source: str) -> frozenset[SessionKey]: ...
