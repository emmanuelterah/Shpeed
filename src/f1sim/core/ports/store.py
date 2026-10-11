"""Data store ports (ADR-0002). Reading and writing are separate (interface segregation)."""

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

import pandas as pd

from f1sim.core.domain.entities import SnapshotId


class DataWriter(Protocol):
    """Commits validated tables atomically as a new snapshot."""

    def commit(self, tables: Mapping[str, pd.DataFrame], *, run_id: str) -> SnapshotId:
        """Write all tables in one transaction.

        Raises ``ValidationError`` and writes nothing if any table fails validation.
        """
        ...


class DataReader(Protocol):
    """Reads one pinned snapshot, never past the cutoff."""

    def latest_snapshot(self) -> SnapshotId: ...

    def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
        """Rows of ``table`` in ``snapshot`` timestamped at or before ``as_of``.

        Implementations raise ``LeakageError`` if asked for data past ``as_of``.
        """
        ...
