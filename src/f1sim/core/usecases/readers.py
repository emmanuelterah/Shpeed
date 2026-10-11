"""Readers that enforce the prediction cutoff and serve a frozen snapshot (P2-05).

``CutoffReader`` is the leakage guard: models in a backtest only ever see data through it,
and any attempt to read past the cutoff raises ``LeakageError`` instead of quietly
returning future rows. ``FrameReader`` holds one snapshot's tables in memory, so a
backtest reads the store once and every race reads from memory.
"""

from collections.abc import Mapping
from datetime import UTC, datetime

import pandas as pd

from f1sim.core.domain.entities import SnapshotId
from f1sim.core.errors import DataError, LeakageError
from f1sim.core.ports.store import DataReader

FAR_FUTURE = datetime(2200, 1, 1)
"""A cutoff later than any data; pandas timestamps cannot go past the year 2262."""


def as_naive_utc(moment: datetime) -> datetime:
    """Convert a datetime to naive UTC, the form stored in every ``available_at`` column.

    Args:
        moment: A naive (assumed UTC) or timezone-aware datetime.

    Returns:
        The same instant without timezone information.
    """
    return moment.astimezone(UTC).replace(tzinfo=None) if moment.tzinfo else moment


class CutoffReader:
    """A ``DataReader`` that refuses to look past a cutoff."""

    def __init__(self, inner: DataReader, cutoff: datetime) -> None:
        """Wrap a reader.

        Args:
            inner: The reader to guard.
            cutoff: Latest moment whose data may be read.
        """
        self._inner = inner
        self._cutoff = as_naive_utc(cutoff)

    @property
    def cutoff(self) -> datetime:
        """The cutoff this reader enforces, as naive UTC."""
        return self._cutoff

    def latest_snapshot(self) -> SnapshotId:
        """Find the newest snapshot of the wrapped reader.

        Returns:
            Its ID.
        """
        return self._inner.latest_snapshot()

    def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
        """Read a table, refusing anything past the cutoff.

        Args:
            table: Canonical table name.
            snapshot: Snapshot to read.
            as_of: Requested cutoff; must not be later than this reader's cutoff.

        Returns:
            The rows available at ``as_of``.

        Raises:
            LeakageError: If ``as_of`` is past the cutoff, or the wrapped reader returned
                a row from after it.
        """
        if as_naive_utc(as_of) > self._cutoff:
            raise LeakageError(
                f"Read of {table!r} as of {as_of.isoformat()} is past the prediction "
                f"cutoff {self._cutoff.isoformat()}"
            )
        rows = self._inner.read(table, snapshot=snapshot, as_of=as_of)
        if len(rows) and bool((pd.to_datetime(rows["available_at"]) > self._cutoff).any()):
            raise LeakageError(f"{table!r} returned rows from after {self._cutoff.isoformat()}")
        return rows


class FrameReader:
    """A ``DataReader`` over one snapshot's tables, already loaded into memory."""

    def __init__(self, snapshot: SnapshotId, tables: Mapping[str, pd.DataFrame]) -> None:
        """Hold the tables of one snapshot.

        Args:
            snapshot: The snapshot the tables were read from.
            tables: Table name to every row of that table in the snapshot.
        """
        self._snapshot = snapshot
        self._tables = {
            name: df.assign(available_at=pd.to_datetime(df["available_at"]))
            for name, df in tables.items()
        }

    @classmethod
    def load(
        cls, reader: DataReader, snapshot: SnapshotId, tables: tuple[str, ...]
    ) -> "FrameReader":
        """Read whole tables from a store once.

        Args:
            reader: The store to read.
            snapshot: Snapshot to read.
            tables: Table names to load.

        Returns:
            A reader serving those tables from memory.
        """
        return cls(
            snapshot, {t: reader.read(t, snapshot=snapshot, as_of=FAR_FUTURE) for t in tables}
        )

    def latest_snapshot(self) -> SnapshotId:
        """Report the snapshot these tables came from.

        Returns:
            Its ID.
        """
        return self._snapshot

    def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
        """Read a loaded table up to a cutoff.

        Args:
            table: A table name passed to the constructor.
            snapshot: Must be the snapshot the tables came from.
            as_of: Only rows with ``available_at`` at or before it are returned.

        Returns:
            A copy of the matching rows.

        Raises:
            DataError: If the table was not loaded or the snapshot differs.
        """
        if snapshot != self._snapshot:
            raise DataError(f"Frames hold snapshot {self._snapshot}, not {snapshot}")
        if table not in self._tables:
            raise DataError(f"Table {table!r} was not loaded; loaded: {sorted(self._tables)}")
        df = self._tables[table]
        selected: pd.DataFrame = df.loc[df["available_at"] <= as_naive_utc(as_of)]
        return selected.copy()
