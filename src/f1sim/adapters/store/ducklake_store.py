"""DuckLake store (P1-09 to P1-14, ADR-0002).

One local DuckDB-file catalog over Parquet data files:

* **Atomic:** a commit writes every table inside one transaction; any error rolls back.
* **Consistent:** every table is validated before the transaction starts, and row
  uniqueness is re-checked inside it before COMMIT.
* **Isolated:** reads name a snapshot (``AT (VERSION => n)``), so a sync in progress is
  invisible; a file lock allows one writer.
* **Durable:** committed snapshots and immutable Parquet files survive crashes;
  ``verify()`` checks that every file the catalog references is present.
"""

import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

import duckdb
import pandas as pd

from f1sim import __version__
from f1sim.adapters.store.lock import writer_lock
from f1sim.adapters.store.schemas import validate
from f1sim.adapters.store.tables import DATA_TABLES, TABLES, TableSpec
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType, SnapshotId
from f1sim.core.errors import DataError, ValidationError

CATALOG = "lake"

_SYNC_RUNS = (
    f"CREATE TABLE IF NOT EXISTS {CATALOG}._sync_runs "
    "(run_id VARCHAR, snapshot_id BIGINT, committed_at TIMESTAMP, "
    "sources VARCHAR, versions VARCHAR, row_counts VARCHAR)"
)
_SYNC_RUN_TYPES = {
    "run_id": "VARCHAR",
    "snapshot_id": "BIGINT",
    "committed_at": "TIMESTAMP",
    "sources": "VARCHAR",
    "versions": "VARCHAR",
    "row_counts": "VARCHAR",
}
_SYNC_RUN_COLUMNS = frozenset(_SYNC_RUN_TYPES)


BUNDLED_EXTENSION_PACKAGE = "duckdb_extension_ducklake"


@dataclass(frozen=True)
class VerifyReport:
    """The result of checking the store's data files and sync metadata.

    Attributes:
        files_checked: Data files the catalog references.
        missing: Paths that are missing or empty.
        mismatched: ``table`` to ``(recorded, actual)`` row counts that disagree with
            the last sync's metadata.
    """

    files_checked: int
    missing: list[str] = field(default_factory=list[str])
    mismatched: dict[str, tuple[int, int]] = field(default_factory=dict[str, tuple[int, int]])

    @property
    def ok(self) -> bool:
        """Whether every referenced file is present and every row count matches."""
        return not self.missing and not self.mismatched


def load_ducklake(con: duckdb.DuckDBPyConnection) -> None:
    """Load the DuckLake extension into a connection.

    Tries, in order: an already-installed extension; the signed binary bundled in the
    ``duckdb-extension-ducklake`` wheel (a dev dependency, so tests run offline); and
    finally ``INSTALL ducklake`` from DuckDB's official repository.

    Args:
        con: The DuckDB connection to load it into.
    """
    try:
        con.execute("LOAD ducklake")
        return
    except duckdb.Error:
        pass
    # find_spec locates the bundled wheel's files without importing it
    spec = find_spec(BUNDLED_EXTENSION_PACKAGE)
    if spec is not None and spec.origin is not None:
        pattern = "extensions/*/ducklake.duckdb_extension"
        files = sorted(Path(spec.origin).parent.glob(pattern))
        if files:
            con.execute(f"LOAD '{_sql_str(str(files[-1]))}'")
            return
    con.execute("INSTALL ducklake")
    con.execute("LOAD ducklake")


def _sql_str(value: str) -> str:
    return value.replace("'", "''")


def _sources(checked: Mapping[str, pd.DataFrame]) -> list[str]:
    """The distinct source names a commit's ledger rows came from."""
    ledger = checked.get("_ingested")
    if ledger is None or ledger.empty:
        return []
    return sorted({str(source) for source in ledger["source"]})


class DuckLakeStore:
    """The transactional data store: a ``DataWriter``, ``DataReader`` and ``SyncLedger``."""

    def __init__(self, root: Path) -> None:
        """Open, or create, the store under ``root``.

        Args:
            root: Directory for ``catalog.ducklake`` and the Parquet ``files/``.
        """
        self._root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "files").mkdir(exist_ok=True)
        self._con = duckdb.connect()
        load_ducklake(self._con)
        catalog = _sql_str(str(root / "catalog.ducklake"))
        data_path = _sql_str(str(root / "files"))
        self._con.execute(
            f"ATTACH 'ducklake:{catalog}' AS {CATALOG} "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        self._ensure_tables()

    # ------------------------------------------------------------------ setup
    def _ensure_tables(self) -> None:
        existing = {r[0] for r in self._con.execute("SHOW TABLES FROM lake").fetchall()}
        wanted = {*TABLES, "_sync_runs"}
        if wanted <= existing and self._sync_runs_complete():
            return
        with writer_lock(self._root, timeout_s=30), self._transaction():
            for spec in TABLES.values():
                self._con.execute(spec.ddl(CATALOG))
            self._con.execute(_SYNC_RUNS)
            self._migrate_sync_runs()

    def _migrate_sync_runs(self) -> None:
        """Add sync-metadata columns a catalog created by an older version is missing."""
        present = {row[0] for row in self._con.execute(f"DESCRIBE {CATALOG}._sync_runs").fetchall()}
        for column in sorted(_SYNC_RUN_COLUMNS - present):
            self._con.execute(
                f"ALTER TABLE {CATALOG}._sync_runs ADD COLUMN {column} {_SYNC_RUN_TYPES[column]}"
            )

    def _sync_runs_complete(self) -> bool:
        present = {row[0] for row in self._con.execute(f"DESCRIBE {CATALOG}._sync_runs").fetchall()}
        return _SYNC_RUN_COLUMNS.issubset(present)

    @contextmanager
    def _transaction(self) -> Generator[None]:
        self._con.execute("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            self._con.execute("ROLLBACK")
            raise
        self._con.execute("COMMIT")

    # ------------------------------------------------------------------ DataWriter
    def commit(self, tables: Mapping[str, pd.DataFrame], *, run_id: str) -> SnapshotId:
        """Validate every table, then write them all in one transaction.

        Args:
            tables: Canonical tables keyed by table name.
            run_id: ID of the run making the commit, recorded in ``_sync_runs``.

        Returns:
            The snapshot the commit created.

        Raises:
            ValidationError: If a table fails validation or would hold duplicate rows;
                nothing is written.
            StoreBusyError: If another process holds the writer lock.
        """
        checked = {name: validate(name, df) for name, df in tables.items()}
        with writer_lock(self._root), self._transaction():
            for name, df in checked.items():
                if not df.empty:
                    self._replace(TABLES[name], df)
            for name in checked:
                self._assert_unique(TABLES[name])
            self._record_run(run_id, checked)
        return self.latest_snapshot()

    def _record_run(self, run_id: str, checked: Mapping[str, pd.DataFrame]) -> None:
        """Record the sync's metadata, including the snapshot this transaction creates.

        It runs inside the commit's transaction, so the snapshot ID is the one the
        transaction is about to create (``ducklake_current_snapshot`` plus one: the
        current snapshot is the last committed one, and a single writer holds the
        lock, so nothing else can take the next ID first).
        """
        self._con.execute(
            f"INSERT INTO {CATALOG}._sync_runs VALUES (?, ?, ?, ?, ?, ?)",
            [
                run_id,
                self._next_snapshot(),
                datetime.now(UTC).replace(tzinfo=None),
                json.dumps(_sources(checked)),
                json.dumps({"f1sim": __version__}),
                json.dumps(self._current_counts()),
            ],
        )

    def _next_snapshot(self) -> int:
        row = self._con.execute(f"SELECT * FROM ducklake_current_snapshot('{CATALOG}')").fetchone()
        latest = row[0] if row else None
        return (int(latest) if latest is not None else 0) + 1

    def _current_counts(self) -> dict[str, int]:
        """Row counts per table as they are right now, including uncommitted writes."""
        counts: dict[str, int] = {}
        for name in TABLES:
            row = self._con.execute(f"SELECT count(*) FROM {CATALOG}.{name}").fetchone()
            counts[name] = int(row[0]) if row else 0
        return counts

    def _replace(self, spec: TableSpec, df: pd.DataFrame) -> None:
        keys = list(spec.replace_key)
        slices = df[keys].drop_duplicates()
        match = " AND ".join(f"t.{k} = s.{k}" for k in keys)
        self._con.register("_slices", slices)
        self._con.register("_batch", df)
        try:
            self._con.execute(
                f"DELETE FROM {CATALOG}.{spec.name} t "
                f"WHERE EXISTS (SELECT 1 FROM _slices s WHERE {match})"
            )
            self._con.execute(f"INSERT INTO {CATALOG}.{spec.name} BY NAME SELECT * FROM _batch")
        finally:
            self._con.unregister("_slices")
            self._con.unregister("_batch")

    def _assert_unique(self, spec: TableSpec) -> None:
        cols = ", ".join(spec.unique)
        row = self._con.execute(
            f"SELECT count(*) FROM (SELECT {cols} FROM {CATALOG}.{spec.name} "
            f"GROUP BY {cols} HAVING count(*) > 1)"
        ).fetchone()
        if row and row[0]:
            raise ValidationError(f"{spec.name}: {row[0]} duplicate rows after commit; rolled back")

    # ------------------------------------------------------------------ DataReader
    def latest_snapshot(self) -> SnapshotId:
        """Find the newest committed snapshot.

        Returns:
            Its ID.
        """
        row = self._con.execute(f"SELECT max(snapshot_id) FROM {CATALOG}.snapshots()").fetchone()
        return SnapshotId(int(row[0]) if row and row[0] is not None else 0)

    def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
        """Read one table as it was in a snapshot, up to a cutoff.

        Args:
            table: Canonical table name.
            snapshot: Snapshot to read; one that does not exist reads as empty.
            as_of: Cutoff; only rows with ``available_at`` at or before it are returned.
                Timezone-aware values are compared in UTC.

        Returns:
            The matching rows, sorted by the table's unique key.

        Raises:
            DataError: If ``table`` is unknown or internal.
        """
        if table not in DATA_TABLES:
            raise DataError(f"Unknown or internal table {table!r}; readable: {list(DATA_TABLES)}")
        spec = TABLES[table]
        order = ", ".join(spec.unique)
        cutoff = as_of.astimezone(UTC).replace(tzinfo=None) if as_of.tzinfo else as_of
        try:
            df = self._con.execute(
                f"SELECT * FROM {CATALOG}.{table} AT (VERSION => {int(snapshot)}) "
                f"WHERE available_at <= ? ORDER BY {order}",
                [cutoff],
            ).df()
        except duckdb.Error as exc:
            if "does not exist" in str(exc) or "snapshot" in str(exc).lower():
                return pd.DataFrame(columns=list(spec.columns))
            raise
        return df

    # ------------------------------------------------------------------ SyncLedger
    def ingested_sessions(self, source: str) -> frozenset[SessionKey]:
        """List the sessions a source has already stored.

        Args:
            source: Source name.

        Returns:
            Every session that source has committed.
        """
        rows = self._con.execute(
            f"SELECT season, round, session FROM {CATALOG}._ingested WHERE source = ?", [source]
        ).fetchall()
        return frozenset(
            SessionKey(EventKey(int(s), int(r)), SessionType(str(t))) for s, r, t in rows
        )

    # ------------------------------------------------------------------ maintenance
    def verify(self) -> VerifyReport:
        """Check the catalog's data files and that row counts match the sync metadata (P1-14).

        Returns:
            How many files were checked, which are missing, and any table whose row
            count disagrees with the last sync's record.
        """
        missing: list[str] = []
        checked = 0
        for name in TABLES:
            for (path,) in self._con.execute(
                f"SELECT data_file FROM ducklake_list_files('{CATALOG}', '{name}')"
            ).fetchall():
                checked += 1
                p = Path(str(path))
                if not p.is_file() or p.stat().st_size == 0:
                    missing.append(str(path))
        # Counting rows needs the files, so only compare them once every file is present.
        mismatched = {} if missing else self._count_mismatches()
        return VerifyReport(files_checked=checked, missing=missing, mismatched=mismatched)

    def _count_mismatches(self) -> dict[str, tuple[int, int]]:
        """Compare each table's rows to the last sync's recorded totals."""
        row = self._con.execute(
            f"SELECT snapshot_id, row_counts FROM {CATALOG}._sync_runs "
            "WHERE snapshot_id IS NOT NULL ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return {}
        expected: dict[str, int] = json.loads(row[1])
        try:
            actual = self.row_counts(snapshot=SnapshotId(int(row[0])))
        except duckdb.Error:  # the recorded snapshot has since been expired
            return {}
        return {
            name: (int(want), int(actual.get(name, 0)))
            for name, want in expected.items()
            if int(actual.get(name, 0)) != int(want)
        }

    def expire_snapshots(
        self, *, keep_last: int, protect: frozenset[SnapshotId] = frozenset()
    ) -> list[SnapshotId]:
        """Expire old snapshots and delete the files only they used (P1-19).

        The initial table-creation snapshots are never expired.

        Args:
            keep_last: How many of the newest data snapshots to keep.
            protect: Snapshots to keep regardless, such as those cited by published
                predictions or benchmarks.

        Returns:
            The expired snapshot IDs, oldest first.

        Raises:
            StoreBusyError: If another process holds the writer lock.
        """
        ids = [
            int(r[0])
            for r in self._con.execute(
                f"SELECT snapshot_id FROM {CATALOG}.snapshots() ORDER BY snapshot_id"
            ).fetchall()
        ]
        data_snapshots = [i for i in ids if i > self._schema_snapshot()]
        keep = set(data_snapshots[-keep_last:]) if keep_last > 0 else set[int]()
        doomed = [i for i in data_snapshots if i not in keep and SnapshotId(i) not in protect]
        if not doomed:
            return []
        with writer_lock(self._root):
            listed = ", ".join(str(i) for i in doomed)
            self._con.execute(
                f"CALL ducklake_expire_snapshots('{CATALOG}', versions => [{listed}])"
            )
            self._con.execute(f"CALL ducklake_cleanup_old_files('{CATALOG}', cleanup_all => true)")
        return [SnapshotId(i) for i in doomed]

    def _schema_snapshot(self) -> int:
        """The last snapshot that only created tables; never expired."""
        row = self._con.execute(
            f"SELECT min(snapshot_id) FROM {CATALOG}.snapshots() "
            "WHERE NOT list_contains(map_keys(changes), 'tables_created') "
            "AND NOT list_contains(map_keys(changes), 'schemas_created')"
        ).fetchone()
        first_data = int(row[0]) if row and row[0] is not None else 10**9
        return first_data - 1

    def row_counts(self, *, snapshot: SnapshotId | None = None) -> dict[str, int]:
        """Count the rows in every table.

        Args:
            snapshot: Snapshot to count; None means the latest.

        Returns:
            Row count per table name.
        """
        version = int(snapshot if snapshot is not None else self.latest_snapshot())
        counts: dict[str, int] = {}
        for name in TABLES:
            row = self._con.execute(
                f"SELECT count(*) FROM {CATALOG}.{name} AT (VERSION => {version})"
            ).fetchone()
            counts[name] = int(row[0]) if row else 0
        return counts

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._con.close()
