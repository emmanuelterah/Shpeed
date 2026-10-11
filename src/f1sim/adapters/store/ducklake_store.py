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

import importlib
import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from f1sim.adapters.store.lock import writer_lock
from f1sim.adapters.store.schemas import validate
from f1sim.adapters.store.tables import DATA_TABLES, TABLES, TableSpec
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType, SnapshotId
from f1sim.core.errors import DataError, ValidationError

CATALOG = "lake"

_SYNC_RUNS = (
    f"CREATE TABLE IF NOT EXISTS {CATALOG}._sync_runs "
    "(run_id VARCHAR, committed_at TIMESTAMP, row_counts VARCHAR)"
)


@dataclass(frozen=True)
class VerifyReport:
    files_checked: int
    missing: list[str] = field(default_factory=list[str])

    @property
    def ok(self) -> bool:
        return not self.missing


def load_ducklake(con: duckdb.DuckDBPyConnection) -> None:
    """Load the DuckLake extension: already installed, bundled wheel, or official repository."""
    try:
        con.execute("LOAD ducklake")
        return
    except duckdb.Error:
        pass
    try:
        bundled: Any = importlib.import_module("duckdb_extension_ducklake")
        files = sorted(Path(bundled.__file__).parent.glob("extensions/*/ducklake.duckdb_extension"))
        if files:
            con.execute(f"LOAD '{_sql_str(str(files[-1]))}'")
            return
    except ImportError:
        pass
    con.execute("INSTALL ducklake")
    con.execute("LOAD ducklake")


def _sql_str(value: str) -> str:
    return value.replace("'", "''")


class DuckLakeStore:
    """Implements ``DataWriter``, ``DataReader`` and ``SyncLedger``."""

    def __init__(self, root: Path) -> None:
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
        if wanted <= existing:
            return
        with writer_lock(self._root, timeout_s=30), self._transaction():
            for spec in TABLES.values():
                self._con.execute(spec.ddl(CATALOG))
            self._con.execute(_SYNC_RUNS)

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
        checked = {name: validate(name, df) for name, df in tables.items()}
        with writer_lock(self._root), self._transaction():
            for name, df in checked.items():
                if not df.empty:
                    self._replace(TABLES[name], df)
            for name in checked:
                self._assert_unique(TABLES[name])
            counts = json.dumps({name: len(df) for name, df in checked.items()})
            self._con.execute(
                f"INSERT INTO {CATALOG}._sync_runs VALUES (?, ?, ?)",
                [run_id, datetime.now(UTC).replace(tzinfo=None), counts],
            )
        return self.latest_snapshot()

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
        row = self._con.execute(f"SELECT max(snapshot_id) FROM {CATALOG}.snapshots()").fetchone()
        return SnapshotId(int(row[0]) if row and row[0] is not None else 0)

    def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
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
        rows = self._con.execute(
            f"SELECT season, round, session FROM {CATALOG}._ingested WHERE source = ?", [source]
        ).fetchall()
        return frozenset(
            SessionKey(EventKey(int(s), int(r)), SessionType(str(t))) for s, r, t in rows
        )

    # ------------------------------------------------------------------ maintenance
    def verify(self) -> VerifyReport:
        """Check every data file the catalog references exists and is non-empty (P1-14)."""
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
        return VerifyReport(files_checked=checked, missing=missing)

    def expire_snapshots(
        self, *, keep_last: int, protect: frozenset[SnapshotId] = frozenset()
    ) -> list[SnapshotId]:
        """Expire old snapshots and delete files only they used (P1-19).

        Keeps the newest ``keep_last`` snapshots, the initial table-creation snapshots,
        and every snapshot in ``protect`` (those cited by published predictions or
        benchmarks). Returns the expired snapshot IDs.
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
        version = int(snapshot if snapshot is not None else self.latest_snapshot())
        counts: dict[str, int] = {}
        for name in TABLES:
            row = self._con.execute(
                f"SELECT count(*) FROM {CATALOG}.{name} AT (VERSION => {version})"
            ).fetchone()
            counts[name] = int(row[0]) if row else 0
        return counts

    def close(self) -> None:
        self._con.close()
