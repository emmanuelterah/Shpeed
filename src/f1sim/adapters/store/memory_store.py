"""In-memory store for tests (P1-13). Same contract as ``DuckLakeStore``, no files."""

from collections.abc import Mapping
from datetime import UTC, datetime

import pandas as pd

from f1sim.adapters.store.schemas import validate
from f1sim.adapters.store.tables import DATA_TABLES, TABLES
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType, SnapshotId
from f1sim.core.errors import DataError, ValidationError


class InMemoryStore:
    """``DataWriter``, ``DataReader`` and ``SyncLedger`` with copy-on-commit snapshots."""

    def __init__(self) -> None:
        empty = {name: pd.DataFrame(columns=list(spec.columns)) for name, spec in TABLES.items()}
        self._snapshots: list[dict[str, pd.DataFrame]] = [empty]

    def commit(self, tables: Mapping[str, pd.DataFrame], *, run_id: str) -> SnapshotId:
        checked = {name: validate(name, df) for name, df in tables.items()}
        staged = dict(self._snapshots[-1])
        for name, df in checked.items():
            if df.empty:
                continue
            spec = TABLES[name]
            current = staged[name]
            keys = list(spec.replace_key)
            if not current.empty:
                hit = current[keys].merge(df[keys].drop_duplicates(), how="left", indicator=True)
                current = current[(hit["_merge"] == "left_only").to_numpy()]
            merged = df if current.empty else pd.concat([current, df], ignore_index=True)
            if merged.duplicated(subset=list(spec.unique)).any():
                raise ValidationError(f"{name}: duplicate rows after commit; rolled back")
            staged[name] = merged.reset_index(drop=True)
        self._snapshots.append(staged)
        return self.latest_snapshot()

    def latest_snapshot(self) -> SnapshotId:
        return SnapshotId(len(self._snapshots) - 1)

    def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
        if table not in DATA_TABLES:
            raise DataError(f"Unknown or internal table {table!r}; readable: {list(DATA_TABLES)}")
        if not 0 <= int(snapshot) < len(self._snapshots):
            return pd.DataFrame(columns=list(TABLES[table].columns))
        df = self._snapshots[int(snapshot)][table]
        if df.empty:
            return df.copy()
        cutoff = as_of.astimezone(UTC).replace(tzinfo=None) if as_of.tzinfo else as_of
        out = df[df["available_at"] <= pd.Timestamp(cutoff)]
        return out.sort_values(list(TABLES[table].unique)).reset_index(drop=True)

    def ingested_sessions(self, source: str) -> frozenset[SessionKey]:
        df = self._snapshots[-1]["_ingested"]
        if df.empty:
            return frozenset()
        rows = df[df["source"] == source]
        return frozenset(
            SessionKey(EventKey(int(s), int(r)), SessionType(str(t)))
            for s, r, t in rows[["season", "round", "session"]].itertuples(index=False)
        )
