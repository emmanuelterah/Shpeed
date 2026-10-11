"""Fetch from every source, validate, and commit one snapshot (P1-11, P1-15).

The sync is incremental: sessions a source has already stored are skipped, and a run
that finds nothing new commits nothing (so it creates no snapshot). A session that a
source cannot provide is reported and retried next time; it never aborts the others.
Everything fetched in one run is committed in a single transaction.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from f1sim.core.domain.entities import SessionKey, SnapshotId
from f1sim.core.errors import DataError
from f1sim.core.ports.sources import SourceAdapter
from f1sim.core.ports.store import DataWriter, SyncLedger


@dataclass(frozen=True)
class SessionFailure:
    source: str
    session: SessionKey | None
    """``None`` when the source could not even list the season's sessions."""
    reason: str


@dataclass(frozen=True)
class SyncReport:
    season: int
    snapshot_id: SnapshotId | None
    fetched: list[tuple[str, SessionKey]] = field(default_factory=list[tuple[str, SessionKey]])
    skipped: int = 0
    failures: list[SessionFailure] = field(default_factory=list[SessionFailure])
    rows: dict[str, int] = field(default_factory=dict[str, int])

    @property
    def committed(self) -> bool:
        return self.snapshot_id is not None


def sync_season(
    season: int,
    *,
    sources: Sequence[SourceAdapter],
    writer: DataWriter,
    ledger: SyncLedger,
    run_id: str,
    now: datetime,
    on_progress: Callable[[str, SessionKey], None] | None = None,
) -> SyncReport:
    batches: dict[str, list[pd.DataFrame]] = defaultdict(list)
    fetched: list[tuple[str, SessionKey]] = []
    failures: list[SessionFailure] = []
    skipped = 0

    for source in sources:
        done = ledger.ingested_sessions(source.name)
        try:
            sessions = source.sessions(season)
        except DataError as exc:
            failures.append(SessionFailure(source.name, None, str(exc)))
            continue
        for session in sessions:
            if session in done:
                skipped += 1
                continue
            if on_progress is not None:
                on_progress(source.name, session)
            try:
                tables = source.fetch(session)
            except DataError as exc:
                failures.append(SessionFailure(source.name, session, str(exc)))
                continue
            for name, df in tables.items():
                if not df.empty:
                    batches[name].append(df)
            batches["_ingested"].append(_ledger_row(source.name, session, run_id, now))
            fetched.append((source.name, session))

    if not fetched:
        return SyncReport(season, None, fetched, skipped, failures)

    merged = {name: _concat(dfs) for name, dfs in batches.items()}
    snapshot = writer.commit(merged, run_id=run_id)
    rows = {name: len(df) for name, df in merged.items() if not name.startswith("_")}
    return SyncReport(season, snapshot, fetched, skipped, failures, rows)


def _ledger_row(source: str, session: SessionKey, run_id: str, now: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": source,
                "season": session.event.season,
                "round": session.event.round,
                "session": session.session.value,
                "run_id": run_id,
                "ingested_at": now,
            }
        ]
    )


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack frames, ignoring all-empty columns when inferring types (keeps every column)."""
    columns = list(dict.fromkeys(c for f in frames for c in f.columns))
    trimmed = [f.dropna(axis=1, how="all") for f in frames]
    return pd.concat(trimmed, ignore_index=True).reindex(columns=columns)
