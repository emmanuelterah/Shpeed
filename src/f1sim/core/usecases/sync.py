"""Fetch from every source, validate, and commit one snapshot (P1-11, P1-15).

The sync is incremental: sessions a source has already stored are skipped, and a run
that finds nothing new commits nothing, so it creates no snapshot. A session that a
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

ProgressCallback = Callable[[str, SessionKey], None]
"""Called with the source name and session just before each fetch."""


@dataclass(frozen=True)
class SessionFailure:
    """A session, or a whole season listing, that a source could not provide.

    Attributes:
        source: Source name.
        session: The failed session, or None if the season's sessions could not be listed.
        reason: The error message.
    """

    source: str
    session: SessionKey | None
    reason: str


@dataclass(frozen=True)
class SyncReport:
    """What one sync run did.

    Attributes:
        season: The season synced.
        snapshot_id: The snapshot created, or None if nothing new was committed.
        fetched: ``(source, session)`` pairs fetched in this run.
        skipped: Sessions skipped because they were already stored.
        failures: Sessions or listings that failed and will be retried next run.
        rows: Rows committed per data table.
    """

    season: int
    snapshot_id: SnapshotId | None
    fetched: list[tuple[str, SessionKey]] = field(default_factory=list[tuple[str, SessionKey]])
    skipped: int = 0
    failures: list[SessionFailure] = field(default_factory=list[SessionFailure])
    rows: dict[str, int] = field(default_factory=dict[str, int])

    @property
    def committed(self) -> bool:
        """Whether this run created a snapshot."""
        return self.snapshot_id is not None


class _Collector:
    """Accumulates fetched tables, ledger rows and failures across sources."""

    def __init__(self, run_id: str, now: datetime, on_progress: ProgressCallback | None):
        self.run_id = run_id
        self.now = now
        self.on_progress = on_progress
        self.batches: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self.fetched: list[tuple[str, SessionKey]] = []
        self.failures: list[SessionFailure] = []
        self.skipped = 0

    def collect(self, source: SourceAdapter, season: int, done: frozenset[SessionKey]) -> None:
        """Fetch every session of ``season`` from ``source`` that is not already stored.

        Args:
            source: The source to read.
            season: Championship year.
            done: Sessions this source has already stored; they are skipped.
        """
        try:
            sessions = source.sessions(season)
        except DataError as exc:
            self.failures.append(SessionFailure(source.name, None, str(exc)))
            return
        for session in sessions:
            if session in done:
                self.skipped += 1
            else:
                self._fetch(source, session)

    def _fetch(self, source: SourceAdapter, session: SessionKey) -> None:
        if self.on_progress is not None:
            self.on_progress(source.name, session)
        try:
            tables = source.fetch(session)
        except DataError as exc:
            self.failures.append(SessionFailure(source.name, session, str(exc)))
            return
        for name, df in tables.items():
            if not df.empty:
                self.batches[name].append(df)
        self.batches["_ingested"].append(_ledger_row(source.name, session, self.run_id, self.now))
        self.fetched.append((source.name, session))


def sync_season(  # pylint: disable=too-many-arguments
    season: int,
    *,
    sources: Sequence[SourceAdapter],
    writer: DataWriter,
    ledger: SyncLedger,
    run_id: str,
    now: datetime,
    on_progress: ProgressCallback | None = None,
    force: bool = False,
) -> SyncReport:
    """Fetch every new session of a season from every source and commit them together.

    Args:
        season: Championship year to sync.
        sources: Data sources, fetched in order.
        writer: Store that commits the fetched tables as one snapshot.
        ledger: Store record of sessions already ingested, per source.
        run_id: ID of this run, recorded with the commit and the ledger rows.
        now: Current UTC time, recorded as the ingestion time.
        on_progress: Optional callback before each fetch, for progress output.
        force: Re-fetch sessions a source has already stored and replace their rows, so a
            change in how this program converts a source's data can be republished.

    Returns:
        What was fetched, skipped and committed, and what failed.

    Raises:
        ValidationError: If fetched data fails validation; nothing is committed.
        StoreBusyError: If another process is writing to the store.
    """
    collector = _Collector(run_id, now, on_progress)
    for source in sources:
        done: frozenset[SessionKey] = (
            frozenset() if force else ledger.ingested_sessions(source.name)
        )
        collector.collect(source, season, done)

    if not collector.fetched:
        return SyncReport(season, None, collector.fetched, collector.skipped, collector.failures)

    merged = {name: _concat(dfs) for name, dfs in collector.batches.items()}
    snapshot = writer.commit(merged, run_id=run_id)
    rows = {name: len(df) for name, df in merged.items() if not name.startswith("_")}
    return SyncReport(
        season, snapshot, collector.fetched, collector.skipped, collector.failures, rows
    )


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
