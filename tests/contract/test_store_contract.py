"""Contract suite every store must pass (Liskov substitution, ACID behaviour)."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from f1sim.adapters.store.ducklake_store import DuckLakeStore
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType
from f1sim.core.errors import DataError, ValidationError
from tests.support import T0, lap_rows, result_rows

pytestmark = pytest.mark.contract
LATER = T0 + timedelta(days=7)


@pytest.fixture(params=["memory", "ducklake"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    """Each store implementation in turn; every test below runs against both."""
    if request.param == "memory":
        yield InMemoryStore()
    else:
        s = DuckLakeStore(tmp_path / "lake")
        yield s
        s.close()


def _ingested(source: str, round_: int, session: str = "R") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": source,
                "season": 2025,
                "round": round_,
                "session": session,
                "run_id": "r",
                "ingested_at": T0,
            }
        ]
    )


def test_commit_creates_a_new_snapshot_and_data_reads_back(store: Any) -> None:
    before = store.latest_snapshot()
    snap = store.commit({"laps": lap_rows(), "results": result_rows()}, run_id="run-1")
    assert snap > before
    laps = store.read("laps", snapshot=snap, as_of=LATER)
    assert len(laps) == 10
    assert set(laps["driver_id"]) == {"max_verstappen", "lando_norris"}


def test_old_snapshot_does_not_see_later_commits(store: Any) -> None:
    first = store.commit({"laps": lap_rows(round_=1)}, run_id="a")
    store.commit({"laps": lap_rows(round_=2)}, run_id="b")
    assert set(store.read("laps", snapshot=first, as_of=LATER)["round"]) == {1}
    latest = store.latest_snapshot()
    assert set(store.read("laps", snapshot=latest, as_of=LATER)["round"]) == {1, 2}


def test_failed_validation_writes_nothing(store: Any) -> None:
    store.commit({"laps": lap_rows(round_=1)}, run_id="ok")
    snap = store.latest_snapshot()
    bad = lap_rows(round_=2)
    bad.loc[0, "lap_time_s"] = 5.0
    with pytest.raises(ValidationError):
        store.commit({"results": result_rows(round_=2), "laps": bad}, run_id="bad")
    assert store.latest_snapshot() == snap
    assert store.read("results", snapshot=snap, as_of=LATER).empty


def test_resync_replaces_a_session_instead_of_duplicating_it(store: Any) -> None:
    store.commit({"laps": lap_rows(n_laps=5)}, run_id="a")
    snap = store.commit({"laps": lap_rows(n_laps=3)}, run_id="b")
    assert len(store.read("laps", snapshot=snap, as_of=LATER)) == 6


def test_reads_never_return_rows_available_after_the_cutoff(store: Any) -> None:
    snap = store.commit(
        {
            "laps": pd.concat(
                [lap_rows(round_=1, available_at=T0), lap_rows(round_=2, available_at=LATER)]
            )
        },
        run_id="a",
    )
    seen = store.read("laps", snapshot=snap, as_of=T0 + timedelta(hours=1))
    assert set(seen["round"]) == {1}


def test_timezone_aware_cutoffs_are_compared_in_utc(store: Any) -> None:
    snap = store.commit({"laps": lap_rows(available_at=T0)}, run_id="a")
    aware = datetime(T0.year, T0.month, T0.day, T0.hour, tzinfo=UTC)
    assert len(store.read("laps", snapshot=snap, as_of=aware)) == 10


def test_ledger_tracks_ingested_sessions_per_source(store: Any) -> None:
    store.commit({"_ingested": _ingested("fastf1", 1)}, run_id="a")
    store.commit({"_ingested": _ingested("fastf1", 2)}, run_id="b")
    store.commit({"_ingested": _ingested("fastf1", 2)}, run_id="c")
    key = SessionKey(EventKey(2025, 2), SessionType.RACE)
    assert key in store.ingested_sessions("fastf1")
    assert len(store.ingested_sessions("fastf1")) == 2
    assert store.ingested_sessions("jolpica") == frozenset()


def test_unknown_and_internal_tables_cannot_be_read(store: Any) -> None:
    snap = store.latest_snapshot()
    with pytest.raises(DataError):
        store.read("_ingested", snapshot=snap, as_of=LATER)
    with pytest.raises(DataError):
        store.read("telemetry", snapshot=snap, as_of=LATER)
