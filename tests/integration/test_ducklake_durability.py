"""DuckLake-specific guarantees: crash safety, durability, single writer, verification."""

import subprocess
import sys
import textwrap
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from f1sim.adapters.store.ducklake_store import DuckLakeStore
from f1sim.adapters.store.lock import writer_lock
from f1sim.core.errors import StoreBusyError
from tests.support import T0, lap_rows

LATER = T0 + timedelta(days=7)
REPO = Path(__file__).resolve().parents[2]


def test_process_killed_mid_commit_leaves_previous_snapshot_intact(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    store = DuckLakeStore(root)
    good = store.commit({"laps": lap_rows(round_=1)}, run_id="good")
    store.close()

    # A second process starts a commit and dies after writing rows but before COMMIT.
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(REPO)!r})
        from pathlib import Path
        from f1sim.adapters.store.ducklake_store import DuckLakeStore
        from tests.support import lap_rows
        store = DuckLakeStore(Path({str(root)!r}))
        store._assert_unique = lambda spec: os._exit(9)  # crash inside the transaction
        store.commit({{"laps": lap_rows(round_=2)}}, run_id="doomed")
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True)
    assert proc.returncode == 9, proc.stderr.decode()

    reopened = DuckLakeStore(root)
    assert reopened.latest_snapshot() == good
    laps = reopened.read("laps", snapshot=reopened.latest_snapshot(), as_of=LATER)
    assert set(laps["round"]) == {1}
    reopened.close()


def test_committed_data_survives_reopening(tmp_path: Path) -> None:
    store = DuckLakeStore(tmp_path / "lake")
    snap = store.commit({"laps": lap_rows()}, run_id="a")
    store.close()
    again = DuckLakeStore(tmp_path / "lake")
    assert len(again.read("laps", snapshot=snap, as_of=LATER)) == 10
    assert again.row_counts()["laps"] == 10
    again.close()


def test_second_writer_is_refused_while_lock_is_held(tmp_path: Path) -> None:
    store = DuckLakeStore(tmp_path / "lake")
    with writer_lock(tmp_path / "lake"), pytest.raises(StoreBusyError):
        store.commit({"laps": lap_rows()}, run_id="blocked")
    store.close()


def test_data_is_written_as_parquet_and_verify_finds_missing_files(tmp_path: Path) -> None:
    store = DuckLakeStore(tmp_path / "lake")
    store.commit({"laps": lap_rows()}, run_id="a")
    report = store.verify()
    assert report.ok
    assert report.files_checked >= 1
    parquet = next((tmp_path / "lake" / "files").rglob("*.parquet"))
    parquet.unlink()
    broken = store.verify()
    assert not broken.ok
    assert any(parquet.name in m for m in broken.missing)
    store.close()


def test_retention_keeps_recent_and_cited_snapshots(tmp_path: Path) -> None:
    store = DuckLakeStore(tmp_path / "lake")
    snaps = [store.commit({"laps": lap_rows(n_laps=n)}, run_id=str(n)) for n in (5, 4, 3, 2)]
    cited = frozenset({snaps[0]})
    expired = store.expire_snapshots(keep_last=1, protect=cited)
    assert expired == [snaps[1], snaps[2]]
    # The cited snapshot still reads exactly as it was published
    assert len(store.read("laps", snapshot=snaps[0], as_of=LATER)) == 10
    assert len(store.read("laps", snapshot=snaps[3], as_of=LATER)) == 4
    assert store.verify().ok
    assert store.expire_snapshots(keep_last=1, protect=cited) == []
    store.close()
