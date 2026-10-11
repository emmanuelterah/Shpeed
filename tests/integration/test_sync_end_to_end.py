"""The whole Phase 1 pipeline offline: both sources -> validation -> DuckLake -> report -> CLI."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from f1sim import cli
from f1sim.adapters.sources.fastf1_adapter import FastF1Adapter
from f1sim.adapters.sources.http import HttpClient, RateLimiter
from f1sim.adapters.sources.jolpica_adapter import JolpicaAdapter
from f1sim.adapters.store.ducklake_store import DuckLakeStore
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType
from f1sim.core.usecases.sync import sync_season
from f1sim.reports.data_quality import render_markdown, session_quality
from f1sim.settings import Settings
from tests.fixtures.fastf1_fakes import fake_session, loaders
from tests.fixtures.jolpica_transport import jolpica_transport

pytestmark = pytest.mark.integration
NOW = datetime(2024, 3, 3, 12, 0)  # the day after the 2024 Bahrain GP
R1 = EventKey(2024, 1)


def _sources(now: datetime = NOW) -> list[Any]:
    limiter = RateLimiter(per_second=1e6, burst=1000, per_hour=10**6, sleep=lambda s: None)
    http = HttpClient(cache_dir=None, limiter=limiter, transport=jolpica_transport())
    jolpica = JolpicaAdapter(http, clock=lambda: now)
    ff1_sessions = {
        (2024, 1, "Q"): fake_session(datetime(2024, 3, 1, 16, 0)),
        (2024, 1, "R"): fake_session(datetime(2024, 3, 2, 15, 0)),
    }
    load_session, load_schedule = loaders(ff1_sessions)
    fastf1 = FastF1Adapter(
        load_session=load_session, load_schedule=load_schedule, clock=lambda: now
    )
    return [jolpica, fastf1]


def test_full_sync_into_ducklake_is_complete_and_leak_free(tmp_path: Path) -> None:
    store = DuckLakeStore(tmp_path / "lake")
    report = sync_season(
        2024, sources=_sources(), writer=store, ledger=store, run_id="e2e", now=NOW
    )

    assert report.committed
    assert ("jolpica", SessionKey(R1, SessionType.RACE)) in report.fetched
    assert ("fastf1", SessionKey(R1, SessionType.RACE)) in report.fetched
    # FastF1 practice sessions on the schedule have no data in the fakes: reported, not fatal
    assert {f.session.session for f in report.failures if f.session} == {
        SessionType.FP1,
        SessionType.FP2,
        SessionType.FP3,
    }

    snap = report.snapshot_id
    assert snap is not None
    laps = store.read("laps", snapshot=snap, as_of=NOW)
    results = store.read("results", snapshot=snap, as_of=NOW)
    assert set(laps["session"]) == {"Q", "R"}
    assert set(results["session"]) == {"Q", "R"}

    # Shared canonical keys: every lap's driver has a Jolpica result
    assert set(laps["driver_id"]) <= set(results["driver_id"])

    # At the prediction cutoff (end of qualifying) race data is invisible
    cutoff = datetime(2024, 3, 1, 17, 30)
    before_race = store.read("laps", snapshot=snap, as_of=cutoff)
    assert set(before_race["session"]) == {"Q"}
    assert store.read("pit_stops", snapshot=snap, as_of=cutoff).empty

    md = render_markdown(session_quality(store, 2024, snap), 2024, snap)
    assert "| 1 | R |" in md
    store.close()


def test_second_sync_is_incremental_and_creates_no_snapshot() -> None:
    store = InMemoryStore()
    first = sync_season(2024, sources=_sources(), writer=store, ledger=store, run_id="a", now=NOW)
    again = sync_season(2024, sources=_sources(), writer=store, ledger=store, run_id="b", now=NOW)
    assert first.committed
    assert not again.committed
    assert not again.fetched
    assert again.skipped == len(first.fetched)
    assert store.latest_snapshot() == first.snapshot_id


def test_new_sessions_are_picked_up_on_a_later_sync() -> None:
    store = InMemoryStore()
    friday = datetime(2024, 3, 1, 23, 0)
    first = sync_season(
        2024, sources=_sources(friday), writer=store, ledger=store, run_id="a", now=friday
    )
    later = sync_season(2024, sources=_sources(), writer=store, ledger=store, run_id="b", now=NOW)
    assert all(s.session is SessionType.QUALIFYING for _, s in first.fetched)
    assert {s.session for _, s in later.fetched} == {SessionType.RACE}


def test_cli_sync_verify_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F1SIM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("F1SIM_PREDICTIONS_DIR", str(tmp_path / "predictions"))
    monkeypatch.setattr(cli, "build_sources", lambda settings, names: _sources())
    monkeypatch.setattr(cli, "_now", lambda: NOW)
    runner = CliRunner()

    synced = runner.invoke(cli.app, ["data", "sync", "--season", "2024"])
    assert synced.exit_code == 0, synced.output
    assert "Committed snapshot" in synced.output

    again = runner.invoke(cli.app, ["data", "sync", "--season", "2024"])
    assert "Nothing new to store" in again.output

    verified = runner.invoke(cli.app, ["data", "verify"])
    assert verified.exit_code == 0, verified.output
    assert "OK:" in verified.output

    out = tmp_path / "quality.md"
    reported = runner.invoke(cli.app, ["data", "report", "--season", "2024", "--out", str(out)])
    assert reported.exit_code == 0, reported.output
    assert "Data quality: 2024 season" in out.read_text()

    gc = runner.invoke(cli.app, ["data", "gc", "--keep-last", "1"])
    assert gc.exit_code == 0, gc.output


def test_cited_snapshots_are_found_in_published_records(tmp_path: Path) -> None:
    (tmp_path / "2027").mkdir()
    (tmp_path / "2027" / "01.json").write_text('{"snapshot_id": 7}')
    (tmp_path / "2027" / "bad.json").write_text("not json")
    assert cli.cited_snapshots([tmp_path, tmp_path / "missing"]) == {7}


def test_unknown_source_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F1SIM_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(cli.app, ["data", "sync", "--season", "2024", "--source", "x"])
    assert result.exit_code != 0


def test_composition_root_builds_both_sources_without_network(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    built = cli.build_sources(settings, ("jolpica", "fastf1"))
    assert [s.name for s in built] == ["jolpica", "fastf1"]
    assert (tmp_path / "cache" / "fastf1").is_dir()
