"""Command-line interface and composition root.

This is the only module that knows concrete adapter classes: it reads settings, builds the
adapters, and injects them into the core use cases (dependency inversion). Each command
passes its ``--help`` text to Typer explicitly, so the docstrings can follow the Google style
without that markup appearing in the terminal.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from f1sim.adapters.sources.fastf1_adapter import FastF1Adapter, fastf1_loaders
from f1sim.adapters.sources.http import HttpClient
from f1sim.adapters.sources.jolpica_adapter import JolpicaAdapter
from f1sim.adapters.store.ducklake_store import DuckLakeStore
from f1sim.core.domain.entities import SessionKey, SnapshotId
from f1sim.core.ports.sources import SourceAdapter
from f1sim.core.usecases.sync import SyncReport, sync_season
from f1sim.observability import configure_logging
from f1sim.reports.data_quality import render_markdown, session_quality
from f1sim.settings import Settings

app = typer.Typer(help="Shpeed F1 race simulator", no_args_is_help=True)
data_app = typer.Typer(help="Fetch, validate and manage the local data store", no_args_is_help=True)
report_app = typer.Typer(help="Backtest and scoring reports", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(report_app, name="report")

SOURCES = ("jolpica", "fastf1")
CITATION_FOLDERS = (Path("benchmarks"),)

Season = Annotated[int, typer.Option(help="Season, for example 2025")]
Round = Annotated[int, typer.Option("--round", help="Round number within the season")]


@app.callback()
def main(ctx: typer.Context) -> None:
    """Load settings and configure logging before any command runs.

    Args:
        ctx: Typer context; settings and the run ID are stored on ``ctx.obj``.
    """
    settings = Settings()
    ctx.obj = {"settings": settings, "run_id": configure_logging(settings.log_level)}


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def open_store(settings: Settings) -> DuckLakeStore:
    """Open the local data store.

    Args:
        settings: Runtime settings; the store lives in ``<data_dir>/lake``.

    Returns:
        The open store; the caller must close it.
    """
    return DuckLakeStore(settings.data_dir / "lake")


def build_sources(settings: Settings, names: Sequence[str]) -> list[SourceAdapter]:
    """Build the requested source adapters with their real HTTP clients and caches.

    Args:
        settings: Runtime settings; caches live under ``cache_dir``.
        names: Source names to build, from ``SOURCES``.

    Returns:
        The adapters, in ``SOURCES`` order.
    """
    built: list[SourceAdapter] = []
    if "jolpica" in names:
        http = HttpClient(cache_dir=settings.cache_dir / "http")
        built.append(JolpicaAdapter(http, clock=_now))
    if "fastf1" in names:
        load_session, load_schedule = fastf1_loaders(settings.cache_dir / "fastf1")
        built.append(
            FastF1Adapter(load_session=load_session, load_schedule=load_schedule, clock=_now)
        )
    return built


def _print_progress(source: str, session: SessionKey) -> None:
    typer.echo(f"  {source:<8} round {session.event.round:>2}  {session.session.value}")


def _print_sync_report(report: SyncReport) -> None:
    if report.committed:
        rows = ", ".join(f"{k} {v:,}" for k, v in sorted(report.rows.items()))
        typer.secho(f"Committed snapshot {report.snapshot_id}: {rows}", fg=typer.colors.GREEN)
    else:
        typer.echo("Nothing new to store; no snapshot created.")
    typer.echo(f"{len(report.fetched)} sessions fetched, {report.skipped} already stored.")
    for failure in report.failures:
        session = failure.session
        where = f"round {session.event.round} {session.session.value}" if session else "schedule"
        typer.secho(f"  skipped {failure.source} {where}: {failure.reason}", fg=typer.colors.YELLOW)


@data_app.command("sync", help="Fetch, validate and commit one season as a new snapshot.")
def data_sync(
    ctx: typer.Context,
    season: Season,
    source: Annotated[str, typer.Option(help="all, jolpica or fastf1")] = "all",
) -> None:
    """Fetch, validate and commit one season as a new snapshot (P1-17).

    Args:
        ctx: Typer context holding settings and the run ID.
        season: Season to sync.
        source: ``"all"`` or one source name.

    Raises:
        BadParameter: If ``source`` is not a known source.
        Exit: With code 1 if every session failed and nothing was fetched.
    """
    settings: Settings = ctx.obj["settings"]
    names = SOURCES if source == "all" else (source,)
    if any(n not in SOURCES for n in names):
        raise typer.BadParameter(f"source must be all, {' or '.join(SOURCES)}")
    sources = build_sources(settings, names)
    store = open_store(settings)

    typer.echo(f"Syncing {season} from {', '.join(names)} into {settings.data_dir / 'lake'}")
    try:
        report = sync_season(
            season,
            sources=sources,
            writer=store,
            ledger=store,
            run_id=ctx.obj["run_id"],
            now=_now(),
            on_progress=_print_progress,
        )
    finally:
        store.close()

    _print_sync_report(report)
    if report.failures and not report.fetched:
        raise typer.Exit(1)


@data_app.command("verify", help="Check every data file the catalog references.")
def data_verify(ctx: typer.Context) -> None:
    """Check every data file the catalog references (P1-14).

    Args:
        ctx: Typer context holding settings.

    Raises:
        Exit: With code 1 if any file is missing.
    """
    store = open_store(ctx.obj["settings"])
    try:
        result = store.verify()
        counts = store.row_counts()
        snapshot = store.latest_snapshot()
    finally:
        store.close()
    typer.echo(f"Snapshot {snapshot}: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    if result.ok:
        typer.secho(f"OK: {result.files_checked} data files present.", fg=typer.colors.GREEN)
        return
    typer.secho(
        f"{len(result.missing)} of {result.files_checked} files missing:", fg=typer.colors.RED
    )
    for path in result.missing:
        typer.echo(f"  {path}")
    raise typer.Exit(1)


@data_app.command("report", help="Print or save a per-session data-quality report.")
def data_report(
    ctx: typer.Context,
    season: Season,
    out: Annotated[
        Path | None, typer.Option(help="Write Markdown here instead of printing it")
    ] = None,
) -> None:
    """Write the per-session data-quality report (P1-16).

    Args:
        ctx: Typer context holding settings.
        season: Season to report on.
        out: File to write; None prints to the terminal.
    """
    store = open_store(ctx.obj["settings"])
    try:
        snapshot = store.latest_snapshot()
        markdown = render_markdown(session_quality(store, season, snapshot), season, snapshot)
    finally:
        store.close()
    if out is None:
        typer.echo(markdown)
    else:
        out.write_text(markdown, encoding="utf-8")
        typer.echo(f"Wrote {out}")


@data_app.command("gc", help="Expire old snapshots, keeping any a prediction or benchmark cites.")
def data_gc(
    ctx: typer.Context,
    keep_last: Annotated[int, typer.Option(help="Newest snapshots to keep")] = 5,
) -> None:
    """Expire old snapshots, never one a published prediction or benchmark cites (P1-19).

    Args:
        ctx: Typer context holding settings.
        keep_last: How many of the newest snapshots to keep.
    """
    settings: Settings = ctx.obj["settings"]
    folders = [settings.predictions_dir, *CITATION_FOLDERS]
    cited = frozenset(SnapshotId(s) for s in cited_snapshots(folders))
    store = open_store(settings)
    try:
        expired = store.expire_snapshots(keep_last=keep_last, protect=cited)
    finally:
        store.close()
    typer.echo(f"Expired {len(expired)} snapshots; protected {len(cited)} cited ones.")


def cited_snapshots(folders: Sequence[Path]) -> set[int]:
    """Collect the snapshot IDs that published files cite.

    Args:
        folders: Folders to search recursively for JSON files; missing ones are skipped.

    Returns:
        Every integer ``snapshot_id`` found; unreadable or non-JSON files are ignored.
    """
    found: set[int] = set()
    for folder in folders:
        files: list[Path] = sorted(folder.rglob("*.json")) if folder.is_dir() else []
        for file in files:
            try:
                data: Any = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            record = cast(dict[str, Any], data) if isinstance(data, dict) else {}
            snapshot = record.get("snapshot_id")
            if isinstance(snapshot, int):
                found.add(snapshot)
    return found


@app.command(help="Predict one race and write its prediction record.")
def predict(season: Season, round_: Round) -> None:
    """Predict one race and write its prediction record (P8-01).

    Args:
        season: Season of the race.
        round_: Round number of the race.

    Raises:
        NotImplementedError: Always, until P8-01 is built.
    """
    raise NotImplementedError(f"P8-01: predict {season} round {round_}")


@app.command(help="Walk-forward backtest of a registered model.")
def backtest(
    model: Annotated[str, typer.Option(help="Registered model name")] = "plackett_luce",
    seasons: Annotated[str, typer.Option(help="Season range, such as 2022-2025")] = "2022-2025",
) -> None:
    """Run a walk-forward backtest of a registered model (P2-04).

    Args:
        model: Registered model name.
        seasons: Inclusive season range.

    Raises:
        NotImplementedError: Always, until P2-04 is built.
    """
    raise NotImplementedError(f"P2-04: backtest {model} over {seasons}")


@report_app.command("score", help="Score a published prediction against the official result.")
def report_score(season: Season, round_: Round) -> None:
    """Score a published prediction against the official result (P8-07).

    Args:
        season: Season of the race.
        round_: Round number of the race.

    Raises:
        NotImplementedError: Always, until P8-07 is built.
    """
    raise NotImplementedError(f"P8-07: score {season} round {round_}")
