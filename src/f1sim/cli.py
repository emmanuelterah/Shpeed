"""Command-line interface and composition root.

This is the only module that knows concrete adapter classes: it reads settings,
builds the adapters, and injects them into the core use cases (dependency inversion).
Heavy imports happen inside commands, so ``f1sim --help`` is fast and works without extras.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from f1sim.observability import configure_logging
from f1sim.settings import Settings

if TYPE_CHECKING:
    from f1sim.adapters.store.ducklake_store import DuckLakeStore
    from f1sim.core.ports.sources import SourceAdapter

app = typer.Typer(help="Shpeed F1 race simulator", no_args_is_help=True)
data_app = typer.Typer(help="Fetch, validate and manage the local data store", no_args_is_help=True)
report_app = typer.Typer(help="Backtest and scoring reports", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(report_app, name="report")

SOURCES = ("jolpica", "fastf1")


@app.callback()
def main(ctx: typer.Context) -> None:
    """Configure logging for every command."""
    settings = Settings()
    ctx.obj = {"settings": settings, "run_id": configure_logging(settings.log_level)}


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _missing_extra(exc: ImportError) -> typer.Exit:
    typer.secho(
        f"Missing dependency ({exc.name}). Install the data tools with: uv sync --extra data",
        fg=typer.colors.RED,
        err=True,
    )
    return typer.Exit(2)


def open_store(settings: Settings) -> "DuckLakeStore":
    from f1sim.adapters.store.ducklake_store import DuckLakeStore

    return DuckLakeStore(settings.data_dir / "lake")


def build_sources(settings: Settings, names: Sequence[str]) -> list["SourceAdapter"]:
    from f1sim.adapters.sources.fastf1_adapter import FastF1Adapter, fastf1_loaders
    from f1sim.adapters.sources.http import HttpClient
    from f1sim.adapters.sources.jolpica_adapter import JolpicaAdapter

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


@data_app.command("sync")
def data_sync(
    ctx: typer.Context,
    season: int = typer.Option(..., help="Season to sync, e.g. 2025"),
    source: str = typer.Option("all", help="all, jolpica or fastf1"),
) -> None:
    """Fetch, validate and commit one season as a new snapshot (P1-17)."""
    from f1sim.core.usecases.sync import sync_season

    settings: Settings = ctx.obj["settings"]
    names = SOURCES if source == "all" else (source,)
    if any(n not in SOURCES for n in names):
        raise typer.BadParameter(f"source must be all, {' or '.join(SOURCES)}")
    try:
        sources = build_sources(settings, names)
        store = open_store(settings)
    except ImportError as exc:
        raise _missing_extra(exc) from exc

    typer.echo(f"Syncing {season} from {', '.join(names)} into {settings.data_dir / 'lake'}")
    try:
        report = sync_season(
            season,
            sources=sources,
            writer=store,
            ledger=store,
            run_id=ctx.obj["run_id"],
            now=_now(),
            on_progress=lambda src, s: typer.echo(
                f"  {src:<8} round {s.event.round:>2}  {s.session.value}"
            ),
        )
    finally:
        store.close()

    if report.committed:
        rows = ", ".join(f"{k} {v:,}" for k, v in sorted(report.rows.items()))
        typer.secho(f"Committed snapshot {report.snapshot_id}: {rows}", fg=typer.colors.GREEN)
    else:
        typer.echo("Nothing new to store; no snapshot created.")
    typer.echo(f"{len(report.fetched)} sessions fetched, {report.skipped} already stored.")
    for f in report.failures:
        where = (
            f"round {f.session.event.round} {f.session.session.value}" if f.session else "schedule"
        )
        typer.secho(f"  skipped {f.source} {where}: {f.reason}", fg=typer.colors.YELLOW)
    if report.failures and not report.fetched:
        raise typer.Exit(1)


@data_app.command("verify")
def data_verify(ctx: typer.Context) -> None:
    """Check every data file the catalog references (P1-14)."""
    try:
        store = open_store(ctx.obj["settings"])
    except ImportError as exc:
        raise _missing_extra(exc) from exc
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


@data_app.command("report")
def data_report(
    ctx: typer.Context,
    season: int = typer.Option(..., help="Season to report on"),
    out: Path | None = typer.Option(None, help="Write Markdown here instead of printing it"),
) -> None:
    """Per-session data-quality report (P1-16)."""
    from f1sim.reports.data_quality import render_markdown, session_quality

    try:
        store = open_store(ctx.obj["settings"])
    except ImportError as exc:
        raise _missing_extra(exc) from exc
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


@data_app.command("gc")
def data_gc(
    ctx: typer.Context,
    keep_last: int = typer.Option(5, help="Newest snapshots to keep"),
) -> None:
    """Expire old snapshots, never one a published prediction or benchmark cites (P1-19)."""
    from f1sim.core.domain.entities import SnapshotId

    settings: Settings = ctx.obj["settings"]
    cited = frozenset(
        SnapshotId(s) for s in cited_snapshots([settings.predictions_dir, Path("benchmarks")])
    )
    try:
        store = open_store(settings)
    except ImportError as exc:
        raise _missing_extra(exc) from exc
    try:
        expired = store.expire_snapshots(keep_last=keep_last, protect=cited)
    finally:
        store.close()
    typer.echo(f"Expired {len(expired)} snapshots; protected {len(cited)} cited ones.")


def cited_snapshots(folders: Sequence[Path]) -> set[int]:
    """Snapshot IDs referenced by any JSON file under ``folders``."""
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


@app.command()
def predict(season: int = typer.Option(...), round_: int = typer.Option(..., "--round")) -> None:
    """Predict one race and write its prediction record (P8-01)."""
    raise NotImplementedError(f"P8-01: predict {season} round {round_}")


@app.command()
def backtest(
    model: str = typer.Option("plackett_luce"),
    seasons: str = typer.Option("2022-2025"),
) -> None:
    """Walk-forward backtest of a registered model (P2-04)."""
    raise NotImplementedError(f"P2-04: backtest {model} over {seasons}")


@report_app.command("score")
def report_score(
    season: int = typer.Option(...), round_: int = typer.Option(..., "--round")
) -> None:
    """Score a published prediction against the official result (P8-07)."""
    raise NotImplementedError(f"P8-07: score {season} round {round_}")
