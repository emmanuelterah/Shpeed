"""Command-line interface and composition root.

This is the only module that knows concrete adapter classes: it reads settings, builds the
adapters, and injects them into the core use cases (dependency inversion). Each command
passes its ``--help`` text to Typer explicitly, so the docstrings can follow the Google style
without that markup appearing in the terminal.
"""

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, cast

import pandas as pd
import typer

from f1sim import live
from f1sim.adapters.benchmarks import check_regression, current_commit, record_run
from f1sim.adapters.config_files import (
    load_natural_experiments,
    load_pace_hyper,
    load_team_lineage,
)
from f1sim.adapters.models.pace import PacePredictor
from f1sim.adapters.models.pace_checks import predictive_check
from f1sim.adapters.models.pace_reference import fit_reference, prepare_laps
from f1sim.adapters.models.pace_training import SamplerSettings, season_observations, train_pace
from f1sim.adapters.models.registry import ModelOptions, get_model, pace_context
from f1sim.adapters.models.resets import (
    DEFAULT_HALF_LIFE,
    DEFAULT_PRACTICE_WEIGHT,
    DEFAULT_RESET_SD,
    STRATEGIES,
    ResetConfig,
)
from f1sim.adapters.models.tyres import (
    attach_track_temperature,
    fit_tyre_model,
    held_out_stint_rmse,
    temperature_gain,
)
from f1sim.adapters.sources.fastf1_adapter import FastF1Adapter, fastf1_loaders
from f1sim.adapters.sources.http import HttpClient
from f1sim.adapters.sources.jolpica_adapter import JolpicaAdapter
from f1sim.adapters.store.ducklake_store import DuckLakeStore
from f1sim.adapters.store.state_files import FileStateStore
from f1sim.cli_args import parse_components, parse_model_list, parse_seasons
from f1sim.core.domain.entities import EventKey, SessionKey, SnapshotId
from f1sim.core.errors import ShpeedError
from f1sim.core.ports.sources import SourceAdapter
from f1sim.core.usecases.backtest import BacktestResult, walk_forward
from f1sim.core.usecases.readers import FAR_FUTURE
from f1sim.core.usecases.sync import SyncReport, sync_season
from f1sim.observability import configure_logging
from f1sim.reports.ablation import render_ablation, run_ablation
from f1sim.reports.backtest_report import leaderboard, render_backtest
from f1sim.reports.data_quality import render_markdown, session_quality
from f1sim.reports.pit_windows import calibrate_pit_windows, render_pit_windows
from f1sim.reports.recovery import render_recovery, run_recovery
from f1sim.settings import Settings

app = typer.Typer(help="Shpeed F1 race simulator", no_args_is_help=True)
data_app = typer.Typer(help="Fetch, validate and manage the local data store", no_args_is_help=True)
report_app = typer.Typer(help="Backtest and scoring reports", no_args_is_help=True)
pace_app = typer.Typer(help="Fit, check and update the pace model", no_args_is_help=True)
tyres_app = typer.Typer(help="Fit and check the tyre and strategy models", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(report_app, name="report")
app.add_typer(pace_app, name="pace")
app.add_typer(tyres_app, name="tyres")

SOURCES = ("jolpica", "fastf1")
BENCHMARKS_DIR = Path("benchmarks")
CITATION_FOLDERS = (BENCHMARKS_DIR,)

Season = Annotated[int, typer.Option(help="Season, for example 2025")]
Round = Annotated[int, typer.Option("--round", help="Round number within the season")]
OptionalRound = Annotated[
    int | None, typer.Option("--round", help="Round number; omit when passing --latest")
]


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
    force: Annotated[
        bool, typer.Option("--force", help="Re-fetch and replace sessions already stored.")
    ] = False,
) -> None:
    """Fetch, validate and commit one season as a new snapshot (P1-17).

    Args:
        ctx: Typer context holding settings and the run ID.
        season: Season to sync.
        source: ``"all"`` or one source name.
        force: Re-fetch sessions already stored and replace them, for when the
            conversion has changed.

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
            force=force,
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
    if result.missing:
        typer.secho(
            f"{len(result.missing)} of {result.files_checked} files missing:", fg=typer.colors.RED
        )
        for path in result.missing:
            typer.echo(f"  {path}")
    for name, (recorded, actual) in result.mismatched.items():
        typer.secho(
            f"  {name}: {actual:,} rows, sync metadata recorded {recorded:,}",
            fg=typer.colors.RED,
        )
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
            found.update(_snapshot_ids(data))
    return found


def _snapshot_ids(data: Any) -> set[int]:
    """Every integer ``snapshot_id`` anywhere in parsed JSON, such as in benchmark runs."""
    if isinstance(data, list):
        return {i for item in cast(list[Any], data) for i in _snapshot_ids(item)}
    if not isinstance(data, dict):
        return set()
    record = cast(dict[str, Any], data)
    snapshot = record.get("snapshot_id")
    found: set[int] = set()
    if isinstance(snapshot, int) and not isinstance(snapshot, bool):
        found.add(snapshot)
    for value in record.values():
        found |= _snapshot_ids(value)
    return found


@app.command(help="Forecast one race and publish its prediction record.")
def predict(  # pylint: disable=too-many-arguments
    ctx: typer.Context,
    season: Season,
    round_: OptionalRound = None,
    *,
    model: Annotated[str, typer.Option(help="Registered model to forecast with")] = "simulator",
    latest: Annotated[
        bool,
        typer.Option("--latest", help="Use the newest weekend past qualifying (scheduled runs)"),
    ] = False,
    rain: Annotated[
        float, typer.Option("--rain", min=0.0, max=1.0, help="Rain chance fed to the simulator")
    ] = 0.0,
    commit: Annotated[
        bool, typer.Option("--commit/--no-commit", help="Commit the record to git")
    ] = True,
) -> None:
    """Forecast one race and publish its record (P8-01, P8-02, P8-03, P8-06).

    Before lights out a re-run with new data replaces the record and a re-run with the same
    result is a no-op; after lights out the forecast is locked (ADR-0011).

    Args:
        ctx: Typer context holding settings.
        season: Season of the race.
        round_: Round number; omit and pass ``--latest`` for the newest unraced weekend.
        model: Registered model name to forecast with.
        latest: Pick the newest weekend with qualifying and no race result.
        rain: Rain probability fed to the simulator.
        commit: Whether to commit the record and index to git.

    Raises:
        BadParameter: If ``--round`` and ``--latest`` are both or neither given, or the model
            name is unknown.
        Exit: With status 1 when the race cannot be forecast or the record is locked.
    """
    settings: Settings = ctx.obj["settings"]
    names = parse_model_list(model)
    if len(names) != 1:
        raise typer.BadParameter("predict takes exactly one model")
    options = replace(_model_options(settings), p_rain=rain)
    store = open_store(settings)
    try:
        outcome = live.run_predict(
            store,
            settings,
            live.Target(season, round_, latest),
            name=names[0],
            options=options,
            commit=commit,
            now=_now(),
        )
    except ShpeedError as exc:
        typer.secho(f"predict failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    finally:
        store.close()
    typer.secho(outcome.message, fg=typer.colors.GREEN if outcome.result else None)


@app.command(help="Walk-forward backtest of registered models, with report and benchmarks.")
def backtest(  # pylint: disable=too-many-arguments,too-many-locals
    ctx: typer.Context,
    *,
    model: Annotated[
        str, typer.Option(help="Model name, comma-separated names, or 'all'")
    ] = "plackett_luce",
    seasons: Annotated[str, typer.Option(help="Seasons, such as 2022-2025")] = "2022-2025",
    report: Annotated[Path | None, typer.Option(help="Write the Markdown report here")] = None,
    record: Annotated[bool, typer.Option(help="Append the scores to benchmarks/")] = False,
    check: Annotated[
        bool, typer.Option(help="Fail if RPS is worse than the best recorded run")
    ] = False,
    ablate: Annotated[
        str, typer.Option(help="Comma-separated event components to switch off (P6-09)")
    ] = "",
    workers: Annotated[int, typer.Option(help="Races forecast in parallel")] = 4,
) -> None:
    """Run a walk-forward backtest (P2-04) and report, record or gate it (P2-12 to P2-14).

    Args:
        ctx: Typer context holding settings.
        model: Registered model name, a comma-separated list, or ``all``.
        seasons: Seasons to forecast, such as ``2022-2025``.
        report: File for the Markdown report; None prints the leaderboard only.
        record: Append each model's scores to its benchmark file.
        check: Exit with status 1 if any model regresses against its benchmark.
        ablate: Comma-separated simulator event components to disable, for the ablation runner.
        workers: Races forecast in parallel.

    Raises:
        BadParameter: If an ablated component is not a known event component.
        typer.Exit: With status 1 when ``check`` finds a regression.
    """
    settings: Settings = ctx.obj["settings"]
    names = parse_model_list(model)
    years = parse_seasons(seasons)
    options = replace(_model_options(settings), disabled=parse_components(ablate))
    store = open_store(settings)
    try:
        snapshot = store.latest_snapshot()
        results = [
            walk_forward(
                store,
                snapshot=snapshot,
                seasons=years,
                model=get_model(name, options),
                seed=settings.seed,
                workers=workers,
            )
            for name in names
        ]
    finally:
        store.close()
    if report is None:
        typer.echo("\n".join(leaderboard(results)))
    else:
        report.write_text(render_backtest(results), encoding="utf-8")
        typer.echo(f"Wrote {report}")
    regressed = _gate_and_record(results, record=record, check=check, seed=settings.seed)
    if regressed:
        raise typer.Exit(1)


def _gate_and_record(
    results: Sequence[BacktestResult], *, record: bool, check: bool, seed: int
) -> bool:
    folder = BENCHMARKS_DIR
    regressed = False
    if check:
        for result in results:
            gate = check_regression(folder, result)
            typer.echo(gate.message)
            regressed |= not gate.passed
    if record:
        commit = current_commit()
        for result in results:
            typer.echo(f"Recorded {record_run(folder, result, commit=commit, seed=seed)}")
    return regressed


def _model_options(settings: Settings, *, with_state: bool = False) -> ModelOptions:
    return ModelOptions(
        pace_hyper=load_pace_hyper(settings.pace_hyper_path),
        state_store=FileStateStore(settings.state_dir) if with_state else None,
        rules_dir=settings.rules_dir,
        n_sims=settings.n_sims,
    )


@pace_app.command("fit", help="Fit the pace model's hyperparameters with NUTS (NumPyro).")
def pace_fit(
    ctx: typer.Context,
    *,
    seasons: Annotated[str, typer.Option(help="Training seasons, such as 2021-2023")],
    warmup: Annotated[int, typer.Option(help="NUTS warm-up draws per chain")] = 500,
    samples: Annotated[int, typer.Option(help="NUTS draws kept per chain")] = 500,
    chains: Annotated[int, typer.Option(help="NUTS chains (at least 2 for R-hat)")] = 2,
    out: Annotated[Path | None, typer.Option(help="Where to write the result")] = None,
    method: Annotated[
        str, typer.Option(help="nuts (full posterior, R-hat and ESS) or laplace (fast)")
    ] = "nuts",
) -> None:
    """Fit hyperparameters (P3-04, P3-11) and choose race-day noise by backtest.

    Args:
        ctx: Typer context holding settings.
        seasons: Training seasons; keep later seasons for ``pace check``.
        warmup: Warm-up draws per chain.
        samples: Kept draws per chain.
        chains: Number of chains.
        out: Output JSON; defaults to the ``pace_hyper_path`` setting.
        method: ``nuts`` or ``laplace``.

    Raises:
        typer.Exit: With status 1 when the fit did not converge.
    """
    settings: Settings = ctx.obj["settings"]
    context = pace_context(_model_options(settings))
    store = open_store(settings)
    try:
        result = train_pace(
            store,
            snapshot=store.latest_snapshot(),
            seasons=parse_seasons(seasons),
            context=context,
            sampler=SamplerSettings(warmup, samples, chains, settings.seed, method),
        )
    finally:
        store.close()
    for name, d in result.fit.diagnostics.items():
        flag = "" if d.converged or method != "nuts" else "  <- not converged"
        typer.echo(
            f"{name:22} {d.median:8.4f} sd {d.sd:.4f}  R-hat {d.r_hat:.3f}  ESS {d.ess:6.0f}{flag}"
        )
    typer.echo(f"order_noise {result.fit.hyper.order_noise} (RPS: {result.order_noise_rps})")
    typer.echo(f"NUTS took {result.fit.seconds:.1f} s for {result.fit.observations} observations")
    target = out or settings.pace_hyper_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.record(current_commit()), indent=2) + "\n", "utf-8")
    typer.echo(f"Wrote {target}")
    if not result.fit.converged:
        raise typer.Exit(1)


@pace_app.command("check", help="Posterior predictive check on held-out seasons.")
def pace_check(
    ctx: typer.Context,
    *,
    seasons: Annotated[str, typer.Option(help="All seasons to run, such as 2021-2024")],
    holdout: Annotated[str, typer.Option(help="Held-out seasons, such as 2024")],
) -> None:
    """Check that held-out race forecasts are calibrated (P3-11).

    Args:
        ctx: Typer context holding settings.
        seasons: Seasons to run the filter over, training and held-out.
        holdout: Seasons whose races are checked.

    Raises:
        typer.Exit: With status 1 when the check fails.
    """
    settings: Settings = ctx.obj["settings"]
    context = pace_context(_model_options(settings))
    store = open_store(settings)
    try:
        snapshot = store.latest_snapshot()
        observations = season_observations(store, snapshot, parse_seasons(seasons), context)
    finally:
        store.close()
    result = predictive_check(observations, context, parse_seasons(holdout))
    typer.echo(
        f"{result.sessions} races, {result.drivers} driver forecasts: "
        f"50% interval holds {result.coverage50:.0%}, 90% holds {result.coverage90:.0%}, "
        f"z sd {result.z_sd:.2f}, chi-squared ratio {result.chi2_ratio:.2f}"
    )
    typer.echo("PASS" if result.passed else "FAIL")
    if not result.passed:
        raise typer.Exit(1)


@pace_app.command("reference", help="Fit the mixed-effects reference model and report error.")
def pace_reference(
    ctx: typer.Context,
    *,
    seasons: Annotated[str, typer.Option(help="Seasons, such as 2024")],
) -> None:
    """Fit the statsmodels reference model (P3-03) and print held-out lap error.

    Args:
        ctx: Typer context holding settings.
        seasons: Seasons whose race laps are used.
    """
    settings: Settings = ctx.obj["settings"]
    store = open_store(settings)
    try:
        laps = store.read("laps", snapshot=store.latest_snapshot(), as_of=FAR_FUTURE)
    finally:
        store.close()
    laps = laps.loc[laps["season"].isin(list(parse_seasons(seasons)))]
    result = fit_reference(prepare_laps(laps, load_team_lineage()), seed=settings.seed)
    typer.echo(
        f"Held-out lap error: {result.rmse_pct:.3f}% with team and driver effects, "
        f"{result.baseline_rmse_pct:.3f}% without ({result.test_laps} laps; "
        f"trained on {result.train_laps})"
    )
    typer.echo(f"Team spread {result.team_sd:.3f}%, driver-within-team {result.driver_sd:.3f}%")
    for team, value in sorted(result.team_effects.items(), key=lambda kv: kv[1]):
        typer.echo(f"  {team:16} {value:+.3f}%")


@pace_app.command("update", help="Update the pace model with every session so far.")
def pace_update(ctx: typer.Context, season: Season, round_: Round) -> None:
    """Weekend update (P3-09): resume from the saved state and add new sessions.

    Args:
        ctx: Typer context holding settings.
        season: Season of the weekend.
        round_: Round of the weekend.
    """
    settings: Settings = ctx.obj["settings"]
    context = pace_context(_model_options(settings, with_state=True))
    started = _now()
    store = open_store(settings)
    try:
        model = PacePredictor(settings.seed, context)
        model.fit(store, snapshot=store.latest_snapshot(), cutoff=_now())
    finally:
        store.close()
    seconds = (_now() - started).total_seconds()
    report = model.report
    resumed = (
        f"resumed from {report.resumed_from.season} round {report.resumed_from.round}"
        if report.resumed_from
        else "fitted from the first session"
    )
    typer.echo(f"{report.sessions} sessions in {seconds:.1f} s ({resumed})")
    model.filter.advance(season, round_)
    _print_pace_table(model, EventKey(season, round_))


def _print_pace_table(model: PacePredictor, event: EventKey) -> None:
    typer.echo(f"Pace at {event.season} round {event.round} (percent of lap time; lower is faster)")
    labels = [lab for lab in model.filter.state.labels if lab.startswith(("team:", "dr:"))]
    rows = sorted((model.filter.effect(lab), lab) for lab in labels)
    for (mean, sd), label in rows:
        typer.echo(f"  {label:32} {mean:+.3f} +/- {sd:.3f}")


@tyres_app.command("fit", help="Fit degradation, warm-up, pit loss and stop times.")
def tyres_fit(
    ctx: typer.Context,
    *,
    out: Annotated[Path | None, typer.Option(help="Where to write the result")] = None,
    temperature: Annotated[
        bool, typer.Option("--temperature", help="Add a track-temperature interaction (P4-03).")
    ] = False,
) -> None:
    """Fit the tyre model on every stored race (P4-01 to P4-05).

    Args:
        ctx: Typer context holding settings.
        out: Output JSON; defaults to the ``tyre_model_path`` setting.
        temperature: Whether to add a track-temperature interaction (P4-03). Use
            ``f1sim tyres check --temperature`` to decide whether it earns its place.
    """
    settings: Settings = ctx.obj["settings"]
    store = open_store(settings)
    try:
        snapshot = store.latest_snapshot()
        laps = store.read("laps", snapshot=snapshot, as_of=FAR_FUTURE)
        events = store.read("events", snapshot=snapshot, as_of=FAR_FUTURE)
        pit_stops = store.read("pit_stops", snapshot=snapshot, as_of=FAR_FUTURE)
        if temperature:
            weather = store.read("weather", snapshot=snapshot, as_of=FAR_FUTURE)
            laps = attach_track_temperature(laps, weather)
    finally:
        store.close()
    model = fit_tyre_model(laps, events, pit_stops, temperature=temperature)
    target = out or settings.tyre_model_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), indent=2) + "\n", "utf-8")
    typer.echo(f"Compounds {', '.join(model.degradation.compounds)}")
    typer.echo(
        f"Fuel {model.fuel_s_per_lap:.4f} s/lap; pit loss {model.pit_loss_default.value:.2f} s"
    )
    typer.echo(f"Wrote {target}")


@tyres_app.command("check", help="Held-out stint error, the Phase 4 gate.")
def tyres_check(  # pylint: disable=too-many-locals
    ctx: typer.Context,
    *,
    seasons: Annotated[str, typer.Option(help="Training seasons, such as 2021-2023")],
    holdout: Annotated[str, typer.Option(help="Held-out seasons, such as 2024")],
    max_rmse: Annotated[float, typer.Option(help="Largest acceptable RMSE, seconds")] = 1.5,
    temperature: Annotated[
        bool,
        typer.Option(
            "--temperature", help="Test a track-temperature interaction and keep it if it helps."
        ),
    ] = False,
) -> None:
    """Check that held-out stint lap times match the model (Phase 4 gate).

    Args:
        ctx: Typer context holding settings.
        seasons: Seasons the model is fitted on.
        holdout: Seasons whose stint laps are measured.
        max_rmse: Largest held-out RMSE in seconds that still passes.
        temperature: Whether to test a track-temperature interaction (P4-03), keeping it only
            when it improves the held-out RMSE.

    Raises:
        typer.Exit: With status 1 when the held-out error is too large or nothing was measured.
    """
    settings: Settings = ctx.obj["settings"]
    train_seasons = parse_seasons(seasons)
    holdout_seasons = parse_seasons(holdout)
    store = open_store(settings)
    try:
        snapshot = store.latest_snapshot()
        laps = store.read("laps", snapshot=snapshot, as_of=FAR_FUTURE)
        events = store.read("events", snapshot=snapshot, as_of=FAR_FUTURE)
        pit_stops = store.read("pit_stops", snapshot=snapshot, as_of=FAR_FUTURE)
        if temperature:
            weather = store.read("weather", snapshot=snapshot, as_of=FAR_FUTURE)
            laps = attach_track_temperature(laps, weather)
    finally:
        store.close()
    train = laps.loc[laps["season"].isin(list(train_seasons))]
    train_stops = pit_stops.loc[pit_stops["season"].isin(list(train_seasons))]
    use_temperature = temperature and _temperature_earns_its_place(
        laps, events, train_seasons, holdout_seasons
    )
    model = fit_tyre_model(train, events, train_stops, temperature=use_temperature)
    rmse = held_out_stint_rmse(
        laps,
        events,
        model.degradation,
        holdout_seasons,
        fuel_s_per_lap=model.fuel_s_per_lap,
    )
    if not isfinite(rmse):
        held = ", ".join(str(s) for s in holdout_seasons)
        typer.echo(f"No held-out stint laps to measure for {held}")
        raise typer.Exit(1)
    typer.echo(f"Held-out stint RMSE {rmse:.3f} s (limit {max_rmse:.3f})")
    typer.echo("PASS" if rmse <= max_rmse else "FAIL")
    if rmse > max_rmse:
        raise typer.Exit(1)


def _temperature_earns_its_place(
    laps: pd.DataFrame,
    events: pd.DataFrame,
    train_seasons: Sequence[int],
    holdout_seasons: Sequence[int],
) -> bool:
    check = temperature_gain(
        laps,
        events,
        train_seasons=train_seasons,
        holdout_seasons=holdout_seasons,
    )
    verdict = "kept" if check.warranted else "dropped"
    typer.echo(
        f"Temperature: held-out RMSE {check.rmse_without:.3f} -> {check.rmse_with:.3f} ({verdict})"
    )
    return check.warranted


@tyres_app.command("windows", help="Share of actual stops inside predicted pit windows.")
def tyres_windows(
    ctx: typer.Context,
    *,
    train: Annotated[str, typer.Option(help="Training seasons, such as 2021-2023")],
    holdout: Annotated[str, typer.Option(help="Held-out seasons, such as 2024")],
    out: Annotated[Path | None, typer.Option(help="Write the Markdown here")] = None,
) -> None:
    """Report the pit-window calibration on held-out races (P4-10).

    Args:
        ctx: Typer context holding settings.
        train: Seasons the tyre model and tendencies are fitted on.
        holdout: Seasons whose races are measured.
        out: File for the Markdown report; None prints it.

    Raises:
        Exit: With status 1 when no held-out race could be measured.
    """
    settings: Settings = ctx.obj["settings"]
    store = open_store(settings)
    try:
        result = calibrate_pit_windows(
            store,
            store.latest_snapshot(),
            train_seasons=parse_seasons(train),
            holdout_seasons=parse_seasons(holdout),
            rules_dir=settings.rules_dir,
        )
    finally:
        store.close()
    markdown = render_pit_windows(result)
    if out is None:
        typer.echo(markdown)
    else:
        out.write_text(markdown, encoding="utf-8")
        typer.echo(f"Wrote {out}")
    if result.stops == 0:
        raise typer.Exit(1)


@report_app.command("score", help="Score a published prediction against the official result.")
def report_score(
    ctx: typer.Context,
    season: Season,
    round_: OptionalRound = None,
    *,
    latest: Annotated[
        bool,
        typer.Option("--latest", help="Use the newest race with a stored result (scheduled runs)"),
    ] = False,
    commit: Annotated[
        bool, typer.Option("--commit/--no-commit", help="Commit the scored record to git")
    ] = True,
) -> None:
    """Fill in the score of one published prediction and refresh the scoreboard (P8-07).

    Args:
        ctx: Typer context holding settings.
        season: Season of the race.
        round_: Round number; omit and pass ``--latest`` for the newest raced weekend.
        latest: Pick the newest weekend with an official race result.
        commit: Whether to commit the scored record and index to git.

    Raises:
        BadParameter: If ``--round`` and ``--latest`` are both or neither given.
        Exit: With status 1 when the race has no result or no published prediction.
    """
    settings: Settings = ctx.obj["settings"]
    store = open_store(settings)
    try:
        outcome = live.run_score(
            store, settings, live.Target(season, round_, latest), commit=commit, now=_now()
        )
    except ShpeedError as exc:
        typer.secho(f"scoring failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    finally:
        store.close()
    typer.secho(outcome.message, fg=typer.colors.GREEN if outcome.result else None)
    if outcome.score is not None:
        typer.echo(
            f"RPS {outcome.score.rps:.4f} · log loss {outcome.score.log_loss:.4f} · "
            f"Brier (win) {outcome.score.brier_win:.4f}"
        )


@report_app.command("ablation", help="Switch each event component off and score the change.")
def report_ablation(
    ctx: typer.Context,
    *,
    seasons: Annotated[str, typer.Option(help="Seasons, such as 2022-2025")] = "2022-2025",
    out: Annotated[Path, typer.Option(help="Where to write the Markdown table")] = Path(
        "docs/ablation.md"
    ),
    check: Annotated[
        bool, typer.Option(help="Exit non-zero if a kept component makes the backtest worse")
    ] = False,
    rain: Annotated[
        float, typer.Option("--rain", help="Rain chance mixed into every run (P6-07)")
    ] = 0.0,
    workers: Annotated[int, typer.Option(help="Races forecast in parallel")] = 4,
) -> None:
    """Publish the Phase 6 ablation table and, with ``--check``, fail on a harmful component.

    Args:
        ctx: Typer context holding settings.
        seasons: Seasons the backtest covers.
        out: File for the Markdown table.
        check: Exit with status 1 when a kept component makes the backtest worse.
        rain: Chance each simulated run is wet, mixed into every configuration.
        workers: Races forecast in parallel.

    Raises:
        typer.Exit: With status 1 when ``check`` finds a harmful component.
    """
    settings: Settings = ctx.obj["settings"]
    options = replace(_model_options(settings), p_rain=rain)
    store = open_store(settings)
    try:
        result = run_ablation(
            store,
            snapshot=store.latest_snapshot(),
            seasons=parse_seasons(seasons),
            seed=settings.seed,
            options=options,
            workers=workers,
        )
    finally:
        store.close()
    markdown = render_ablation(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    typer.echo(markdown)
    typer.echo(f"Wrote {out}")
    harmful = result.harmful()
    if check and harmful:
        typer.secho(f"Harmful components (drop these): {', '.join(harmful)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@report_app.command("recovery", help="Compare prior-handling strategies across a regulation reset.")
def report_recovery(  # pylint: disable=too-many-arguments,too-many-locals
    ctx: typer.Context,
    *,
    experiment: Annotated[
        str, typer.Option(help="Natural-experiment id from the manifest")
    ] = "reg-2022",
    strategies: Annotated[str, typer.Option(help="Comma-separated strategies to compare")] = (
        ",".join(STRATEGIES)
    ),
    out: Annotated[Path, typer.Option(help="Where to write the Markdown report")] = Path(
        "docs/research/regulation-reset.md"
    ),
    half_life: Annotated[
        float, typer.Option("--half-life", help="Decay half-life in races (P7-02)")
    ] = DEFAULT_HALF_LIFE,
    sd: Annotated[float, typer.Option(help="Reset prior spread, percent (P7-02)")] = (
        DEFAULT_RESET_SD
    ),
    practice_weight: Annotated[
        float,
        typer.Option("--practice-weight", help="Weight of the reset season's opening practice"),
    ] = DEFAULT_PRACTICE_WEIGHT,
    workers: Annotated[int, typer.Option(help="Races forecast in parallel")] = 4,
) -> None:
    """Run the regulation-reset study and write recovery curves (P7-03).

    Args:
        ctx: Typer context holding settings.
        experiment: Natural-experiment id from ``configs/natural-experiments.yaml``.
        strategies: Strategies to compare, comma-separated.
        out: File for the Markdown report.
        half_life: Half-life, in races, for the exponential-decay strategy.
        sd: Prior spread, in percent, for the reset strategies.
        practice_weight: Extra weight on the reset season's opening practice.
        workers: Races forecast in parallel.

    Raises:
        typer.Exit: With status 1 when the experiment is unknown or its data is not synced.
    """
    settings: Settings = ctx.obj["settings"]
    experiments = load_natural_experiments(settings.experiments_path)
    chosen = next((e for e in experiments if e.experiment_id == experiment), None)
    if chosen is None:
        known = ", ".join(e.experiment_id for e in experiments)
        typer.secho(f"Unknown experiment {experiment!r}; choose from {known}", fg=typer.colors.RED)
        raise typer.Exit(1)
    if chosen.snapshot is None:
        typer.secho(f"{experiment} is not synced: {chosen.notes}", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    names = tuple(n.strip() for n in strategies.split(",") if n.strip())
    config = ResetConfig(
        reset_season=chosen.reset_season,
        half_life=half_life,
        sd=sd,
        practice_weight=practice_weight,
    )
    store = open_store(settings)
    try:
        study = run_recovery(
            store,
            snapshot=SnapshotId(chosen.snapshot),
            experiment=chosen,
            strategies=names,
            seed=settings.seed,
            options=replace(_model_options(settings), reset=config),
            workers=workers,
        )
    finally:
        store.close()
    markdown = render_recovery(study)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    typer.echo(markdown)
    typer.echo(f"Wrote {out}")
