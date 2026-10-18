"""Composition helpers for the live ``predict`` and ``report score`` commands (P8-01, P8-06).

The CLI is the composition root. These helpers hold the wiring that turns a data store, a
model registry entry, the schema validator and the git publisher into one published or
scored record, so the command definitions in ``cli.py`` stay short. Scheduled runs pass
``--latest``; when there is nothing to do (no weekend waiting, or the record is already
locked) they report it and succeed, so a cron that fires outside a race weekend is a no-op.
"""

from dataclasses import dataclass
from datetime import datetime

import typer

from f1sim import __version__
from f1sim.adapters.benchmarks import current_commit
from f1sim.adapters.models.registry import ModelOptions, get_model
from f1sim.adapters.publishers.git_pages import GitPagesPublisher
from f1sim.adapters.records import JsonSchemaValidator
from f1sim.adapters.store.ducklake_store import DuckLakeStore
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.record import RecordScore
from f1sim.core.errors import DataError
from f1sim.core.ports.publisher import PublishResult
from f1sim.core.usecases.backtest import race_seed
from f1sim.core.usecases.planning import (
    event_details,
    latest_raced,
    latest_unraced,
    official_classification,
)
from f1sim.core.usecases.predict import predict_event, version_string
from f1sim.core.usecases.publish import is_locked, publish_record
from f1sim.core.usecases.score import score_record
from f1sim.settings import Settings


@dataclass(frozen=True)
class RunOutcome:
    """What a predict or score run did.

    Attributes:
        message: One line for the terminal and the workflow log.
        result: The publish result, or None when nothing needed doing.
        score: The record's score after a scoring run, when it has one.
    """

    message: str
    result: PublishResult | None = None
    score: RecordScore | None = None


@dataclass(frozen=True)
class Target:
    """Which race a command acts on.

    Attributes:
        season: Season of the race.
        round: Round number, or None to pick one automatically.
        latest: Pick the newest matching weekend (the scheduled runs).
    """

    season: int
    round: int | None
    latest: bool


def run_predict(  # pylint: disable=too-many-arguments
    store: DuckLakeStore,
    settings: Settings,
    target: Target,
    *,
    name: str,
    options: ModelOptions,
    commit: bool,
    now: datetime,
) -> RunOutcome:
    """Forecast one race and publish its record.

    Args:
        store: The data store; the caller owns its lifetime.
        settings: Resolved settings.
        target: The race to forecast.
        name: Registry name of the model to forecast with.
        options: Model options, including the rain chance.
        commit: Whether to commit the record and index to git.
        now: The current time (naive UTC), for the lights-out lock.

    Returns:
        What happened, including a no-op when no weekend is waiting.

    Raises:
        DataError: If the weekend cannot be planned or git fails.
        RecordLockedError: If the race has started and no forecast was published, or a
            different forecast would replace one after lights out.
    """
    snapshot = store.latest_snapshot()
    event = resolve_event(store, snapshot, target, predicting=True)
    if event is None:
        return RunOutcome(f"no weekend in {target.season} is waiting for a forecast")
    details = event_details(store, snapshot, event)
    publisher = GitPagesPublisher(settings.predictions_dir, commit=commit)
    if target.latest and is_locked(details.lights_out, now) and publisher.load(event):
        return RunOutcome(f"{_race(event)} has started; its forecast is locked")
    record = predict_event(
        store,
        snapshot=snapshot,
        model=get_model(name, options),
        event=event,
        seed=race_seed(settings.seed, event),
        version=version_string(name, __version__),
        git_commit=current_commit(),
        n_sims=settings.n_sims,
        event_name=details.name,
        p_rain=options.p_rain,
    )
    result = publish_record(
        record,
        validator=JsonSchemaValidator(),
        records=publisher,
        publisher=publisher,
        lights_out=details.lights_out,
        now=now,
    )
    verb = "published" if result.changed else "already up to date"
    return RunOutcome(f"{verb}: {result.location}", result=result)


def run_score(
    store: DuckLakeStore,
    settings: Settings,
    target: Target,
    *,
    commit: bool,
    now: datetime,
) -> RunOutcome:
    """Score one published prediction against the official result.

    Args:
        store: The data store; the caller owns its lifetime.
        settings: Resolved settings.
        target: The race to score.
        commit: Whether to commit the scored record and index to git.
        now: The current time (naive UTC), stored as the scoring time.

    Returns:
        What happened and the score; a no-op when ``--latest`` finds no race result, or
        the newest race was never forecast.

    Raises:
        DataError: If an explicitly chosen race has no result or no published prediction.
    """
    snapshot = store.latest_snapshot()
    event = resolve_event(store, snapshot, target, predicting=False)
    if event is None:
        return RunOutcome(f"no race in {target.season} has a result to score yet")
    publisher = GitPagesPublisher(settings.predictions_dir, commit=commit)
    record = publisher.load(event)
    if record is None:
        message = f"no published prediction for {_race(event)}"
        if target.latest:
            return RunOutcome(f"{message}; nothing to score")
        raise DataError(message)
    classified = official_classification(store, snapshot, event)
    scored = score_record(record, classified, scored_at=now)
    result = publish_record(
        scored,
        validator=JsonSchemaValidator(),
        records=publisher,
        publisher=publisher,
        lights_out=event_details(store, snapshot, event).lights_out,
        now=now,
    )
    verb = "scored" if result.changed else "already scored"
    return RunOutcome(f"{verb}: {result.location}", result=result, score=scored.score)


def resolve_event(
    store: DuckLakeStore, snapshot: SnapshotId, target: Target, *, predicting: bool
) -> EventKey | None:
    """Resolve the race a command acts on, from ``--round`` or ``--latest``.

    Args:
        store: The data store.
        snapshot: Snapshot to read.
        target: The ``--season``, ``--round`` and ``--latest`` options.
        predicting: True for a forecast (newest unraced), False for scoring (newest raced).

    Returns:
        The event to act on, or None when ``--latest`` finds nothing to do.

    Raises:
        BadParameter: If ``--round`` and ``--latest`` are both or neither given.
    """
    if target.latest and target.round is not None:
        raise typer.BadParameter("give --round or --latest, not both")
    if target.latest:
        find = latest_unraced if predicting else latest_raced
        return find(store, snapshot, target.season)
    if target.round is None:
        raise typer.BadParameter("give --round or --latest")
    return EventKey(target.season, target.round)


def _race(event: EventKey) -> str:
    return f"{event.season} round {event.round}"
