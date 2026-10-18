"""Planning a race forecast: the cutoff and the starting grid (ADR-0005).

Both the backtester and the live ``predict`` run need the same thing: everything known
before a race, frozen at the end of the last pre-race session, with the grid taken from
qualifying (or the sprint where it sets the grid). Keeping the logic here means a live
forecast and a backtest replay plan a race identically.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.race import RaceEntry
from f1sim.core.errors import DataError
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.readers import FAR_FUTURE

SPRINT_SETS_GRID = frozenset({2021, 2022})
"""Seasons in which the sprint result, not qualifying, decided the race grid."""

PRE_RACE_SESSIONS = ("Q", "S")
"""Sessions whose result is known before the race and may set the grid."""

LIVE_GRID_SESSION = "Q"
"""A live forecast waits for qualifying: on a sprint weekend the sprint result arrives first
and must not be mistaken for the race grid."""


@dataclass(frozen=True)
class RacePlan:
    """One race to forecast, with everything known at its cutoff.

    Attributes:
        event: The race weekend.
        cutoff: End of the last pre-race session; the model may see nothing later.
        entries: The starters, with their expected grid positions.
        classified: ``(driver_id, position)`` of every classified finisher, for scoring.
    """

    event: EventKey
    cutoff: datetime
    entries: tuple[RaceEntry, ...]
    classified: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class EventDetails:
    """Calendar facts about one race weekend.

    Attributes:
        name: Human-readable event name, or None when the calendar has no row for it.
        lights_out: Scheduled race start (naive UTC), or None when unknown.
    """

    name: str | None
    lights_out: datetime | None


@dataclass(frozen=True)
class SkippedRace:
    """A race the backtest could not forecast.

    Attributes:
        event: The race weekend.
        reason: Why it was skipped.
    """

    event: EventKey
    reason: str


def session_ends(laps: pd.DataFrame) -> dict[tuple[int, int], datetime]:
    """When the lap timing says each weekend's last pre-race session ended.

    Args:
        laps: The ``laps`` table.

    Returns:
        ``(season, round)`` to the latest pre-race session end recorded by lap timing.
    """
    if laps.empty:
        return {}
    pre = laps.loc[laps["session"].isin(list(PRE_RACE_SESSIONS))]
    ends: dict[tuple[int, int], datetime] = {}
    moments = pd.to_datetime(pre["available_at"]).tolist()
    for season, round_, moment in zip(
        pre["season"].tolist(), pre["round"].tolist(), moments, strict=True
    ):
        key = (int(season), int(round_))
        when = pd.Timestamp(moment).to_pydatetime()
        if key not in ends or when > ends[key]:
            ends[key] = when
    return ends


def grid_entries(session: pd.DataFrame) -> tuple[RaceEntry, ...]:
    """Starters in grid order: classified drivers by position, then the rest.

    Args:
        session: A ``results`` frame for the session that sets the grid.

    Returns:
        One entry per driver, in grid order.
    """
    drivers = [str(d) for d in session["driver_id"].tolist()]
    teams = [str(t) for t in session["team_id"].tolist()]
    positions = [math.nan if pd.isna(p) else float(p) for p in session["position"].tolist()]
    order = sorted(range(len(drivers)), key=lambda i: (np.isnan(positions[i]), positions[i]))
    return tuple(
        RaceEntry(driver_id=drivers[i], team_id=teams[i], grid=g)
        for g, i in enumerate(order, start=1)
    )


def plan_races(
    reader: DataReader, snapshot: SnapshotId, seasons: Sequence[int]
) -> tuple[list[RacePlan], list[SkippedRace]]:
    """Find every scorable race in some seasons and what was known before each.

    Planning reads results with no cutoff because it is the evaluator, not a model;
    models only ever see data through the ``CutoffReader`` built from each plan. The cutoff
    is the latest moment any source records for the weekend's pre-race sessions ending, from
    the results and from the lap timing (ADR-0005): one source may know only the date of a
    session and guess its time, which would otherwise hide that session from the models.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        seasons: Seasons to include.

    Returns:
        The races to forecast in calendar order, and the races that had to be skipped.
    """
    results = reader.read("results", snapshot=snapshot, as_of=FAR_FUTURE)
    results = results.loc[results["season"].isin(list(seasons))]
    ends = session_ends(reader.read("laps", snapshot=snapshot, as_of=FAR_FUTURE))
    plans: list[RacePlan] = []
    skipped: list[SkippedRace] = []
    races = results.loc[results["session"] == "R", ["season", "round"]].drop_duplicates()
    for season, round_ in sorted(map(tuple, races.to_numpy().tolist())):
        event = EventKey(int(season), int(round_))
        weekend: pd.DataFrame = results.loc[
            (results["season"] == event.season) & (results["round"] == event.round)
        ]
        plan = _plan_one(event, weekend, ends.get((event.season, event.round)))
        if isinstance(plan, SkippedRace):
            skipped.append(plan)
        else:
            plans.append(plan)
    return plans, skipped


def plan_event(reader: DataReader, snapshot: SnapshotId, event: EventKey) -> RacePlan:
    """Plan a single live race: its cutoff and grid, from what is stored now.

    Unlike a backtest, the race result need not exist yet; a weekend whose qualifying is
    synced is enough, and qualifying is required (``LIVE_GRID_SESSION``) so a sprint result
    is never published as the race grid. The cutoff is the latest pre-race session end the
    store records.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        event: The race weekend to plan.

    Returns:
        The plan, ready to fit a model and forecast.

    Raises:
        DataError: If the weekend has no qualifying result to build a grid from.
    """
    results = reader.read("results", snapshot=snapshot, as_of=FAR_FUTURE)
    weekend: pd.DataFrame = results.loc[
        (results["season"] == event.season) & (results["round"] == event.round)
    ]
    if LIVE_GRID_SESSION not in set(weekend["session"].tolist()):
        raise DataError(f"no qualifying result stored for {event.season} round {event.round}")
    ends = session_ends(reader.read("laps", snapshot=snapshot, as_of=FAR_FUTURE))
    plan = _plan_one(event, weekend, ends.get((event.season, event.round)))
    if isinstance(plan, SkippedRace):
        raise DataError(f"cannot forecast {event.season} round {event.round}: {plan.reason}")
    return plan


def official_classification(
    reader: DataReader, snapshot: SnapshotId, event: EventKey
) -> tuple[tuple[str, int], ...]:
    """Read the official race classification for scoring (P8-07).

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        event: The race weekend.

    Returns:
        ``(driver_id, position)`` for every classified finisher, in store order.

    Raises:
        DataError: If the race has no stored result yet.
    """
    results = reader.read("results", snapshot=snapshot, as_of=FAR_FUTURE)
    race = results.loc[
        (results["season"] == event.season)
        & (results["round"] == event.round)
        & (results["session"] == "R")
    ]
    if race.empty:
        raise DataError(f"no race result for {event.season} round {event.round}")
    return tuple(
        (str(d), int(p))
        for d, p in zip(race["driver_id"].tolist(), race["position"].tolist(), strict=True)
        if not pd.isna(p)
    )


def latest_unraced(reader: DataReader, snapshot: SnapshotId, season: int) -> EventKey | None:
    """Find the newest weekend in a season that is past qualifying but not yet raced.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        season: Season to search.

    Returns:
        The event key of the latest such weekend, or None when no weekend is waiting for a
        forecast (a scheduled run outside a race weekend).
    """
    results = reader.read("results", snapshot=snapshot, as_of=FAR_FUTURE)
    results = results.loc[results["season"] == season]
    raced = set(results.loc[results["session"] == "R", "round"].tolist())
    qualified = results.loc[results["session"] == LIVE_GRID_SESSION, "round"].tolist()
    waiting = sorted(int(r) for r in set(qualified) - raced)
    return EventKey(season, waiting[-1]) if waiting else None


def latest_raced(reader: DataReader, snapshot: SnapshotId, season: int) -> EventKey | None:
    """Find the newest weekend in a season that has an official race result.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        season: Season to search.

    Returns:
        The event key of the latest raced weekend, or None when no race is stored yet.
    """
    results = reader.read("results", snapshot=snapshot, as_of=FAR_FUTURE)
    results = results.loc[(results["season"] == season) & (results["session"] == "R")]
    rounds = sorted(int(r) for r in results["round"].unique().tolist())
    return EventKey(season, rounds[-1]) if rounds else None


def event_details(reader: DataReader, snapshot: SnapshotId, event: EventKey) -> EventDetails:
    """Look up a race weekend's name and scheduled start in the calendar.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        event: The race weekend.

    Returns:
        The event's name and lights-out time; both None when the calendar has no row.
    """
    events = reader.read("events", snapshot=snapshot, as_of=FAR_FUTURE)
    if events.empty:
        return EventDetails(name=None, lights_out=None)
    rows = events.loc[(events["season"] == event.season) & (events["round"] == event.round)]
    if rows.empty:
        return EventDetails(name=None, lights_out=None)
    name = rows["event_name"].iloc[0]
    start = rows["event_date"].iloc[0]
    return EventDetails(
        name=None if pd.isna(name) else str(name),
        lights_out=None if pd.isna(start) else _naive_utc(pd.Timestamp(start)),
    )


def _naive_utc(moment: pd.Timestamp) -> datetime:
    utc = moment.tz_convert("UTC").tz_localize(None) if moment.tzinfo is not None else moment
    return utc.to_pydatetime()


def _plan_one(
    event: EventKey, weekend: pd.DataFrame, timing_end: datetime | None
) -> RacePlan | SkippedRace:
    pre = weekend.loc[weekend["session"].isin(list(PRE_RACE_SESSIONS))]
    if pre.empty:
        return SkippedRace(event, "no qualifying or sprint results before the race")
    race = weekend.loc[weekend["session"] == "R"]
    grid_session = "Q" if "Q" in set(pre["session"].tolist()) else "S"
    sprint = "S" in set(pre["session"].tolist())
    if event.season in SPRINT_SETS_GRID and sprint:
        grid_session = "S"
    entries = grid_entries(pre.loc[pre["session"] == grid_session])
    classified = tuple(
        (str(d), int(p))
        for d, p in zip(race["driver_id"].tolist(), race["position"].tolist(), strict=True)
        if not pd.isna(p)
    )
    cutoff = pd.Timestamp(pre["available_at"].max()).to_pydatetime()
    if len(race):
        race_known = pd.Timestamp(pd.to_datetime(race["available_at"]).min()).to_pydatetime()
        if timing_end is not None and timing_end < race_known:
            cutoff = max(cutoff, timing_end)
    elif timing_end is not None:
        cutoff = max(cutoff, timing_end)
    return RacePlan(event, cutoff, entries, classified)
