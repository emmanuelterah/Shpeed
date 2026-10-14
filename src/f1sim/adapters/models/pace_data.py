"""Turn laps into pace observations: one number per driver per session (P3-01 to P3-03).

Each observation is a driver's corrected pace in a session, in percent of the session's
median driver, so 0.5 means half a percent slower than the typical car (about 0.45 s on a
90 s lap). Percent keeps circuits of different lengths on one scale.

- **Qualifying and sprint qualifying** give a *qualifying* observation: the best push lap,
  corrected for track evolution and compound.
- **Races and sprints** give a *race* observation: pace with fuel, tyre wear and compound
  removed.
- **Practice** gives a *race* observation from long runs (stints of several clean laps),
  the best pre-race signal of race pace, though noisier because fuel loads are unknown.

Sessions are processed in time order, and each one's fuel and evolution estimates are
pooled only with earlier sessions at the same circuit, so an observation never depends on
anything that happened after it.
"""

from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1sim.adapters.models.pace_corrections import (
    EVOLUTION_PRIOR,
    FUEL_PRIOR,
    CircuitHistory,
    Estimate,
    SessionFit,
    dry_laps,
    fit_push_laps,
    fit_race_laps,
    pool,
)
from f1sim.core.domain.reference import CircuitTable
from f1sim.core.usecases.readers import as_naive_utc

QUALI_SESSIONS = frozenset({"Q", "SQ"})
RACE_SESSIONS = frozenset({"R", "S"})
PRACTICE_SESSIONS = frozenset({"FP1", "FP2", "FP3"})
SESSION_ORDER = {"FP1": 0, "FP2": 1, "FP3": 2, "SQ": 3, "S": 4, "Q": 5, "R": 6}

QUALI, RACE = "q", "r"
"""Observation kinds: qualifying pace and race pace."""

MIN_RACE_LAPS = 5
"""Clean laps a driver needs in a race, or in one practice stint, to be observed."""

MIN_DRIVERS = 4
"""Drivers a session needs before its median is a meaningful reference."""

PERCENT = 100.0

_Key = tuple[int, int, str]


@dataclass(frozen=True)
class SessionObs:
    """Every driver's corrected pace in one session.

    Attributes:
        season: Season of the session.
        round: Round of the session.
        session: Session code, such as ``Q``.
        circuit: Jolpica circuit ID, or ``""`` when the event is unknown.
        cluster: Speed cluster of the circuit: ``low``, ``medium`` or ``high``.
        kind: ``q`` for qualifying pace, ``r`` for race pace.
        drivers: Driver IDs.
        teams: Constructor ID of each driver in this session.
        values: Pace in percent of the session's median driver; lower is faster.
        available_at: When the session's data became known (naive UTC).
    """

    season: int
    round: int
    session: str
    circuit: str
    cluster: str
    kind: str
    drivers: tuple[str, ...]
    teams: tuple[str, ...]
    values: NDArray[np.float64]
    available_at: datetime

    @property
    def group(self) -> str:
        """Noise group: ``Q``, ``R``, ``S`` or ``FP``; each has its own noise level.

        Returns:
            The group code.
        """
        if self.session in PRACTICE_SESSIONS:
            return "FP"
        return "Q" if self.session in QUALI_SESSIONS else self.session


@dataclass(frozen=True)
class _Result:
    obs: SessionObs | None
    fuel: Estimate | None = None
    evolution: Estimate | None = None


@dataclass
class ObservationCache:
    """Session results already computed, shared by every race of one backtest.

    A session's result depends only on that session and earlier ones in the same
    snapshot, so it never changes once computed.

    Attributes:
        results: ``(store key, season, round, session)`` to the session's result.
        lock: Guards ``results``; backtest races run in threads.
    """

    results: dict[Hashable, _Result] = field(default_factory=dict[Hashable, _Result])
    lock: Lock = field(default_factory=Lock)

    def get(self, key: Hashable) -> _Result | None:
        """Look up a session.

        Args:
            key: ``(store key, season, round, session)``.

        Returns:
            The cached result, or None.
        """
        with self.lock:
            return self.results.get(key)

    def put(self, key: Hashable, result: _Result) -> None:
        """Remember a session's result.

        Args:
            key: ``(store key, season, round, session)``.
            result: The result to keep.
        """
        with self.lock:
            self.results[key] = result


def build_observations(
    laps: pd.DataFrame,
    events: pd.DataFrame,
    circuits: CircuitTable,
    *,
    cache: ObservationCache | None = None,
    store_key: Hashable = None,
    history: CircuitHistory | None = None,
    window: tuple[tuple[int, int] | None, tuple[int, int] | None] = (None, None),
) -> list[SessionObs]:
    """Build pace observations for every session in some laps, in time order.

    Args:
        laps: Canonical ``laps`` rows.
        events: Canonical ``events`` rows, for each weekend's circuit.
        circuits: Circuit lengths and speed bands.
        cache: Reuse results computed earlier from the same data.
        store_key: Identifies the data (such as the snapshot) in the cache; the cache is
            used only when this is given.
        history: Fuel and evolution estimates from earlier sessions; updated in place, so
            a later call can continue where this one stopped.
        window: ``(after, until)`` weekends: only sessions of weekends later than
            ``after`` and no later than ``until`` are processed (None means no limit).

    Returns:
        One observation per session with enough data, oldest first.
    """
    circuit_of = circuit_map(events)
    history = history if history is not None else CircuitHistory()
    clusters: dict[tuple[int, int], str] = {}
    out: list[SessionObs] = []
    for key, session_laps in _sessions(laps):
        event = (key[0], key[1])
        if not _in_window(event, window):
            continue
        circuit = circuit_of.get(event, "")
        if event not in clusters:
            clusters[event] = event_cluster(laps, event, circuit, circuits)
        result = cache.get((store_key, *key)) if cache and store_key is not None else None
        if result is None:
            result = _observe(session_laps, key, circuit, clusters[event], history)
            if cache is not None and store_key is not None:
                cache.put((store_key, *key), result)
        history.add(circuit, fuel=result.fuel, evolution=result.evolution)
        if result.obs is not None:
            out.append(result.obs)
    return out


def _in_window(
    event: tuple[int, int], window: tuple[tuple[int, int] | None, tuple[int, int] | None]
) -> bool:
    after, until = window
    return (after is None or event > after) and (until is None or event <= until)


def _sessions(laps: pd.DataFrame) -> Iterator[tuple[_Key, pd.DataFrame]]:
    if laps.empty:
        return
    seasons = [int(v) for v in laps["season"].tolist()]
    rounds = [int(v) for v in laps["round"].tolist()]
    sessions = [str(v) for v in laps["session"].tolist()]
    groups: dict[_Key, list[int]] = {}
    for i, key in enumerate(zip(seasons, rounds, sessions, strict=True)):
        groups.setdefault(key, []).append(i)
    order = sorted(groups, key=lambda k: (k[0], k[1], SESSION_ORDER.get(k[2], 9)))
    for key in order:
        yield key, laps.iloc[groups[key]]


def circuit_map(events: pd.DataFrame) -> dict[tuple[int, int], str]:
    """Map each weekend to its circuit.

    Args:
        events: Canonical ``events`` rows.

    Returns:
        ``(season, round)`` to circuit ID.
    """
    if events.empty:
        return {}
    keys = zip(events["season"].tolist(), events["round"].tolist(), strict=True)
    names = events["circuit_id"].fillna("").astype(str).tolist()
    return {(int(s), int(r)): name for (s, r), name in zip(keys, names, strict=True)}


def event_cluster(
    laps: pd.DataFrame, event: tuple[int, int], circuit: str, circuits: CircuitTable
) -> str:
    """Find a weekend's speed cluster from its fastest qualifying lap, else its fastest lap.

    Args:
        laps: Canonical ``laps`` rows; only the weekend's are used.
        event: ``(season, round)``.
        circuit: Jolpica circuit ID.
        circuits: Circuit lengths and speed bands.

    Returns:
        ``low``, ``medium`` or ``high``; ``medium`` when there are no laps.
    """
    weekend = laps.loc[(laps["season"] == event[0]) & (laps["round"] == event[1])]
    dry = dry_laps(weekend)
    quali = dry.loc[dry["session"].isin(list(QUALI_SESSIONS))]
    source = quali if not quali.empty else dry
    if source.empty:
        return circuits.cluster(None)
    best = float(source["lap_time_s"].min())
    return circuits.cluster(circuits.average_speed(circuit, best))


def _observe(
    laps: pd.DataFrame, key: _Key, circuit: str, cluster: str, history: CircuitHistory
) -> _Result:
    session = key[2]
    if session in RACE_SESSIONS:
        return _race(laps, key, circuit, cluster, history)
    if session in QUALI_SESSIONS:
        return _quali(laps, key, circuit, cluster, history)
    if session in PRACTICE_SESSIONS:
        fuel = history.fuel_at(circuit).value
        long_runs = _long_runs(laps)
        fit = fit_race_laps(long_runs, fuel=fuel) if not long_runs.empty else None
        return _Result(
            _to_obs(fit, laps, key, (circuit, cluster), kind=RACE, counts=_counts(long_runs))
        )
    return _Result(None)


def _race(
    laps: pd.DataFrame, key: _Key, circuit: str, cluster: str, history: CircuitHistory
) -> _Result:
    own = fit_race_laps(laps)
    fuel = Estimate(-own.slope.value, own.slope.se) if own and own.slope else None
    pooled = pool([*history.fuel.get(circuit, []), *([fuel] if fuel else [])], FUEL_PRIOR)
    fit = fit_race_laps(laps, fuel=pooled.value)
    counts = _counts(dry_laps(laps))
    return _Result(_to_obs(fit, laps, key, (circuit, cluster), kind=RACE, counts=counts), fuel=fuel)


def _quali(
    laps: pd.DataFrame, key: _Key, circuit: str, cluster: str, history: CircuitHistory
) -> _Result:
    own = fit_push_laps(laps)
    evolution = own.slope if own else None
    estimates = [*history.evolution.get(circuit, []), *([evolution] if evolution else [])]
    pooled = pool(estimates, EVOLUTION_PRIOR)
    fit = fit_push_laps(laps, evolution=pooled.value)
    counts = dict.fromkeys(fit.drivers, MIN_RACE_LAPS) if fit else {}
    obs = _to_obs(fit, laps, key, (circuit, cluster), kind=QUALI, counts=counts)
    return _Result(obs, evolution=evolution)


def _long_runs(laps: pd.DataFrame) -> pd.DataFrame:
    """Clean dry laps from practice stints with enough laps, numbered within the stint."""
    dry = dry_laps(laps)
    if dry.empty:
        return dry
    keys = [dry["driver_id"].astype(str), dry["stint"].fillna(0)]
    size = dry.groupby(keys)["lap_number"].transform("size")
    runs = dry.loc[size >= MIN_RACE_LAPS]
    in_stint = runs.groupby([runs["driver_id"].astype(str), runs["stint"].fillna(0)])[
        "lap_number"
    ].rank(method="first")
    return runs.assign(lap_number=in_stint.astype(np.float64))


def _counts(laps: pd.DataFrame) -> dict[str, int]:
    if laps.empty:
        return {}
    counts = laps["driver_id"].astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def _to_obs(
    fit: SessionFit | None,
    laps: pd.DataFrame,
    key: _Key,
    place: tuple[str, str],
    *,
    kind: str,
    counts: dict[str, int],
) -> SessionObs | None:
    if fit is None:
        return None
    drivers = sorted(d for d in fit.drivers if counts.get(d, 0) >= MIN_RACE_LAPS)
    if len(drivers) < MIN_DRIVERS:
        return None
    seconds = np.array([fit.drivers[d] for d in drivers], dtype=np.float64)
    reference = float(np.median(seconds))
    team_of = dict(
        zip(
            laps["driver_id"].astype(str).tolist(),
            laps["team_id"].astype(str).tolist(),
            strict=True,
        )
    )
    moment = pd.Timestamp(pd.to_datetime(laps["available_at"]).max()).to_pydatetime()
    return SessionObs(
        season=key[0],
        round=key[1],
        session=key[2],
        circuit=place[0],
        cluster=place[1],
        kind=kind,
        drivers=tuple(drivers),
        teams=tuple(team_of[d] for d in drivers),
        values=PERCENT * (seconds - reference) / reference,
        available_at=as_naive_utc(moment),
    )
