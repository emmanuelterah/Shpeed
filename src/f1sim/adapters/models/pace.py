"""The pace predictor (P3-09): forecast a race from each car and driver's estimated pace.

``fit`` turns every session up to the cutoff into pace observations and runs the Kalman
filter over them. With a fitted-state store, it first looks for the posterior saved after
the last completed race weekend; if the data behind it is unchanged, only the sessions
since then are processed, so a mid-weekend update (after practice, sprint or qualifying)
takes a moment rather than a full refit. ``predict`` ranks drivers by sampled race pace.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np
import pandas as pd

from f1sim.adapters.models.common import N_SIMS, dnf_rate, simulate
from f1sim.adapters.models.pace_corrections import CircuitHistory
from f1sim.adapters.models.pace_data import (
    ObservationCache,
    SessionObs,
    build_observations,
    circuit_map,
    event_cluster,
)
from f1sim.adapters.models.pace_filter import PaceFilter, PaceHyper, PaceState
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.reference import CircuitTable, EngineSuppliers
from f1sim.core.ports.state import FittedStateStore, StateKey
from f1sim.core.ports.store import DataReader

MODEL_NAME = "pace"
MODEL_VERSION = "2"
"""Bump when the observation or filter code changes, so old saved states are not reused."""

TABLES = ("results", "laps", "events")
_Window = tuple[tuple[int, int] | None, tuple[int, int] | None]
FINGERPRINT_COLUMNS = ("season", "round", "session", "driver_id", "lap_number", "lap_time_s")


@dataclass(frozen=True)
class PaceContext:
    """Everything a pace predictor needs besides its seed.

    Attributes:
        hyper: Hyperparameters.
        lineage: Team lineage table.
        suppliers: Engine supplier table.
        circuits: Circuit lengths and speed bands.
        cache: Session results shared across the races of one backtest.
        state_store: Where to save and find fitted states; None disables it.
    """

    hyper: PaceHyper
    lineage: TeamLineage
    suppliers: EngineSuppliers
    circuits: CircuitTable
    cache: ObservationCache = field(default_factory=ObservationCache)
    state_store: FittedStateStore | None = None

    @property
    def version(self) -> str:
        """Model version, including a digest of the hyperparameters.

        Returns:
            Such as ``1-3fa2c9d1``.
        """
        digest = hashlib.sha256(json.dumps(self.hyper.to_dict(), sort_keys=True).encode())
        return f"{MODEL_VERSION}-{digest.hexdigest()[:8]}"

    def with_hyper(self, hyper: PaceHyper) -> "PaceContext":
        """Copy with other hyperparameters, sharing the observation cache.

        Args:
            hyper: The new hyperparameters.

        Returns:
            The new context.
        """
        return replace(self, hyper=hyper)


@dataclass(frozen=True)
class FitReport:
    """What the last ``fit`` did.

    Attributes:
        sessions: Sessions processed by the filter in this fit.
        resumed_from: The weekend whose saved state was reused, or None.
        saved: The weekend whose state was saved, or None.
    """

    sessions: int
    resumed_from: EventKey | None
    saved: EventKey | None


class PacePredictor:
    """Ranks drivers by race pace sampled from the pace model's posterior."""

    name = MODEL_NAME

    def __init__(self, seed: int, context: PaceContext, *, n_sims: int = N_SIMS) -> None:
        """Create an unfitted predictor.

        Args:
            seed: Random seed for the simulated races.
            context: Hyperparameters, reference tables, cache and state store.
            n_sims: Simulated races per forecast.
        """
        self._rng = np.random.default_rng(seed)
        self._context = context
        self._n_sims = n_sims
        self.filter = PaceFilter(context.hyper, context.lineage, context.suppliers)
        self.p_dnf = 0.0
        self.observations: list[SessionObs] = []
        self.report = FitReport(0, None, None)
        self._laps = pd.DataFrame()
        self._circuits: dict[tuple[int, int], str] = {}

    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Update the pace posterior with every session up to the cutoff.

        Args:
            reader: The data store.
            snapshot: Snapshot to read.
            cutoff: Latest moment whose data may be used.
        """
        laps = reader.read("laps", snapshot=snapshot, as_of=cutoff)
        events = reader.read("events", snapshot=snapshot, as_of=cutoff)
        self.p_dnf = dnf_rate(reader.read("results", snapshot=snapshot, as_of=cutoff))
        self._laps = laps
        self._circuits = circuit_map(events)
        completed = _last_race_weekend(laps)
        history, resumed = self._resume(laps, events, snapshot, completed)
        before = (completed.season, completed.round) if completed else None
        observe = self._observer(laps, events, snapshot, history)
        early: list[SessionObs] = [] if resumed or before is None else observe((None, before))
        late = observe((before, None))
        for obs in early:
            self.filter.update(obs)
        saved = None
        if early and completed and self._context.state_store is not None:
            self._save(completed, snapshot, history, _fingerprint(laps, completed))
            saved = completed
        for obs in late:
            self.filter.update(obs)
        self.observations = early + late
        self.report = FitReport(len(early) + len(late), completed if resumed else None, saved)

    def predict(self, event: EventKey, entries: Sequence[RaceEntry]) -> PositionMatrix:
        """Forecast a race by ranking sampled race pace plus race-day noise.

        Args:
            event: The race weekend.
            entries: The starters.

        Returns:
            Finishing-position probabilities in ``entries`` order.
        """
        circuit = self._circuits.get((event.season, event.round), "")
        cluster = event_cluster(
            self._laps, (event.season, event.round), circuit, self._context.circuits
        )
        self.filter.advance(event.season, event.round)
        mean, cov = self.filter.race_pace(
            event.season, cluster, [e.driver_id for e in entries], [e.team_id for e in entries]
        )
        n = len(entries)
        spread = cov + self._context.hyper.order_noise**2 * np.eye(n)
        chol = np.linalg.cholesky(spread + 1e-9 * np.eye(n))
        noise = self._rng.standard_normal((self._n_sims, n)) @ chol.T
        return simulate(entries, -mean, -noise, self.p_dnf, self._rng)

    def _observer(
        self,
        laps: pd.DataFrame,
        events: pd.DataFrame,
        snapshot: SnapshotId,
        history: CircuitHistory,
    ) -> Callable[[_Window], list[SessionObs]]:
        context = self._context

        def _run(window: _Window) -> list[SessionObs]:
            return build_observations(
                laps,
                events,
                context.circuits,
                cache=context.cache,
                store_key=(id(context.cache), int(snapshot)),
                history=history,
                window=window,
            )

        return _run

    def _resume(
        self,
        laps: pd.DataFrame,
        events: pd.DataFrame,
        snapshot: SnapshotId,
        completed: EventKey | None,
    ) -> tuple[CircuitHistory, bool]:
        """Restore the posterior saved after ``completed``, if the data is unchanged."""
        del events
        store = self._context.state_store
        if store is None or completed is None:
            return CircuitHistory(), False
        found = store.load_latest(MODEL_NAME, self._context.version, completed, snapshot)
        if found is None:
            return CircuitHistory(), False
        _, saved = found
        if saved.get("fingerprint") != _fingerprint(laps, completed):
            return CircuitHistory(), False
        self.filter = PaceFilter(
            self._context.hyper,
            self._context.lineage,
            self._context.suppliers,
            PaceState.from_dict(saved["pace_state"]),
        )
        return CircuitHistory.from_dict(saved["history"]), True

    def _save(
        self, event: EventKey, snapshot: SnapshotId, history: CircuitHistory, fingerprint: str
    ) -> None:
        store = self._context.state_store
        if store is None:
            return
        key = StateKey(MODEL_NAME, self._context.version, event, snapshot)
        store.save(
            key,
            {
                "fingerprint": fingerprint,
                "pace_state": self.filter.state.to_dict(),
                "history": history.to_dict(),
                "hyper": self._context.hyper.to_dict(),
            },
        )


def _last_race_weekend(laps: pd.DataFrame) -> EventKey | None:
    """The latest weekend whose race laps are in the data."""
    races = laps.loc[laps["session"] == "R", ["season", "round"]]
    if races.empty:
        return None
    pairs = sorted({(int(s), int(r)) for s, r in races.itertuples(index=False, name=None)})
    return EventKey(*pairs[-1])


def _fingerprint(laps: pd.DataFrame, until: EventKey) -> str:
    """A digest of every lap up to a weekend; it changes if any of that data changes."""
    key = laps["season"].astype(int) * 100 + laps["round"].astype(int)
    rows = laps.loc[key <= until.season * 100 + until.round, list(FINGERPRINT_COLUMNS)]
    rows = rows.sort_values(list(FINGERPRINT_COLUMNS[:5])).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(rows.astype(str), index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()
