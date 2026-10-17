"""The Monte Carlo race simulator (P5-12): the Phase 5 model, forecasting one race per fit.

It composes the pieces built in Phase 3 and 4: the pace posterior gives each car a clean-lap pace
per run, the tyre model gives a degradation curve per run, the strategy chooser samples a stop
plan per car, and the fitted pass model resolves overtakes. Those are turned into the engine's
``LapStep`` pipeline and run once per simulated race, and the finishing orders are counted into a
position matrix. Everything is driven by one seeded generator, so the same snapshot and seed give
the same matrix (P5-09).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1sim.adapters.config_files import load_season_rules
from f1sim.adapters.models.common import N_SIMS
from f1sim.adapters.models.overtaking import OvertakingModel, fit_overtaking
from f1sim.adapters.models.pace import PaceContext, PacePredictor
from f1sim.adapters.models.pace_data import circuit_map, event_cluster
from f1sim.adapters.models.strategy import (
    DRY_COMPOUNDS,
    MAX_STOPS,
    StopPlan,
    StrategyChooser,
    StrategyGenerator,
    TeamTendencies,
    fit_team_tendencies,
)
from f1sim.adapters.models.tyre_priors import default_tyre_model
from f1sim.adapters.models.tyres import SampledLoss, TyreModel, fit_tyre_model
from f1sim.adapters.steps.dirty_air import DirtyAirStep
from f1sim.adapters.steps.lap_time import LapTimeStep
from f1sim.adapters.steps.overtake import OvertakeStep
from f1sim.adapters.steps.pit_stop import PitStopStep
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.reference import CircuitTable
from f1sim.core.domain.rules import SeasonRules
from f1sim.core.engine.grid import FULL_FUEL_KG, starting_state
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext, LapStep
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.simulate import simulate_race

_PlanKey = tuple[str, int, int]
"""Circuit, race laps and season: what a race's plan set depends on."""

MODEL_NAME = "simulator"
TABLES = ("results", "laps", "events", "pit_stops")

FALLBACK_LAP_S = 90.0
"""Reference lap time when a circuit has no timed laps, seconds."""
MIN_RACE_LAPS = 5
DEFAULT_TOTAL_LAPS = 57
"""Race distance in laps when the circuit length is unknown."""
ORDER_NOISE_JITTER = 1e-9
"""Tiny diagonal added before the Cholesky, so a singular covariance still factorises."""


@dataclass(frozen=True)
class _Draws:
    """The per-run inputs a race is simulated from.

    Attributes:
        pace_s: Clean-lap pace per car, seconds.
        loss: Sampled tyre-loss curves.
        start: Starting compound code per car.
        stops: Stop laps of shape ``(n_sims, n_cars, max_stops)``.
        compounds: Compound code fitted at each stop, same shape.
        pit_loss: Time lost at each stop, same shape as ``stops``, seconds.
        retired: Which cars retire, of shape ``(n_sims, n_cars)``.
    """

    pace_s: NDArray[np.float64]
    loss: SampledLoss
    start: NDArray[np.int_]
    stops: NDArray[np.int_]
    compounds: NDArray[np.int_]
    pit_loss: NDArray[np.float64]
    retired: NDArray[np.bool_]


@dataclass(frozen=True)
class SimulatorContext:
    """Everything the simulator needs besides its seed.

    Attributes:
        pace: The pace model's hyperparameters, reference tables and cache.
        circuits: Circuit lengths, for the race distance and the pace cluster.
        rules_dir: Folder holding the season-rules files.
        n_sims: Simulated races per forecast.
    """

    pace: PaceContext
    circuits: CircuitTable
    rules_dir: Path | None = None
    n_sims: int = N_SIMS


class SimulatorPredictor:  # pylint: disable=too-many-instance-attributes
    """Forecast a race by running the lap-by-lap engine many times (P5-12)."""

    name = MODEL_NAME

    def __init__(self, seed: int, context: SimulatorContext) -> None:
        """Create an unfitted simulator.

        Args:
            seed: Random seed for the simulated races (P5-09).
            context: Hyperparameters, reference tables and the race distance source.
        """
        self._rng = np.random.default_rng(seed)
        self._context = context
        self._pace = PacePredictor(seed, context.pace, n_sims=context.n_sims)
        self._laps = pd.DataFrame()
        self._circuits: dict[tuple[int, int], str] = {}
        self._reference: dict[str, float] = {}
        self._tyre: TyreModel | None = None
        self._tendencies: TeamTendencies | None = None
        self._overtaking: OvertakingModel | None = None
        self._plan_cache: dict[_PlanKey, StrategyChooser] = {}

    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Fit the pace, tyre, strategy and pass models up to the cutoff.

        Args:
            reader: The data store.
            snapshot: Snapshot to read.
            cutoff: Latest moment whose data may be used.
        """
        self._pace.fit(reader, snapshot=snapshot, cutoff=cutoff)
        laps = reader.read("laps", snapshot=snapshot, as_of=cutoff)
        events = reader.read("events", snapshot=snapshot, as_of=cutoff)
        pit_stops = reader.read("pit_stops", snapshot=snapshot, as_of=cutoff)
        self._laps = laps
        self._circuits = circuit_map(events)
        self._reference = _reference_laps(laps, self._circuits)
        try:
            self._tyre = fit_tyre_model(laps, events, pit_stops)
        except InvariantError:
            self._tyre = default_tyre_model()
        self._tendencies = fit_team_tendencies(pit_stops, laps)
        self._overtaking = fit_overtaking(laps, events)

    def predict(self, event: EventKey, entries: Sequence[RaceEntry]) -> PositionMatrix:
        """Simulate the race many times and count the finishing positions.

        Args:
            event: The race weekend.
            entries: The starters, with their grid positions.

        Returns:
            Finishing-position probabilities in ``entries`` order.

        Raises:
            InvariantError: If the simulator has not been fitted.
        """
        tyre = self._require_tyre()
        overtaking = self._require_overtaking()
        circuit = self._circuits.get((event.season, event.round), "")
        rules = load_season_rules(event.season, self._context.rules_dir)
        total = _total_laps(rules, circuit, self._context.circuits)
        reference = self._reference.get(circuit, FALLBACK_LAP_S)
        draws = self._draws(
            event,
            entries,
            rules=rules,
            total=total,
            reference=reference,
            circuit=circuit,
            tyre=tyre,
        )
        steps = _steps(
            draws.pace_s,
            draws.loss,
            tyre=tyre,
            total=total,
            stops=draws.stops,
            compounds=draws.compounds,
            pit_loss=draws.pit_loss,
            overtaking=overtaking,
            circuit=circuit,
        )
        ctx = LapContext(total, tyre.pit_loss_s(circuit), rules.overtake_aid)
        state = starting_state(
            entries,
            n_sims=self._context.n_sims,
            start_compounds=draws.start,
            fuel_kg=FULL_FUEL_KG,
            retired=draws.retired,
        )
        drivers = tuple(entry.driver_id for entry in entries)
        return simulate_race(drivers, state, steps, ctx, self._rng)

    def _draws(  # pylint: disable=too-many-arguments
        self,
        event: EventKey,
        entries: Sequence[RaceEntry],
        *,
        rules: SeasonRules,
        total: int,
        reference: float,
        circuit: str,
        tyre: TyreModel,
    ) -> "_Draws":
        n = len(entries)
        n_sims = self._context.n_sims
        loss = tyre.sample_loss(self._rng, n_sims, circuit=circuit)
        start, stops, stop_compounds = self._sample_plans(
            entries, rules=rules, total=total, reference=reference, circuit=circuit, loss=loss
        )
        return _Draws(
            pace_s=self._sample_pace(event, entries, circuit, reference),
            loss=loss,
            start=start,
            stops=stops,
            compounds=stop_compounds,
            pit_loss=self._pit_loss(entries, circuit, n_sims),
            retired=self._rng.random((n_sims, n)) < self._pace.p_dnf,
        )

    def _require_tyre(self) -> TyreModel:
        if self._tyre is None:
            raise InvariantError("the simulator must be fitted before it predicts")
        return self._tyre

    def _require_overtaking(self) -> OvertakingModel:
        if self._overtaking is None:
            raise InvariantError("the simulator must be fitted before it predicts")
        return self._overtaking

    def _sample_pace(
        self, event: EventKey, entries: Sequence[RaceEntry], circuit: str, reference: float
    ) -> NDArray[np.float64]:
        drivers = [entry.driver_id for entry in entries]
        teams = [entry.team_id for entry in entries]
        cluster = event_cluster(
            self._laps, (event.season, event.round), circuit, self._context.circuits
        )
        self._pace.filter.advance(event.season, event.round)
        mean, cov = self._pace.filter.race_pace(event.season, cluster, drivers, teams)
        n = len(entries)
        spread = cov + self._context.pace.hyper.order_noise**2 * np.eye(n)
        chol = np.linalg.cholesky(spread + ORDER_NOISE_JITTER * np.eye(n))
        draws = self._rng.standard_normal((self._context.n_sims, n))
        percent = mean[np.newaxis, :] + draws @ chol.T
        return (reference * (1.0 + percent / 100.0)).astype(np.float64)

    def _sample_plans(  # pylint: disable=too-many-locals
        self,
        entries: Sequence[RaceEntry],
        *,
        rules: SeasonRules,
        total: int,
        reference: float,
        circuit: str,
        loss: SampledLoss,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
        n, n_sims = len(entries), self._context.n_sims
        chooser = self._chooser(rules, total, reference, circuit)
        starts, plan_stops, plan_compounds = _plan_arrays(chooser.plans, loss)
        start = np.zeros((n_sims, n), dtype=np.int64)
        stops = np.zeros((n_sims, n, MAX_STOPS), dtype=np.int64)
        compounds = np.zeros((n_sims, n, MAX_STOPS), dtype=np.int64)
        for column, entry in enumerate(entries):
            probabilities = np.asarray(chooser.distribution(entry.team_id), dtype=np.float64)
            chosen = self._rng.choice(len(chooser.plans), size=n_sims, p=probabilities)
            start[:, column] = starts[chosen]
            stops[:, column, :] = plan_stops[chosen]
            compounds[:, column, :] = plan_compounds[chosen]
        return start, stops, compounds

    def _chooser(
        self, rules: SeasonRules, total: int, reference: float, circuit: str
    ) -> StrategyChooser:
        """The race's plans, generated once at the field-average stop cost.

        A team's stop-time deviation is a fraction of a second against a pit loss of about
        twenty, so it barely moves the optimum; generating per team cost most of the
        forecast's run time. Team habits enter through the chooser, and each team's own stop
        times are still sampled when its stops are executed.
        """
        key = (circuit, total, rules.season)
        cached = self._plan_cache.get(key)
        if cached is not None:
            return cached
        tyre = self._require_tyre()
        if self._tendencies is None:
            raise InvariantError("the simulator must be fitted before it predicts")
        try:
            generator = StrategyGenerator(
                tyre,
                rules,
                total_laps=total,
                base_lap_s=reference,
                circuit=circuit,
                compounds=_available_compounds(tyre, rules) or None,
            )
            plans = generator.plans()
        except InvariantError:
            plans = [_fallback_plan(tyre)]
        chooser = StrategyChooser(plans, self._tendencies)
        self._plan_cache[key] = chooser
        return chooser

    def _pit_loss(
        self, entries: Sequence[RaceEntry], circuit: str, n_sims: int
    ) -> NDArray[np.float64]:
        tyre = self._require_tyre()
        base = tyre.pit_loss_s(circuit)
        loss = np.empty((n_sims, len(entries), MAX_STOPS), dtype=np.float64)
        for column, entry in enumerate(entries):
            offset = tyre.stops.offset(self._rng, entry.team_id, n_sims * MAX_STOPS)
            loss[:, column, :] = offset.reshape(n_sims, MAX_STOPS)
        return np.maximum(base + loss, 0.0)


def _steps(  # pylint: disable=too-many-arguments
    pace_s: NDArray[np.float64],
    loss: SampledLoss,
    *,
    tyre: TyreModel,
    total: int,
    stops: NDArray[np.int_],
    compounds: NDArray[np.int_],
    pit_loss: NDArray[np.float64],
    overtaking: OvertakingModel,
    circuit: str,
) -> list[LapStep]:
    return [
        LapTimeStep(
            pace_s,
            loss,
            fuel_s_per_lap=tyre.fuel_s_per_lap,
            burn_kg_per_lap=FULL_FUEL_KG / total,
            reference_fuel_kg=FULL_FUEL_KG,
            compound_offset_s=np.array(
                [tyre.compound_offset(name) for name in loss.compounds], dtype=np.float64
            ),
        ),
        DirtyAirStep(),
        PitStopStep(stops, compounds, pit_loss),
        OvertakeStep(overtaking, circuit=circuit),
    ]


def _plan_arrays(
    plans: Sequence[StopPlan], loss: SampledLoss
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    count = len(plans)
    start = np.zeros(count, dtype=np.int64)
    stops = np.zeros((count, MAX_STOPS), dtype=np.int64)
    compounds = np.zeros((count, MAX_STOPS), dtype=np.int64)
    for row, plan in enumerate(plans):
        start[row] = loss.compound_index(plan.compounds[0])
        for index, lap in enumerate(plan.stop_laps):
            stops[row, index] = lap
            compounds[row, index] = loss.compound_index(plan.compounds[index + 1])
    return start, stops, compounds


def _available_compounds(tyre: TyreModel, rules: SeasonRules) -> list[str]:
    fitted = set(tyre.degradation.compounds)
    ordered = [compound for compound in DRY_COMPOUNDS if compound in fitted]
    return ordered[: rules.dry_compounds_per_event]


def _total_laps(rules: SeasonRules, circuit: str, circuits: CircuitTable) -> int:
    length = circuits.lengths_km.get(circuit)
    if length is None or length <= 0:
        return DEFAULT_TOTAL_LAPS
    return max(MIN_RACE_LAPS, round(rules.race_distance_km / length))


def _reference_laps(laps: pd.DataFrame, circuits: dict[tuple[int, int], str]) -> dict[str, float]:
    if laps.empty:
        return {}
    seasons = laps["season"].tolist()
    rounds = laps["round"].tolist()
    times = laps["lap_time_s"].tolist()
    buckets: dict[str, list[float]] = {}
    for season, round_, value in zip(seasons, rounds, times, strict=True):
        circuit = circuits.get((int(season), int(round_)))
        if circuit is None or value is None:
            continue
        moment = float(value)
        if not np.isfinite(moment):
            continue
        buckets.setdefault(circuit, []).append(moment)
    return {circuit: float(np.median(values)) for circuit, values in buckets.items()}


def _fallback_plan(tyre: TyreModel) -> StopPlan:
    compound = next(iter(tyre.degradation.compounds), "MEDIUM")
    return StopPlan((compound,), (), 0.0)
