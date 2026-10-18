"""Walk-forward backtest (P2-04): for each past race, fit up to its cutoff and predict it.

Each race is forecast exactly as it would have been live. The model is fitted through a
``CutoffReader``, so it cannot see the race or anything after the cutoff, and the forecast
is scored against the official result. The cutoff and grid rules are in ADR-0005.

Races are independent, so they run in parallel threads; every race gets its own seed
derived from the base seed, so results do not depend on scheduling.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import encode_outcome
from f1sim.core.errors import InvariantError
from f1sim.core.ports.predictor import ModelSpec, PredictorFactory
from f1sim.core.ports.store import DataReader
from f1sim.core.scoring.metrics import CalibrationBin, calibration_bins
from f1sim.core.scoring.race_score import EVENTS, RaceScore, event_forecasts, score_race
from f1sim.core.usecases.planning import RacePlan, SkippedRace, plan_races
from f1sim.core.usecases.readers import CutoffReader, FrameReader

METRICS = ("brier_win", "brier_podium", "brier_top10", "log_loss", "rps", "spearman")


@dataclass(frozen=True)
class BacktestResult:
    """Scores for every race a model forecast.

    Attributes:
        model: Registry name of the model.
        seasons: Seasons covered.
        snapshot: Data snapshot the backtest read.
        races: One score per forecast race, in calendar order.
        skipped: Races that could not be forecast.
        calibration: Calibration bins per binary event (``win``, ``podium``, ``top10``).
    """

    model: str
    seasons: tuple[int, ...]
    snapshot: SnapshotId
    races: list[RaceScore]
    skipped: list[SkippedRace] = field(default_factory=list[SkippedRace])
    calibration: dict[str, list[CalibrationBin]] = field(
        default_factory=dict[str, list[CalibrationBin]]
    )

    def summary(self) -> dict[str, float]:
        """Average every metric over the forecast races, each race counting equally.

        Returns:
            Metric name to mean value, plus ``races`` (the count); empty if no races.
        """
        if not self.races:
            return {}
        means = {m: float(np.mean([getattr(r, m) for r in self.races])) for m in METRICS}
        return {**means, "races": float(len(self.races))}


def walk_forward(
    reader: DataReader,
    *,
    snapshot: SnapshotId,
    seasons: Sequence[int],
    model: ModelSpec,
    seed: int = 0,
    workers: int = 4,
) -> BacktestResult:
    """Backtest one model over whole seasons, race by race.

    Args:
        reader: The data store; each table the model needs is read once.
        snapshot: Snapshot to read.
        seasons: Seasons to forecast.
        model: The model; its factory is called once per race.
        seed: Base seed; each race derives its own from it.
        workers: Races forecast in parallel.

    Returns:
        Scores for every forecast race, plus calibration bins.

    Raises:
        LeakageError: If the model reads past a race's cutoff.
        InvariantError: If the model returns rows that do not match the entries.
    """
    tables = tuple(dict.fromkeys(("results", "laps", *model.tables)))
    frames = FrameReader.load(reader, snapshot, tables)
    plans, skipped = plan_races(frames, snapshot, seasons)
    run_one = _race_runner(frames, snapshot, model.factory, seed)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = list(pool.map(run_one, plans))
    scores = [score for score, _ in outcomes]
    return BacktestResult(
        model=model.name,
        seasons=tuple(seasons),
        snapshot=snapshot,
        races=scores,
        skipped=skipped,
        calibration=_calibration([pair for _, pair in outcomes]),
    )


def race_seed(base: int, event: EventKey) -> int:
    """Derive one race's seed, so every race is reproducible on its own.

    Args:
        base: The backtest's base seed.
        event: The race.

    Returns:
        A seed that differs for every race and base seed.
    """
    return base * 1_000_003 + event.season * 100 + event.round


_Forecast = tuple[PositionMatrix, NDArray[np.int_]]


def _race_runner(
    frames: FrameReader, snapshot: SnapshotId, factory: PredictorFactory, seed: int
) -> Callable[[RacePlan], tuple[RaceScore, _Forecast]]:
    def _run(plan: RacePlan) -> tuple[RaceScore, _Forecast]:
        predictor = factory(race_seed(seed, plan.event))
        predictor.fit(CutoffReader(frames, plan.cutoff), snapshot=snapshot, cutoff=plan.cutoff)
        matrix = predictor.predict(plan.event, plan.entries)
        expected = tuple(e.driver_id for e in plan.entries)
        if matrix.drivers != expected:
            raise InvariantError(f"{predictor.name} returned drivers out of entry order")
        outcome = encode_outcome(plan.entries, plan.classified)
        return score_race(plan.event.season, plan.event.round, matrix, outcome), (matrix, outcome)

    return _run


def _calibration(forecasts: Sequence[_Forecast]) -> dict[str, list[CalibrationBin]]:
    bins: dict[str, list[CalibrationBin]] = {}
    for name in EVENTS:
        pairs = [event_forecasts(matrix, outcome)[name] for matrix, outcome in forecasts]
        if pairs:
            p = np.concatenate([e.p for e in pairs])
            happened = np.concatenate([e.happened for e in pairs])
            bins[name] = calibration_bins(p, happened)
    return bins
