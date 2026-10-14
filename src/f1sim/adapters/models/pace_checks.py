"""Checks on the fitted pace model (P3-11): posterior predictive checks and calibration.

The posterior predictive check runs the filter through history and, for every race in the
held-out seasons, compares the forecast made *before* the race with what the race showed.
If the model's uncertainty is honest, about 90% of drivers land inside their 90% interval,
and the standardised errors have variance close to 1.

Race-day noise for ranking (``order_noise``) is not part of the pace model, so it is
chosen by backtesting a few values on the training seasons and keeping the best RPS.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from f1sim.adapters.models.pace import PaceContext, PacePredictor
from f1sim.adapters.models.pace_data import RACE_SESSIONS, SessionObs
from f1sim.adapters.models.pace_filter import PaceFilter
from f1sim.core.domain.entities import SnapshotId
from f1sim.core.errors import DataError
from f1sim.core.ports.predictor import ModelSpec, Predictor
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.backtest import walk_forward

Z90 = 1.6448536269514722
Z50 = 0.6744897501960817
COVERAGE90_RANGE = (0.82, 0.97)
COVERAGE50_RANGE = (0.38, 0.62)
CHI2_RATIO_RANGE = (0.6, 1.6)
ORDER_NOISE_GRID = (0.05, 0.1, 0.2, 0.3, 0.45, 0.65, 1.0)


@dataclass(frozen=True)
class PredictiveCheck:
    """Result of the posterior predictive check on held-out races.

    Attributes:
        sessions: Held-out race and sprint sessions checked.
        drivers: Driver forecasts checked.
        coverage50: Share of drivers inside their 50% interval.
        coverage90: Share of drivers inside their 90% interval.
        z_sd: Standard deviation of the standardised errors; 1 when calibrated.
        chi2_ratio: Total Mahalanobis distance over its degrees of freedom; 1 when
            calibrated, above 1 when the model is overconfident.
    """

    sessions: int
    drivers: int
    coverage50: float
    coverage90: float
    z_sd: float
    chi2_ratio: float

    @property
    def passed(self) -> bool:
        """Whether every statistic is inside its accepted range.

        Returns:
            True when the coverages and the chi-squared ratio are all in range.
        """
        return (
            self.drivers > 0
            and COVERAGE90_RANGE[0] <= self.coverage90 <= COVERAGE90_RANGE[1]
            and COVERAGE50_RANGE[0] <= self.coverage50 <= COVERAGE50_RANGE[1]
            and CHI2_RATIO_RANGE[0] <= self.chi2_ratio <= CHI2_RATIO_RANGE[1]
        )


def predictive_check(
    observations: Sequence[SessionObs], context: PaceContext, holdout: Sequence[int]
) -> PredictiveCheck:
    """Check one-step-ahead race forecasts in held-out seasons.

    Args:
        observations: Every session in calendar order, training seasons first.
        context: Hyperparameters and reference tables.
        holdout: Seasons whose races are checked. The filter still learns from each
            held-out race after forecasting it, as it would live.

    Returns:
        Coverage and dispersion statistics.

    Raises:
        DataError: If no held-out race was observed.
    """
    pace = PaceFilter(context.hyper, context.lineage, context.suppliers)
    z: list[float] = []
    distance, dof, sessions = 0.0, 0, 0
    for obs in observations:
        forecast = pace.update(obs)
        if obs.season in holdout and obs.session in RACE_SESSIONS:
            z += (forecast.residuals / forecast.sd).tolist()
            distance += forecast.mahalanobis
            dof += len(obs.drivers) - 1
            sessions += 1
    if not z:
        raise DataError(f"no race observations in held-out seasons {list(holdout)}")
    scores = np.abs(np.asarray(z))
    return PredictiveCheck(
        sessions=sessions,
        drivers=len(z),
        coverage50=float(np.mean(scores <= Z50)),
        coverage90=float(np.mean(scores <= Z90)),
        z_sd=float(np.std(z)),
        chi2_ratio=distance / dof,
    )


def calibrate_order_noise(
    reader: DataReader,
    *,
    snapshot: SnapshotId,
    seasons: Sequence[int],
    context: PaceContext,
    grid: Sequence[float] = ORDER_NOISE_GRID,
) -> tuple[float, dict[float, float]]:
    """Pick the race-day noise that gives the best backtest RPS.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        seasons: Seasons to backtest; use training seasons, not held-out ones.
        context: Hyperparameters and reference tables; ``order_noise`` is varied.
        grid: Values to try, in percent of lap time.

    Returns:
        The best value and the mean RPS of every value tried.
    """
    scores: dict[float, float] = {}
    for value in grid:
        trial = context.with_hyper(replace(context.hyper, order_noise=value))
        spec = ModelSpec(PacePredictor.name, _factory(trial), ("results", "laps", "events"))
        result = walk_forward(reader, snapshot=snapshot, seasons=seasons, model=spec)
        scores[value] = result.summary().get("rps", float("inf"))
    best = min(scores, key=lambda v: scores[v])
    return best, scores


def _factory(context: PaceContext) -> Callable[[int], Predictor]:
    def _build(seed: int) -> Predictor:
        return PacePredictor(seed, context)

    return _build
