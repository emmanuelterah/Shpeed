"""Score one race forecast against what happened (P2-06)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import DNF
from f1sim.core.scoring.metrics import brier, log_loss, ranked_probability_score, spearman

PODIUM = 3
POINTS = 10

EVENTS: dict[str, int] = {"win": 1, "podium": PODIUM, "top10": POINTS}
"""Binary events scored with the Brier score and shown in calibration charts."""


@dataclass(frozen=True)
class RaceScore:
    """How well one forecast did. Every metric except Spearman is lower-is-better.

    Attributes:
        season: Season of the race.
        round: Round of the race.
        n_drivers: Drivers forecast.
        brier_win: Brier score of the win probabilities.
        brier_podium: Brier score of the podium probabilities.
        brier_top10: Brier score of the points (top 10) probabilities.
        log_loss: Mean negative log probability of each driver's actual outcome.
        rps: Ranked probability score over positions, with DNF as the last category.
        spearman: Rank correlation between expected and actual finishing order.
    """

    season: int
    round: int
    n_drivers: int
    brier_win: float
    brier_podium: float
    brier_top10: float
    log_loss: float
    rps: float
    spearman: float


@dataclass(frozen=True)
class EventForecasts:
    """The forecast probability and the outcome of one binary event, for every driver.

    Attributes:
        p: Forecast probabilities.
        happened: Whether the event happened.
    """

    p: NDArray[np.float64]
    happened: NDArray[np.bool_]


def score_race(
    season: int, round_: int, matrix: PositionMatrix, outcome: NDArray[np.int_]
) -> RaceScore:
    """Score one race forecast.

    Args:
        season: Season of the race.
        round_: Round of the race.
        matrix: The forecast.
        outcome: Each driver's finishing position among the forecast drivers, or ``DNF``.

    Returns:
        Every metric for this race.
    """
    n = len(matrix.drivers)
    category = np.where(outcome == DNF, n + 1, outcome)
    events = event_forecasts(matrix, outcome)
    finished = outcome != DNF
    expected = _expected_position(matrix)
    return RaceScore(
        season=season,
        round=round_,
        n_drivers=n,
        brier_win=brier(events["win"].p, events["win"].happened),
        brier_podium=brier(events["podium"].p, events["podium"].happened),
        brier_top10=brier(events["top10"].p, events["top10"].happened),
        log_loss=log_loss(matrix.categories(), category),
        rps=ranked_probability_score(matrix.categories(), category),
        spearman=spearman(expected[finished], outcome[finished].astype(np.float64)),
    )


def event_forecasts(matrix: PositionMatrix, outcome: NDArray[np.int_]) -> dict[str, EventForecasts]:
    """Pair each binary event's forecast with whether it happened.

    Args:
        matrix: The forecast.
        outcome: Each driver's finishing position, or ``DNF``.

    Returns:
        One ``EventForecasts`` per name in ``EVENTS``.
    """
    finished = outcome != DNF
    return {
        name: EventForecasts(p=matrix.p_top(k), happened=finished & (outcome <= k))
        for name, k in EVENTS.items()
    }


def _expected_position(matrix: PositionMatrix) -> NDArray[np.float64]:
    """Expected finishing position, given that the driver finishes."""
    n = len(matrix.drivers)
    positions = np.arange(1, n + 1, dtype=np.float64)
    finish = np.maximum(1.0 - matrix.p_dnf, 1e-12)
    expected: NDArray[np.float64] = (matrix.p_position @ positions) / finish
    return expected
