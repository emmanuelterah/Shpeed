"""Proper scoring rules and calibration for position forecasts (P2-06, P2-07)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LOG_LOSS_FLOOR = 1e-9
"""Probabilities are floored here so a confident miss costs a large, finite penalty."""

MIN_POINTS_FOR_CORRELATION = 2


@dataclass(frozen=True)
class CalibrationBin:
    """Forecasts that fell in one probability range, and how often the event happened.

    Attributes:
        low: Lower edge of the bin (inclusive).
        high: Upper edge of the bin (exclusive, except the last bin).
        mean_forecast: Mean forecast probability in the bin.
        observed: Fraction of those forecasts whose event happened.
        count: Number of forecasts in the bin.
    """

    low: float
    high: float
    mean_forecast: float
    observed: float
    count: int


def brier(p: NDArray[np.float64], outcome: NDArray[np.bool_]) -> float:
    """Mean Brier score of binary forecasts; lower is better.

    Args:
        p: Forecast probabilities.
        outcome: What happened, aligned with ``p``.

    Returns:
        The mean squared difference between forecast and outcome, from 0 to 1.
    """
    return float(np.mean((p - outcome.astype(np.float64)) ** 2))


def ranked_probability_score(p_position: NDArray[np.float64], actual: NDArray[np.int_]) -> float:
    """Mean ranked probability score over drivers; lower is better.

    Unlike the Brier score, RPS gives credit for near misses: forecasting P2 for a driver
    who finished P3 scores better than forecasting P10.

    Args:
        p_position: Array of shape ``(n_drivers, n_categories)`` of ordered outcome
            probabilities, best outcome first.
        actual: Each driver's outcome category, from 1.

    Returns:
        The mean RPS across drivers, from 0 (perfect) to 1.
    """
    n = p_position.shape[1]
    cdf_pred = np.cumsum(p_position, axis=1, dtype=np.float64)
    positions = np.arange(1, n + 1, dtype=np.int64)
    cdf_obs = np.asarray(positions[np.newaxis, :] >= actual[:, np.newaxis], dtype=np.float64)
    diff = np.subtract(cdf_pred, cdf_obs)
    per_driver: NDArray[np.float64] = np.einsum("ij,ij->i", diff, diff) / (n - 1)
    return float(np.mean(per_driver))


def log_loss(p_categories: NDArray[np.float64], actual: NDArray[np.int_]) -> float:
    """Mean negative log probability of what actually happened; lower is better.

    Args:
        p_categories: Array of shape ``(n_drivers, n_categories)``.
        actual: Each driver's outcome category, from 1.

    Returns:
        The mean of ``-ln p(actual)`` across drivers, with probabilities floored at
        ``LOG_LOSS_FLOOR``.
    """
    rows: NDArray[np.int_] = np.arange(p_categories.shape[0], dtype=np.int64)
    chosen = p_categories[rows, actual - 1]
    return float(np.mean(-np.log(np.maximum(chosen, LOG_LOSS_FLOOR))))


def spearman(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    """Spearman rank correlation between two orderings.

    Args:
        predicted: Predicted scores, such as expected finishing position.
        actual: Actual values, aligned with ``predicted``.

    Returns:
        The correlation from -1 to 1, or 0 when either input has no variation or fewer
        than two values.
    """
    if len(predicted) < MIN_POINTS_FOR_CORRELATION:
        return 0.0
    a = _ranks(predicted)
    b = _ranks(actual)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def calibration_bins(
    p: NDArray[np.float64], outcome: NDArray[np.bool_], n_bins: int = 10
) -> list[CalibrationBin]:
    """Group forecasts by probability and compare each group with what happened.

    A well-calibrated model has ``observed`` close to ``mean_forecast`` in every bin.

    Args:
        p: Forecast probabilities.
        outcome: What happened, aligned with ``p``.
        n_bins: Number of equal-width bins between 0 and 1.

    Returns:
        One entry per non-empty bin, lowest first.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    hits = outcome.astype(np.float64)
    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        if count:
            bins.append(
                CalibrationBin(
                    low=float(edges[b]),
                    high=float(edges[b + 1]),
                    mean_forecast=float(p[mask].mean()),
                    observed=float(hits[mask].mean()),
                    count=count,
                )
            )
    return bins


def _ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average ranks (ties share the mean rank), from 1."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks
