"""Proper scoring rules for position forecasts (P2-06)."""

import numpy as np
from numpy.typing import NDArray


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
        p_position: Array of shape ``(n_drivers, n_positions)`` of position probabilities.
        actual: Each driver's finishing position, from 1.

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
