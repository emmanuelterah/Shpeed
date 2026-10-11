"""Proper scoring rules for position forecasts (P2-06)."""

import numpy as np
from numpy.typing import NDArray


def brier(p: NDArray[np.float64], outcome: NDArray[np.bool_]) -> float:
    """Mean Brier score of binary forecasts ``p`` against outcomes. Lower is better."""
    return float(np.mean((p - outcome.astype(np.float64)) ** 2))


def ranked_probability_score(p_position: NDArray[np.float64], actual: NDArray[np.int_]) -> float:
    """Mean RPS over drivers. ``actual[i]`` is driver i's 1-based finishing position."""
    n = p_position.shape[1]
    cdf_pred = np.cumsum(p_position, axis=1, dtype=np.float64)
    positions = np.arange(1, n + 1, dtype=np.int64)
    cdf_obs = np.asarray(positions[np.newaxis, :] >= actual[:, np.newaxis], dtype=np.float64)
    diff = np.subtract(cdf_pred, cdf_obs)
    per_driver: NDArray[np.float64] = np.einsum("ij,ij->i", diff, diff) / (n - 1)
    return float(np.mean(per_driver))
